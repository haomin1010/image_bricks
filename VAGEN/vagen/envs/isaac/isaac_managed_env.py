"""
Isaac Managed Environment — GymImageEnv proxy backed by the EnvManager system.
This environment is a drop-in replacement for :class:`BrickIsaac` in the VAGEN
agent loop.  Instead of holding a local simulator, it delegates all
reset / step / render calls to a shared :class:`IsaacSkillExecutor` via the
per-worker :class:`EnvCoordinator`.
From the agent loop's perspective this behaves identically to any other
``GymImageEnv``:  ``reset(seed)`` returns multi-view images + info, and
``step(action_str)`` parses the LLM output, executes the high-level goal
remotely, and returns the formatted observation.
Usage
-----
Register this class in the env_registry config::
    env_registry:
      BrickIsaacFullView: vagen.envs.isaac_full_view.isaac_managed_env.IsaacManagedEnv
The ``env_config`` dict in the dataset YAML should contain at least:
- ``num_total_envs`` (int): Total sub-envs in the Isaac DirectVectorEnv.
- Standard BrickIsaac fields (``n_views``, ``image_size``, ``max_steps``, …).
"""

from __future__ import annotations

import logging
import ray
import asyncio
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..gym_image_env import GymImageEnv
from .utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
)
from .reward_manager import IsaacRewardConfig, IsaacRewardManager, PlacementRewardResult
from .task_spec import BrickPosition, TaskSpec, load_task_spec, scan_ground_truth_entries
from .termination_manager import IsaacTerminationConfig, IsaacTerminationManager, TerminationStatus
from .utils.utils import parse_response

logger = logging.getLogger(__name__)
DEFAULT_GROUND_TRUTH_ROOT = str(
    Path(__file__).resolve().parents[4]
    / "IsaacLab"
    / "scripts"
    / "data_gen"
    / "convex_json_batch"
)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IsaacManagedEnvConfig:
    """Configuration for the Isaac-managed environment."""

    # Executor / manager settings
    num_total_envs: int = 64  # total sub-envs in the DirectVectorEnv

    # Observation
    n_views: int = 3
    image_size: Tuple[int, int] = (224, 224)

    # Step limits
    max_steps: int = 200

    # Prompt
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True

    # Reward shaping
    format_reward: float = 0.1
    success_reward: float = 1.0
    correct_placement_reward: Optional[float] = None
    floating_placement_penalty: float = -10.0
    non_candidate_penalty: float = -5.0
    max_attempts_factor: float = 1.5

    # Dataset
    dataset_root: str = "/mnt/data/image_bricks/assets/snapshots"
    ground_truth_root: str = DEFAULT_GROUND_TRUTH_ROOT
    collapse_mock_after_attempt: int = -1

    def __post_init__(self) -> None:
        self.num_total_envs = int(self.num_total_envs)
        self.n_cameras = int(self.n_cameras)
        self.image_size = tuple(int(value) for value in self.image_size)
        self.max_steps = int(self.max_steps)
        self.use_example_in_sys_prompt = _coerce_bool(self.use_example_in_sys_prompt)
        self.format_reward = float(self.format_reward)
        self.success_reward = float(self.success_reward)
        if self.correct_placement_reward in ("", None):
            self.correct_placement_reward = None
        elif self.correct_placement_reward is not None:
            self.correct_placement_reward = float(self.correct_placement_reward)
        self.floating_placement_penalty = float(self.floating_placement_penalty)
        self.non_candidate_penalty = float(self.non_candidate_penalty)
        self.max_attempts_factor = float(self.max_attempts_factor)
        if not self.ground_truth_root:
            self.ground_truth_root = DEFAULT_GROUND_TRUTH_ROOT
        self.collapse_mock_after_attempt = int(self.collapse_mock_after_attempt)


_CONFIG_FIELDS = {f.name for f in fields(IsaacManagedEnvConfig)}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IsaacManagedEnv(GymImageEnv):
    """
    GymImageEnv proxy backed by the IsaacEnvServer Ray Actor.
    Each agent loop creates one ``IsaacManagedEnv`` instance. On ``reset()``
    a sub-env ID is allocated from the global server; on ``close()`` it is
    automatically returned when the instance is destroyed or explicitly closed.
    """

    _server_handle = None
    _server_lock = asyncio.Lock()

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Filter to known config fields
        known = {k: v for k, v in env_config.items() if k in _CONFIG_FIELDS}
        self.config = IsaacManagedEnvConfig(**known)

        # Instance state
        self._sub_env_id: Optional[int] = None
        self.total_reward: float = 0.0
        self.steps_taken: int = 0
        self.trajectory: List[Dict[str, Any]] = []
        self._latest_scene_images: List[Image.Image] = []
        self._current_task_spec: TaskSpec = TaskSpec.empty()
        self._current_ground_truth_path: Optional[Path] = None

        reward_config = IsaacRewardConfig(
            format_reward=self.config.format_reward,
            correct_placement_reward=self._get_correct_placement_reward(),
            floating_placement_penalty=self.config.floating_placement_penalty,
            non_candidate_penalty=self.config.non_candidate_penalty,
        )
        self.reward_manager = IsaacRewardManager(reward_config)
        self.termination_manager = IsaacTerminationManager(
            IsaacTerminationConfig(
                max_attempts_factor=self.config.max_attempts_factor,
                collapse_mock_after_attempt=self.config.collapse_mock_after_attempt,
            )
        )

        # Scan dataset directory and cache valid entries on startup
        self._dataset_entries: List[Dict] = self._scan_dataset(self.config.dataset_root)
        self._ground_truth_entries: List[Path] = scan_ground_truth_entries(self.config.ground_truth_root)
        self._ground_truth_by_stem = {path.stem: path for path in self._ground_truth_entries}
        if not self._dataset_entries:
            logger.warning("Dataset is empty or not found at: %s", self.config.dataset_root)
        else:
            logger.info("Loaded %d dataset entries from %s", len(self._dataset_entries), self.config.dataset_root)
        if not self._ground_truth_entries:
            logger.warning("Ground truth JSONs are empty or not found at: %s", self.config.ground_truth_root)
        else:
            logger.info("Loaded %d ground truth JSON files from %s", len(self._ground_truth_entries), self.config.ground_truth_root)

    async def _get_server(self):
        """Lazily obtain the per-worker server handle singleton."""
        if IsaacManagedEnv._server_handle is not None:
            return IsaacManagedEnv._server_handle

        async with IsaacManagedEnv._server_lock:
            if IsaacManagedEnv._server_handle is not None:
                return IsaacManagedEnv._server_handle
            
            try:
                IsaacManagedEnv._server_handle = ray.get_actor("IsaacEnvServer")
                logger.info("Connected to existing IsaacEnvServer actor.")
            except ValueError:
                logger.info("IsaacEnvServer actor not found. It should be started by the main script.")
                raise RuntimeError("IsaacEnvServer actor not found. Ensure it is started as a detached actor.")
            
            return IsaacManagedEnv._server_handle

    # ------------------------------------------------------------------
    # GymImageEnv interface
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        """Return the system-level prompt observation."""
        return {"obs_str": self._build_system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Allocate a sub-env ID (on first call) and reset it on the Isaac side.
        """
        server = await self._get_server()

        # Allocate once; reuse the same sub-env ID across resets
        if self._sub_env_id is None:
            self._sub_env_id = await server.allocate_env_id.remote()

        # Reset physics side (slot state / cube positions) but ignore its images
        try:
            response = await asyncio.wait_for(
                server.remote_reset.remote(self._sub_env_id, seed), timeout=120.0
            )
            env_info = response.get("info", {})
        except Exception as e:
            logger.error("Failed to reset Isaac environment (timeout or crash): %s", e)
            env_info = {}

        self.total_reward = 0.0
        self.steps_taken = 0
        self.trajectory = []

        # Load pre-rendered images from dataset (deterministic via seed)
        all_images = self._load_dataset_images(seed)
        # Cache for use in step() query actions (cam 0=top, 1=front, 2=side, 3=iso, 4=iso2)
        self._dataset_images_cache = all_images
        self._latest_scene_images = list(all_images)

        self._current_ground_truth_path = self._select_ground_truth_path(seed)
        self._current_task_spec = self._load_current_task_spec()
        self.reward_manager.reset(self._current_task_spec)
        self.termination_manager.reset(self._current_task_spec)

        # Build target description from dataset JSON
        target_desc = self._load_target_description(seed)
        env_info["target_description"] = target_desc

        logger.info(
            "reset: seed=%d dataset_index=%d gt=%s images=%d max_attempts=%d",
            seed,
            seed % max(len(self._dataset_entries), 1),
            self._current_ground_truth_path.name if self._current_ground_truth_path else "none",
            len(all_images),
            self.termination_manager.max_attempts,
        )

        # Build initial observation: ALL dataset views so VLM understands the target from every angle
        cam_labels = ["Top view", "Front view", "Side view", "Iso view", "Iso2 view"]
        img_phs = "\n".join(self.config.image_placeholder for _ in all_images)
        obs_text = (target_desc + "\n" if target_desc else "") + init_observation_template(
            img_placeholders=img_phs,
            camera_labels=cam_labels[: len(all_images)],
        )
        obs = self._make_multi_image_obs(obs_text, all_images)
        return obs, env_info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Parse the LLM's coordinate, execute the high-level goal remotely,
        and return the formatted observation.
        Args:
            action_str: Raw LLM output string.
        Returns:
            ``(obs, reward, done, info)``
        """
        self.steps_taken += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        coordinate = parsed.get("coordinate")
        is_submit = parsed.get("is_submit", False)
        format_correct = parsed.get("format_correct", False)
        reward += self.reward_manager.format_reward(format_correct)

        placement_result: Optional[PlacementRewardResult] = None
        termination_status: Optional[TerminationStatus] = None
        isaac_info: Dict[str, Any] = {}
        external_done = False
        query_cameras = parsed.get("query_cameras")

        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct and (coordinate is not None or is_submit),
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
                "termination_reason": None,
                "task_completed": False,
                "collapse_detected": False,
            },
        }

        if coordinate is not None:
            # ----------------------------------------------------------
            # Place action — execute physics then evaluate reward/termination
            # ----------------------------------------------------------
            goal = {"x": coordinate["x"], "y": coordinate["y"], "z": coordinate["z"]}
            server = await self._get_server()
            try:
                step_response = await asyncio.wait_for(
                    server.remote_step.remote(self._sub_env_id, goal), timeout=300.0
                )
                isaac_info = dict(step_response.get("info", {}) or {})
                external_done = bool(step_response.get("done", False))
            except Exception as e:
                logger.error("Failed to step Isaac (timeout or crash): %s", e)
                isaac_info = {"timeout": True}
                external_done = True

            placement_result = self.reward_manager.evaluate_placement(BrickPosition.from_mapping(goal))
            reward += placement_result.reward_delta
            termination_status = self.termination_manager.evaluate(
                placement_attempted=True,
                submit_requested=False,
                isaac_info=isaac_info,
                task_completed=self.reward_manager.task_completed,
                external_done=external_done,
            )

            isaac_images = await self._render_env_images()
            obs_text = self._build_placement_obs_text(coordinate, placement_result, termination_status)

            done = termination_status.done
            metrics["turn_metrics"]["action_is_effective"] = True
            metrics["traj_metrics"]["task_completed"] = termination_status.task_completed
            metrics["traj_metrics"]["collapse_detected"] = termination_status.collapsed
            metrics["traj_metrics"]["termination_reason"] = termination_status.reason

            obs = self._make_multi_image_obs(obs_text, isaac_images, action_str=action_str)
            info.update(
                {
                    "timeout": bool(isaac_info.get("timeout", False)),
                    "placement_outcome": placement_result.outcome,
                    "placement_feedback": placement_result.feedback,
                }
            )

        elif is_submit:
            # ----------------------------------------------------------
            # Submit action — terminate and report whether the current structure matches the target
            # ----------------------------------------------------------
            goal = {"type": "submit"}
            server = await self._get_server()
            try:
                step_response = await asyncio.wait_for(
                    server.remote_step.remote(self._sub_env_id, goal), timeout=300.0
                )
                isaac_info = dict(step_response.get("info", {}) or {})
                external_done = bool(step_response.get("done", False))
            except Exception as e:
                logger.error("Failed to submit Isaac (timeout or crash): %s", e)
                isaac_info = {"timeout": True}
                external_done = True

            termination_status = self.termination_manager.evaluate(
                placement_attempted=False,
                submit_requested=True,
                isaac_info=isaac_info,
                task_completed=self.reward_manager.task_completed,
                external_done=external_done,
            )

            isaac_images = await self._render_env_images()
            obs_text = self._build_submit_obs_text(termination_status)

            done = termination_status.done
            metrics["turn_metrics"]["action_is_effective"] = True
            metrics["traj_metrics"]["success"] = termination_status.success
            metrics["traj_metrics"]["task_completed"] = termination_status.task_completed
            metrics["traj_metrics"]["collapse_detected"] = termination_status.collapsed
            metrics["traj_metrics"]["termination_reason"] = termination_status.reason

            obs = self._make_multi_image_obs(obs_text, isaac_images, action_str=action_str)
            info.update({"timeout": bool(isaac_info.get("timeout", False))})

        elif query_cameras is not None:
            # ----------------------------------------------------------
            # Query action — keep the current scene unchanged and return selected views
            # ----------------------------------------------------------
            scene_images = await self._render_env_images()
            selected_ids = [cam_id for cam_id in query_cameras if 0 <= cam_id < len(scene_images)]
            selected_images = [scene_images[cam_id] for cam_id in selected_ids]
            if not selected_images:
                selected_images = self._make_fallback_images(count=1, color=(30, 30, 30))
                selected_ids = []

            label_lines = [
                f"Camera {cam_id}: {self.config.image_placeholder}" for cam_id in selected_ids
            ] or [f"Camera unavailable: {self.config.image_placeholder}"]
            obs_text = (
                "[System]: Query result.\n"
                + "\n".join(label_lines)
                + "\nYou may place the next cube or submit when done."
            )
            obs = self._make_multi_image_obs(obs_text, selected_images, action_str=action_str)

        else:
            # ----------------------------------------------------------
            # Parse failure — use blank image, no server call
            # ----------------------------------------------------------
            cam0_images = self._make_fallback_images(count=1, color=(30, 30, 30))
            msg = (
                'Could not parse your action. Valid formats:\n'
                '  Place a brick: {"x": 2, "y": 3, "z": 0}\n'
                '  Submit: submit'
            )
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=msg,
                    img_placeholder=self.config.image_placeholder,
                ),
                cam0_images,
                action_str=action_str,
            )

        # Step limit
        if self.steps_taken >= self.config.max_steps and not done:
            done = True
            if termination_status is None:
                termination_status = self.termination_manager.evaluate(
                    placement_attempted=False,
                    submit_requested=False,
                    isaac_info=isaac_info,
                    task_completed=self.reward_manager.task_completed,
                    external_done=False,
                )
            termination_status = TerminationStatus(
                done=True,
                reason="max_steps_guard",
                success=False,
                collapsed=termination_status.collapsed,
                submitted=termination_status.submitted,
                reached_max_attempts=termination_status.reached_max_attempts,
                task_completed=termination_status.task_completed,
                placement_attempts=termination_status.placement_attempts,
                max_attempts=termination_status.max_attempts,
            )
            metrics["traj_metrics"]["termination_reason"] = termination_status.reason

        if termination_status is None:
            termination_status = self.termination_manager.evaluate(
                placement_attempted=False,
                submit_requested=False,
                isaac_info=isaac_info,
                task_completed=self.reward_manager.task_completed,
                external_done=False,
            )

        metrics["traj_metrics"]["success"] = termination_status.success
        metrics["traj_metrics"]["task_completed"] = termination_status.task_completed
        metrics["traj_metrics"]["collapse_detected"] = termination_status.collapsed
        metrics["traj_metrics"]["termination_reason"] = termination_status.reason
        info["metrics"] = metrics
        info["success"] = termination_status.success
        info["termination"] = {
            "reason": termination_status.reason,
            "collapsed": termination_status.collapsed,
            "submitted": termination_status.submitted,
            "task_completed": termination_status.task_completed,
            "placement_attempts": termination_status.placement_attempts,
            "max_attempts": termination_status.max_attempts,
        }
        info["reward_breakdown"] = {
            "format_reward": self.reward_manager.format_reward(format_correct),
            "placement_reward": 0.0 if placement_result is None else placement_result.reward_delta,
        }
        self.total_reward += reward
        info["total_reward"] = self.total_reward
        info["remaining_target_blocks"] = len(self.reward_manager.remaining_target_positions())

        # Record step trajectory
        self.trajectory.append(
            {
                "step_idx": self.steps_taken,
                "coordinate": coordinate,
                "query_cameras": query_cameras,
                "is_submit": is_submit,
                "reward": reward,
                "success": info["success"],
                "raw_action": parsed.get("action_content", ""),
                "thought": parsed.get("think_content", ""),
                "placement_outcome": None if placement_result is None else placement_result.outcome,
                "termination_reason": termination_status.reason,
                "placement_attempts": termination_status.placement_attempts,
                "total_reward": self.total_reward,
            }
        )

        if done:
            info["trajectory"] = self.trajectory

        return obs, reward, done, info

    async def close(self) -> None:
        """Release the sub-env ID back to the global pool."""
        if self._sub_env_id is not None:
            try:
                server = await self._get_server()
                await server.release_env_id.remote(self._sub_env_id)
            except Exception as exc:
                logger.warning(
                    "Failed to release env_id=%d: %s", self._sub_env_id, exc
                )
            self._sub_env_id = None

    # ------------------------------------------------------------------
    # Reward / termination helpers
    # ------------------------------------------------------------------

    def _get_correct_placement_reward(self) -> float:
        if self.config.correct_placement_reward is not None:
            return float(self.config.correct_placement_reward)
        return float(self.config.success_reward)

    def _select_ground_truth_path(self, seed: int) -> Optional[Path]:
        if self._dataset_entries:
            entry = self._dataset_entries[seed % len(self._dataset_entries)]
            matched = self._ground_truth_by_stem.get(entry["stem"])
            if matched is not None:
                return matched

        if self._ground_truth_entries:
            return self._ground_truth_entries[seed % len(self._ground_truth_entries)]
        return None

    def _load_current_task_spec(self) -> TaskSpec:
        if self._current_ground_truth_path is None:
            return TaskSpec.empty()
        try:
            return load_task_spec(self._current_ground_truth_path)
        except Exception as exc:
            logger.warning("Failed to load ground truth %s: %s", self._current_ground_truth_path, exc)
            return TaskSpec.empty()

    def _make_fallback_images(
        self,
        *,
        count: int,
        color: tuple[int, int, int] = (50, 50, 50),
    ) -> List[Image.Image]:
        return [Image.new("RGB", self.config.image_size, color) for _ in range(max(0, count))]

    async def _render_env_images(self) -> List[Image.Image]:
        if self._sub_env_id is None:
            return list(self._latest_scene_images)

        server = await self._get_server()
        try:
            isaac_images = await asyncio.wait_for(server.render.remote(self._sub_env_id), timeout=30.0)
            if not isaac_images:
                raise ValueError("render returned empty list")
            self._latest_scene_images = list(isaac_images)
            return list(isaac_images)
        except Exception as exc:
            logger.warning("Isaac render failed, using fallback images: %s", exc)
            if self._latest_scene_images:
                return list(self._latest_scene_images)
            n_images = len(getattr(self, "_dataset_images_cache", [])) or 5
            fallback = self._make_fallback_images(count=n_images, color=(50, 50, 50))
            self._latest_scene_images = list(fallback)
            return fallback

    def _build_placement_obs_text(
        self,
        coordinate: Dict[str, int],
        placement_result: PlacementRewardResult,
        termination_status: TerminationStatus,
    ) -> str:
        cam_labels = ["Top", "Front", "Side", "Iso", "Iso2"]
        label_lines = [
            f"{cam_labels[i] if i < len(cam_labels) else f'Cam{i}'}: {self.config.image_placeholder}"
            for i in range(len(self._latest_scene_images))
        ]
        status_lines = [
            f"[System]: Block placed at ({coordinate['x']}, {coordinate['y']}, {coordinate['z']}).",
            f"Rule check: {placement_result.feedback}",
            f"Placement attempts: {termination_status.placement_attempts}/{termination_status.max_attempts}.",
        ]
        if termination_status.done and termination_status.reason is not None:
            status_lines.append(f"Episode status: terminated by {termination_status.reason}.")
        elif termination_status.task_completed:
            status_lines.append("The current structure already matches the target. Submit when you are ready.")
        status_lines.extend(label_lines)
        status_lines.append("Place the next cube or submit when done.")
        return "\n".join(status_lines)

    def _build_submit_obs_text(self, termination_status: TerminationStatus) -> str:
        cam_labels = ["Top", "Front", "Side", "Iso", "Iso2"]
        label_lines = [
            f"{cam_labels[i] if i < len(cam_labels) else f'Cam{i}'}: {self.config.image_placeholder}"
            for i in range(len(self._latest_scene_images))
        ]
        verdict = (
            "Submission accepted: the current structure matches the target."
            if termination_status.success
            else "Submission finished the episode, but the current structure does not match the target."
        )
        lines = [
            f"[System]: {verdict}",
            f"Placement attempts: {termination_status.placement_attempts}/{termination_status.max_attempts}.",
            *label_lines,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _scan_dataset(self, root: str) -> List[Dict]:
        """Scan the dataset directory and return a sorted list of valid entries.

        Each entry is a dict with keys:
        - ``dir``:  ``pathlib.Path`` to the snapshot sub-directory
        - ``stem``: directory name string (e.g. ``"0001"``)
        - ``imgs``: list of 5 ``Path`` objects (top/front/side/iso/iso2)
        - ``json``: ``Path`` to the ``_data.json`` file
        """
        entries: List[Dict] = []
        root_path = Path(root)
        if not root_path.exists():
            logger.warning("Dataset root does not exist: %s", root)
            return entries
        img_suffixes = ["_top", "_front", "_side", "_iso", "_iso2"]
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue
            stem = subdir.name  # e.g. "0001"
            imgs = [subdir / f"{stem}{s}.png" for s in img_suffixes]
            json_path = subdir / f"{stem}_data.json"
            if all(p.exists() for p in imgs) and json_path.exists():
                entries.append({"dir": subdir, "stem": stem, "imgs": imgs, "json": json_path})
        return entries

    def _load_dataset_images(self, seed: int) -> List["Image.Image"]:
        """Return the 5 pre-rendered images for a dataset entry chosen by *seed*.

        Images are in order: top, front, side, iso, iso2 — matching the camera
        index ordering used by the rest of the environment.
        """
        if not self._dataset_entries:
            return []
        entry = self._dataset_entries[seed % len(self._dataset_entries)]
        images: List["Image.Image"] = []
        for img_path in entry["imgs"]:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", img_path, exc)
                images.append(Image.new("RGB", self.config.image_size, (0, 0, 0)))
        return images

    def _load_target_description(self, seed: int) -> str:
        """Return a concise task description based on the selected ground truth."""
        task_spec = self._current_task_spec
        if task_spec.total_blocks <= 0:
            return (
                "Your task is to replicate the block structure shown in the image. "
                "Observe the target configuration carefully and place blocks one by one "
                "to reproduce it."
            )

        length, width, height = task_spec.dimensions
        return (
            "Your task is to replicate the target structure shown in the images. "
            f"The target contains {task_spec.total_blocks} blocks in a {length}x{width}x{height} grid. "
            f"You may make at most {self.termination_manager.max_attempts} placement attempts. "
            "A supported block on a valid target candidate is rewarded; floating or non-candidate placements are penalized."
        )

    # ------------------------------------------------------------------
    # Prompt / observation helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt string."""
        # Use checked system prompt for Isaac environment: if the composed
        # prompt is malformed, return a concise corrective example so the
        # LLM replies in a parseable format instead of producing invalid text.
        try:
            from .utils.prompt import get_checked_system_prompt
        except Exception:
            # Fallback to original behavior on import error
            fmt = format_prompt(add_example=self.config.use_example_in_sys_prompt)
            return system_prompt() + "\n" + fmt

        return get_checked_system_prompt(add_example=self.config.use_example_in_sys_prompt)

    def _img_placeholders(self) -> str:
        """Return n image placeholder tokens separated by newlines."""
        # Only provide placeholders for the first few turns to avoid hitting 
        # token limits and causing desync between text and image list.
        # Max 5 turns of images = 15 images total.
        if self.steps_taken >= 6:
            return "(Latest images omitted to save context)"
            
        return "\n".join(
            self.config.image_placeholder for _ in range(self.config.n_views)
        )

    def _make_multi_image_obs(
        self, obs_str: str, images: List[Image.Image], action_str: str = ""
    ) -> Dict[str, Any]:
        """Wrap an observation string and images into the standard obs dict."""
        # Fundamental Fix: Hallucination Absorber
        # In multi-turn context, if the model hallucinates vision tags in its response,
        # the AgentLoop will store them in history, causing desync with the image list.
        # We detect tags in action_str and provide dummy images to keep the counts aligned.
        
        vision_start_tag = "<|vision_start|>"
        hallucinated_tags = action_str.count(vision_start_tag)
        
        target_count = self.config.n_views if self.steps_taken < 6 else 0
        total_needed = target_count + hallucinated_tags
        
        if total_needed == 0:
            return {"obs_str": obs_str}

        processed_images = []
        # 1. Add dummies for hallucinated tags first (to align with the previous turn's response)
        for _ in range(hallucinated_tags):
            processed_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            
        # 2. Add the actual new images
        if self.steps_taken < 6:
            for i in range(self.config.n_views):
                if i < len(images):
                    img = images[i]
                    if img.size != (224, 224):
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    processed_images.append(img)
                else:
                    processed_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        return {
            "obs_str": obs_str,
            "multi_modal_input": {
                self.config.image_placeholder: processed_images,
            },
        }