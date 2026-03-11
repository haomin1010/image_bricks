"""
Isaac managed environment for BrickIsaac partial-view mode.

Partial-view policy:
- Reset shows full target multi-view images (5 cameras).
- The model may query exactly one camera per turn with {"query": [id]}.
- Reward/termination/task evaluation follows the same logic as full-view mode.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
from PIL import Image

from ..gym_image_env import GymImageEnv
from .reward_manager import IsaacRewardConfig, IsaacRewardManager, PlacementRewardResult
from .task_spec import BrickPosition, TaskSpec, load_snapshot_task_spec
from .utils.prompt import (
    action_template,
    format_prompt,
    get_checked_system_prompt,
    init_observation_template,
    query_result_template,
    system_prompt,
    target_description,
)
from .utils.utils import parse_response

logger = logging.getLogger(__name__)
MAX_PARTIAL_VIEW_CAMERAS = 5
DEFAULT_DATASET_ROOT = str(
    Path(__file__).resolve().parents[4]
    / "assets"
    / "dataset"
    / "output_snapshots"
    / "test"
)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class IsaacManagedEnvConfig:
    """Configuration for the Isaac partial-view managed environment."""

    # Executor / manager settings
    num_total_envs: int = 64

    # Cameras
    n_cameras: int = MAX_PARTIAL_VIEW_CAMERAS
    n_views: Optional[int] = None  # compatibility with legacy YAML keys
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
    dataset_root: str = DEFAULT_DATASET_ROOT
    collapse_mock_after_attempt: int = -1

    def __post_init__(self) -> None:
        self.num_total_envs = int(self.num_total_envs)
        if self.n_views not in (None, ""):
            self.n_cameras = int(self.n_views)
        self.n_cameras = max(1, min(MAX_PARTIAL_VIEW_CAMERAS, int(self.n_cameras)))
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
        if not self.dataset_root:
            self.dataset_root = DEFAULT_DATASET_ROOT
        self.collapse_mock_after_attempt = int(self.collapse_mock_after_attempt)


_CONFIG_FIELDS = {f.name for f in fields(IsaacManagedEnvConfig)}


class IsaacManagedEnv(GymImageEnv):
    """GymImageEnv proxy backed by the detached IsaacEnvServer Ray actor."""

    _server_handle = None
    _server_lock = asyncio.Lock()

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        known = {k: v for k, v in env_config.items() if k in _CONFIG_FIELDS}
        self.config = IsaacManagedEnvConfig(**known)

        self._sub_env_id: Optional[int] = None
        self.total_reward: float = 0.0
        self.steps_taken: int = 0
        self.trajectory: List[Dict[str, Any]] = []
        self._latest_scene_images: List[Image.Image] = []
        self._dataset_images_cache: List[Image.Image] = []
        self._current_task_spec: TaskSpec = TaskSpec.empty()
        self._current_dataset_entry: Optional[Dict[str, Any]] = None
        # Track queried camera IDs since the last placement step.
        self._queried_cameras_since_last_placement: set[int] = set()

        reward_config = IsaacRewardConfig(
            format_reward=self.config.format_reward,
            correct_placement_reward=self._get_correct_placement_reward(),
            floating_placement_penalty=self.config.floating_placement_penalty,
            non_candidate_penalty=self.config.non_candidate_penalty,
        )
        self.reward_manager = IsaacRewardManager(reward_config)

        self._dataset_entries: List[Dict[str, Any]] = self._scan_dataset(self.config.dataset_root)
        if not self._dataset_entries:
            logger.warning("Dataset is empty or not found at: %s", self.config.dataset_root)
        else:
            logger.debug("Loaded %d dataset entries from %s", len(self._dataset_entries), self.config.dataset_root)

    async def _get_server(self):
        """Lazily obtain the per-worker server handle singleton."""
        if IsaacManagedEnv._server_handle is not None:
            return IsaacManagedEnv._server_handle

        async with IsaacManagedEnv._server_lock:
            if IsaacManagedEnv._server_handle is not None:
                return IsaacManagedEnv._server_handle

            try:
                IsaacManagedEnv._server_handle = ray.get_actor("IsaacEnvServer")
                logger.debug("Connected to existing IsaacEnvServer actor.")
            except ValueError:
                logger.info("IsaacEnvServer actor not found. It should be started by the main script.")
                raise RuntimeError("IsaacEnvServer actor not found. Ensure it is started as a detached actor.")

            return IsaacManagedEnv._server_handle

    async def system_prompt(self) -> Dict[str, Any]:
        """Return the system-level prompt observation."""
        return {"obs_str": self._build_system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset one env slot and return partial-view initial observation."""
        server = await self._get_server()

        if self._sub_env_id is None:
            self._sub_env_id = await server.allocate_env_id.remote()

        self._current_dataset_entry = self._select_dataset_entry(seed)
        self._current_task_spec = self._load_current_task_spec()
        requested_num_tasks = int(max(1, self._current_task_spec.total_blocks))

        try:
            reset_payload = {"seed": int(seed), "num_tasks": requested_num_tasks}
            response = await asyncio.wait_for(
                server.remote_reset.remote(self._sub_env_id, reset_payload),
                timeout=120.0,
            )
            env_info = dict(response.get("info", {}) or {})
        except Exception as exc:
            logger.error("Failed to reset Isaac environment (timeout or crash): %s", exc)
            env_info = {}

        self.total_reward = 0.0
        self.steps_taken = 0
        self.trajectory = []
        self._queried_cameras_since_last_placement.clear()

        all_images = self._load_dataset_images()
        self._dataset_images_cache = list(all_images)
        self._latest_scene_images = list(all_images)

        self.reward_manager.reset(self._current_task_spec)

        desc = target_description(
            self._current_task_spec,
            max_attempts=max(
                1,
                int(math.ceil(max(1, int(self._current_task_spec.total_blocks)) * float(self.config.max_attempts_factor))),
            ),
        )
        env_info["target_description"] = desc

        logger.debug(
            "partial-reset: seed=%d dataset_index=%d gt=%s images=%d max_attempts=%d requested_num_tasks=%d",
            seed,
            seed % max(len(self._dataset_entries), 1),
            self._current_dataset_entry["json"].name if self._current_dataset_entry else "none",
            len(all_images),
            max(
                1,
                int(math.ceil(max(1, int(self._current_task_spec.total_blocks)) * float(self.config.max_attempts_factor))),
            ),
            requested_num_tasks,
        )

        target_view_count = MAX_PARTIAL_VIEW_CAMERAS
        target_images = list(all_images[:target_view_count])
        if len(target_images) < target_view_count:
            target_images.extend(
                self._make_fallback_images(
                    count=target_view_count - len(target_images),
                    color=(40, 40, 40),
                )
            )
        target_labels = [f"Target camera {idx}" for idx in range(target_view_count)]
        target_placeholders = "\n".join(self.config.image_placeholder for _ in range(target_view_count))
        obs_text = (desc + "\n" if desc else "") + init_observation_template(
            img_placeholders=target_placeholders,
            camera_labels=target_labels,
        )
        obs = self._make_multi_image_obs(obs_text, target_images)
        return obs, env_info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Parse action, step Isaac when needed, and evaluate reward/termination."""
        self.steps_taken += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        coordinate = parsed.get("coordinate")
        query_cameras = parsed.get("query_cameras")
        is_submit = bool(parsed.get("is_submit", False))
        action_valid = bool(parsed.get("format_correct", False))

        placement_result: Optional[PlacementRewardResult] = None
        isaac_info: Dict[str, Any] = {}

        metrics = {
            "turn_metrics": {
                "action_is_valid": action_valid,
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
            goal = {"x": coordinate["x"], "y": coordinate["y"], "z": coordinate["z"]}
            server = await self._get_server()
            try:
                step_response = await asyncio.wait_for(
                    server.remote_step.remote(self._sub_env_id, goal),
                    timeout=300.0,
                )
                isaac_info = dict(step_response.get("info", {}) or {})
                done = bool(step_response.get("done", False))
            except Exception as exc:
                logger.error("Failed to step Isaac (timeout or crash): %s", exc)
                isaac_info = {"timeout": True}
                done = True

            placement_result = self.reward_manager.evaluate_placement(BrickPosition.from_mapping(goal))
            reward += placement_result.reward_delta

            scene_images = await self._render_env_images()
            cam0_images = [scene_images[0]] if scene_images else self._make_fallback_images(count=1, color=(40, 40, 40))
            feedback = self._build_placement_feedback(coordinate, placement_result)
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=feedback,
                    img_placeholder=self.config.image_placeholder,
                ),
                cam0_images,
                action_str=action_str,
            )

            metrics["turn_metrics"]["action_is_effective"] = True
            metrics["traj_metrics"]["task_completed"] = bool(self.reward_manager.task_completed)
            metrics["traj_metrics"]["collapse_detected"] = bool(
                isaac_info.get("termination_reason") == "collapse"
            )
            metrics["traj_metrics"]["termination_reason"] = isaac_info.get("termination_reason")
            # New placement opens a new query window.
            self._queried_cameras_since_last_placement.clear()
            info.update(
                {
                    "timeout": bool(isaac_info.get("timeout", False)),
                    "placement_outcome": placement_result.outcome,
                    "placement_feedback": placement_result.feedback,
                }
            )

        elif is_submit:
            goal = {"type": "submit"}
            server = await self._get_server()
            try:
                step_response = await asyncio.wait_for(
                    server.remote_step.remote(self._sub_env_id, goal),
                    timeout=300.0,
                )
                isaac_info = dict(step_response.get("info", {}) or {})
                done = bool(step_response.get("done", False))
            except Exception as exc:
                logger.error("Failed to submit Isaac (timeout or crash): %s", exc)
                isaac_info = {"timeout": True}
                done = True

            scene_images = await self._render_env_images()
            cam0_images = [scene_images[0]] if scene_images else self._make_fallback_images(count=1, color=(40, 40, 40))
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=self._build_submit_feedback(isaac_info),
                    img_placeholder=self.config.image_placeholder,
                ),
                cam0_images,
                action_str=action_str,
            )

            metrics["turn_metrics"]["action_is_effective"] = True
            metrics["traj_metrics"]["success"] = bool(isaac_info.get("success", False))
            metrics["traj_metrics"]["task_completed"] = bool(self.reward_manager.task_completed)
            metrics["traj_metrics"]["collapse_detected"] = bool(
                isaac_info.get("termination_reason") == "collapse"
            )
            metrics["traj_metrics"]["termination_reason"] = isaac_info.get("termination_reason")
            info.update({"timeout": bool(isaac_info.get("timeout", False))})

        elif query_cameras is not None:
            scene_images = await self._render_env_images()
            max_cam = min(len(scene_images), int(self.config.n_cameras), MAX_PARTIAL_VIEW_CAMERAS)
            selected_ids = [cam_id for cam_id in query_cameras if 0 <= cam_id < max_cam]

            if not selected_ids:
                action_valid = False
                metrics["turn_metrics"]["action_is_valid"] = False
                cam0_images = [scene_images[0]] if scene_images else self._make_fallback_images(count=1, color=(30, 30, 30))
                msg = (
                    f"Invalid camera query: {query_cameras}. "
                    f"Use exactly one camera ID: {{\"query\": [INT]}} with INT in 0..{max(0, max_cam - 1)}."
                )
                obs = self._make_multi_image_obs(
                    action_template(
                        action_result=msg,
                        img_placeholder=self.config.image_placeholder,
                    ),
                    cam0_images,
                    action_str=action_str,
                )
            else:
                selected_id = selected_ids[0]
                if selected_id in self._queried_cameras_since_last_placement:
                    action_valid = False
                    metrics["turn_metrics"]["action_is_valid"] = False
                    cam0_images = [scene_images[0]] if scene_images else self._make_fallback_images(count=1, color=(30, 30, 30))
                    msg = (
                        f"Camera {selected_id} has already been queried since the last placement. "
                        "Please query a different camera first, or place a block."
                    )
                    obs = self._make_multi_image_obs(
                        action_template(
                            action_result=msg,
                            img_placeholder=self.config.image_placeholder,
                        ),
                        cam0_images,
                        action_str=action_str,
                    )
                else:
                    selected_images = [scene_images[selected_id]]
                    obs = self._make_multi_image_obs(
                        query_result_template(
                            camera_id=selected_id,
                            img_placeholder=self.config.image_placeholder,
                        ),
                        selected_images,
                        action_str=action_str,
                    )
                    self._queried_cameras_since_last_placement.add(int(selected_id))
                    metrics["turn_metrics"]["action_is_effective"] = True

        else:
            action_valid = False
            metrics["turn_metrics"]["action_is_valid"] = False
            scene_images = await self._render_env_images()
            cam0_images = [scene_images[0]] if scene_images else self._make_fallback_images(count=1, color=(30, 30, 30))
            msg = (
                "Could not parse your action. Valid formats:\n"
                f'  Query one camera: {{"query": [2]}} (ID 0..{max(0, self.config.n_cameras - 1)})\n'
                '  Place a brick: {"x": 2, "y": 3, "z": 0}\n'
                "  Submit: submit"
            )
            obs = self._make_multi_image_obs(
                action_template(
                    action_result=msg,
                    img_placeholder=self.config.image_placeholder,
                ),
                cam0_images,
                action_str=action_str,
            )

        reward += self.reward_manager.format_reward(action_valid)

        if self.steps_taken >= self.config.max_steps and not done:
            done = True
            metrics["traj_metrics"]["termination_reason"] = "max_steps_guard"

        success = bool(isaac_info.get("success", False))
        term_reason = metrics["traj_metrics"]["termination_reason"] or isaac_info.get("termination_reason")
        collapsed = bool(term_reason == "collapse")

        metrics["traj_metrics"]["success"] = success
        metrics["traj_metrics"]["task_completed"] = bool(self.reward_manager.task_completed)
        metrics["traj_metrics"]["collapse_detected"] = collapsed
        metrics["traj_metrics"]["termination_reason"] = term_reason

        info["metrics"] = metrics
        info["success"] = success
        info["termination"] = {
            "reason": term_reason,
            "collapsed": collapsed,
            "submitted": bool(is_submit),
            "task_completed": bool(self.reward_manager.task_completed),
        }
        info["reward_breakdown"] = {
            "format_reward": self.reward_manager.format_reward(action_valid),
            "placement_reward": 0.0 if placement_result is None else placement_result.reward_delta,
        }
        self.total_reward += reward
        info["total_reward"] = self.total_reward
        info["remaining_target_blocks"] = len(self.reward_manager.remaining_target_positions())

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
                "termination_reason": term_reason,
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
                logger.warning("Failed to release env_id=%d: %s", self._sub_env_id, exc)
            self._sub_env_id = None

    def _get_correct_placement_reward(self) -> float:
        if self.config.correct_placement_reward is not None:
            return float(self.config.correct_placement_reward)
        return float(self.config.success_reward)

    def _select_dataset_entry(self, seed: int) -> Optional[Dict[str, Any]]:
        if not self._dataset_entries:
            return None
        return self._dataset_entries[seed % len(self._dataset_entries)]

    def _load_current_task_spec(self) -> TaskSpec:
        if self._current_dataset_entry is None:
            return TaskSpec.empty()
        try:
            return load_snapshot_task_spec(self._current_dataset_entry["json"])
        except Exception as exc:
            logger.warning("Failed to load snapshot data %s: %s", self._current_dataset_entry.get("json"), exc)
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
            n_images = len(self._dataset_images_cache) or max(1, int(self.config.n_cameras))
            fallback = self._make_fallback_images(count=n_images, color=(50, 50, 50))
            self._latest_scene_images = list(fallback)
            return fallback

    def _build_placement_feedback(
        self,
        coordinate: Dict[str, int],
        placement_result: PlacementRewardResult,
    ) -> str:
        lines = [
            f"Block placed at ({coordinate['x']}, {coordinate['y']}, {coordinate['z']}).",
            f"Rule check: {placement_result.feedback}",
        ]
        if self.reward_manager.task_completed:
            lines.append("Current structure matches the target. You can submit now.")
        return " ".join(lines)

    def _build_submit_feedback(self, isaac_info: Dict[str, Any]) -> str:
        verdict = (
            "Submission accepted: the current structure matches the target."
            if bool(isaac_info.get("success", False))
            else "Submission finished the episode, but the current structure does not match the target."
        )
        return verdict

    def _scan_dataset(self, root: str) -> List[Dict[str, Any]]:
        """Scan strict subdir dataset format and return valid entries."""
        entries: List[Dict[str, Any]] = []
        root_path = Path(root)
        if not root_path.exists():
            logger.warning("Dataset root does not exist: %s", root)
            return entries

        img_suffixes = ["_top", "_front", "_side", "_iso", "_iso2"]
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue
            stem = subdir.name
            imgs = [subdir / f"{stem}{s}.png" for s in img_suffixes]
            json_path = subdir / f"{stem}_data.json"
            if all(p.exists() for p in imgs) and json_path.exists():
                entries.append({"dir": subdir, "stem": stem, "imgs": imgs, "json": json_path})
        return entries

    def _load_dataset_images(self) -> List[Image.Image]:
        """Return the 5 pre-rendered images for the current dataset entry."""
        if self._current_dataset_entry is None:
            return []
        entry = self._current_dataset_entry
        images: List[Image.Image] = []
        for img_path in entry["imgs"]:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", img_path, exc)
                images.append(Image.new("RGB", self.config.image_size, (0, 0, 0)))
        return images

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt string."""
        try:
            return get_checked_system_prompt(
                n_cameras=self.config.n_cameras,
                add_example=self.config.use_example_in_sys_prompt,
            )
        except Exception:
            fmt = format_prompt(
                n_cameras=self.config.n_cameras,
                add_example=self.config.use_example_in_sys_prompt,
            )
            return system_prompt(n_cameras=self.config.n_cameras) + "\n" + fmt

    def _make_multi_image_obs(
        self,
        obs_str: str,
        images: List[Image.Image],
        action_str: str = "",
    ) -> Dict[str, Any]:
        """Wrap text plus variable-length images into the standard observation dict."""
        vision_start_tag = "<|vision_start|>"
        hallucinated_tags = action_str.count(vision_start_tag)

        if len(images) == 0 and hallucinated_tags == 0:
            return {"obs_str": obs_str}

        processed_images: List[Image.Image] = []
        for _ in range(hallucinated_tags):
            processed_images.append(Image.new("RGB", self.config.image_size, (0, 0, 0)))

        for img in images:
            if img.size != self.config.image_size:
                img = img.resize(self.config.image_size, Image.Resampling.LANCZOS)
            processed_images.append(img)

        return {
            "obs_str": obs_str,
            "multi_modal_input": {
                self.config.image_placeholder: processed_images,
            },
        }
