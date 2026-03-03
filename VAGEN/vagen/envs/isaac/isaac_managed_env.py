"""
Isaac Managed Environment — GymImageEnv proxy backed by the EnvManager system.

This environment supports two action types from the VLM:

1. **Query** – ``{"query": [cam_id, ...]}``
   Returns images from the requested cameras.  No physics step.
2. **Place** – ``{"x": INT, "y": INT, "z": INT}``
   Places a brick and returns camera-0's image.
3. **Submit** – ``submit``
   Ends the episode.

Usage
-----
Register this class in the env_registry config::

    env_registry:
      brick_isaac: vagen.envs.brick_isaac.IsaacManagedEnv

The ``env_config`` dict in the dataset YAML should contain at least:

- ``num_total_envs`` (int): Total sub-envs in the Isaac DirectVectorEnv.
- ``n_cameras`` (int): Number of cameras available (IDs 0 .. n_cameras-1).
"""

from __future__ import annotations

import json
import logging
import random
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
    query_result_template,
    system_prompt,
)
from .utils.utils import parse_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IsaacManagedEnvConfig:
    """Configuration for the Isaac-managed environment."""

    # Executor / manager settings
    num_total_envs: int = 64  # total sub-envs in the DirectVectorEnv

    # Cameras
    n_cameras: int = 3  # total cameras (IDs 0 .. n_cameras-1)
    image_size: Tuple[int, int] = (224, 224)

    # Step limits
    max_steps: int = 200

    # Prompt
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True

    # Reward shaping
    format_reward: float = 0.1
    success_reward: float = 1.0

    # Dataset
    dataset_root: str = "/mnt/data/image_bricks/assets/snapshots"


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

        # Scan dataset directory and cache valid entries on startup
        self._dataset_entries: List[Dict] = self._scan_dataset(self.config.dataset_root)
        if not self._dataset_entries:
            logger.warning("Dataset is empty or not found at: %s", self.config.dataset_root)
        else:
            logger.info("Loaded %d dataset entries from %s", len(self._dataset_entries), self.config.dataset_root)

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
        Allocate a sub-env ID (on first call), then load the initial observation
        from the pre-rendered dataset instead of querying the Isaac simulation.

        Images are selected deterministically from the dataset using ``seed``.
        The JSON metadata associated with the chosen snapshot is used to build
        the target-block description provided to the VLM.
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

        # Build target description from dataset JSON
        target_desc = self._load_target_description(seed)
        env_info["target_description"] = target_desc

        logger.info("reset: loaded dataset entry %d (seed=%d), %d images",
                    seed % max(len(self._dataset_entries), 1), seed, len(all_images))

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
        Parse the LLM action and execute it.

        Supported actions:
        - **Query**:  ``{"query": [cam_id, ...]}`` — returns requested camera views.
        - **Place**:  ``{"x": INT, "y": INT, "z": INT}`` — places a brick, returns camera 0.
        - **Submit**: ``submit`` — ends the episode.

        Args:
            action_str: Raw LLM output string.

        Returns:
            ``(obs, reward, done, info)``
        """
        print(f"Received action string: {action_str}")  # Debug logging
        self.steps_taken += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        coordinate = parsed.get("coordinate")
        is_submit = parsed.get("is_submit", False)
        format_correct = parsed.get("format_correct", False)

        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if coordinate is not None:
            # ----------------------------------------------------------
            # Place action — execute physics then render all 5 cameras
            # ----------------------------------------------------------
            goal = {"x": coordinate["x"], "y": coordinate["y"], "z": coordinate["z"]}
            server = await self._get_server()
            try:
                step_response = await asyncio.wait_for(
                    server.remote_step.remote(self._sub_env_id, goal), timeout=300.0
                )
                step_reward = step_response.get("reward", 0.0)
                step_done = step_response.get("done", False)
                step_success = step_response.get("info", {}).get("success", False)
            except Exception as e:
                logger.error("Failed to step Isaac (timeout or crash): %s", e)
                step_reward = 0.0
                step_done = True
                step_success = False

            # Render all cameras for VLM observation
            try:
                isaac_images = await asyncio.wait_for(
                    server.render.remote(self._sub_env_id), timeout=30.0
                )
                if not isaac_images:
                    raise ValueError("render returned empty list")
            except Exception as e:
                logger.warning("Isaac render failed, using gray fallback: %s", e)
                n = len(getattr(self, "_dataset_images_cache", [None] * 5))
                isaac_images = [Image.new("RGB", self.config.image_size, (50, 50, 50)) for _ in range(n)]

            cam_labels = ["Top", "Front", "Side", "Iso", "Iso2"]
            label_lines = [
                f"{cam_labels[i] if i < len(cam_labels) else f'Cam{i}'}: {self.config.image_placeholder}"
                for i in range(len(isaac_images))
            ]
            img_section = "\n".join(label_lines)
            obs_str = f"Block placed at ({coordinate['x']}, {coordinate['y']}, {coordinate['z']})."
            obs_text = f"[System]: {obs_str}\n{img_section}\nPlace the next cube or submit when done."

            reward += self.config.format_reward + step_reward
            if step_success:
                reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True
            done = step_done
            metrics["turn_metrics"]["action_is_effective"] = True

            obs = self._make_multi_image_obs(obs_text, isaac_images, action_str=action_str)
            info.update({"success": step_success, "timeout": False})

        elif is_submit:
            # ----------------------------------------------------------
            # Submit action — execute physics final evaluation + render all cameras
            # ----------------------------------------------------------
            goal = {"type": "submit"}
            server = await self._get_server()
            try:
                step_response = await asyncio.wait_for(
                    server.remote_step.remote(self._sub_env_id, goal), timeout=300.0
                )
                step_reward = step_response.get("reward", 0.0)
                step_success = step_response.get("info", {}).get("success", False)
            except Exception as e:
                logger.error("Failed to submit Isaac (timeout or crash): %s", e)
                step_reward = 0.0
                step_success = False

            # Render final state for VLM
            try:
                isaac_images = await asyncio.wait_for(
                    server.render.remote(self._sub_env_id), timeout=30.0
                )
                if not isaac_images:
                    raise ValueError("render returned empty list")
            except Exception as e:
                logger.warning("Isaac render failed for submit, using gray fallback: %s", e)
                n = len(getattr(self, "_dataset_images_cache", [None] * 5))
                isaac_images = [Image.new("RGB", self.config.image_size, (50, 50, 50)) for _ in range(n)]

            cam_labels = ["Top", "Front", "Side", "Iso", "Iso2"]
            label_lines = [
                f"{cam_labels[i] if i < len(cam_labels) else f'Cam{i}'}: {self.config.image_placeholder}"
                for i in range(len(isaac_images))
            ]
            img_section = "\n".join(label_lines)
            obs_text = f"[System]: Episode submitted.\n{img_section}"

            reward += step_reward
            if step_success:
                reward += self.config.success_reward
            done = True
            metrics["turn_metrics"]["action_is_effective"] = True
            metrics["traj_metrics"]["success"] = step_success

            obs = self._make_multi_image_obs(obs_text, isaac_images, action_str=action_str)
            info.update({"success": step_success, "timeout": False})

        else:
            # ----------------------------------------------------------
            # Parse failure — use blank image, no server call
            # ----------------------------------------------------------
            cam0_images = [Image.new("RGB", self.config.image_size, (30, 30, 30))]
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

        # Format reward for any correctly formatted action
        if format_correct:
            reward += self.config.format_reward

        # Step limit
        if self.steps_taken >= self.config.max_steps:
            done = True

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += reward

        # Record step trajectory
        self.trajectory.append({
            "step_idx": self.steps_taken,
            "coordinate": coordinate,
            "query_cameras": None,  # query removed
            "is_submit": is_submit,
            "reward": reward,
            "success": info["success"],
            "raw_action": parsed.get("action_content", ""),
            "thought": parsed.get("think_content", ""),
        })

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
        """Return a neutral task instruction.

        The VLM must infer the target configuration from the provided images
        alone.  Actual block coordinates are intentionally withheld to preserve
        the spatial-reasoning challenge.
        """
        return (
            "Your task is to replicate the block structure shown in the image. "
            "Observe the target configuration carefully and place blocks one by one "
            "to reproduce it."
        )

    # ------------------------------------------------------------------
    # Prompt / observation helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt string."""
        try:
            from .utils.prompt import get_checked_system_prompt
        except Exception:
            fmt = format_prompt(
                n_cameras=self.config.n_cameras,
                add_example=self.config.use_example_in_sys_prompt,
            )
            return system_prompt(n_cameras=self.config.n_cameras) + "\n" + fmt

        return get_checked_system_prompt(
            n_cameras=self.config.n_cameras,
            add_example=self.config.use_example_in_sys_prompt,
        )

    def _make_multi_image_obs(
        self,
        obs_str: str,
        images: List[Image.Image],
        action_str: str = "",
    ) -> Dict[str, Any]:
        """Wrap an observation string and a variable-length image list into
        the standard obs dict.

        Args:
            obs_str: Observation text (with ``<image>`` placeholders matching
                the number of *images*).
            images: Actual images to include in the observation.
            action_str: The previous VLM response (used to detect hallucinated
                vision tags and inject dummy images for alignment).
        """
        # Hallucination absorber: if the model hallucinated vision tags in its
        # response, inject dummy images so the tag count stays in sync.
        vision_start_tag = "<|vision_start|>"
        hallucinated_tags = action_str.count(vision_start_tag)

        if len(images) == 0 and hallucinated_tags == 0:
            return {"obs_str": obs_str}

        processed_images: List[Image.Image] = []

        # 1. Dummy images for hallucinated tags (align with previous response)
        for _ in range(hallucinated_tags):
            processed_images.append(
                Image.new("RGB", self.config.image_size, (0, 0, 0))
            )

        # 2. Actual images — resize if necessary
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
