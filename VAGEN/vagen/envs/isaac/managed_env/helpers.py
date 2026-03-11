from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import ray
from PIL import Image

from ..reward_manager import PlacementRewardResult
from ..task_spec import TaskSpec, load_task_spec, scan_ground_truth_entries
from isaaclab_tasks.manager_based.manipulation.assembling.termination_manager import TerminationStatus
from ..utils.prompt import action_template, format_prompt, init_observation_template, system_prompt

logger = logging.getLogger(__name__)


class IsaacManagedEnvHelperMixin:
    async def _get_server(self):
        """Lazily obtain the per-worker server handle singleton."""
        env_cls = type(self)
        if env_cls._server_handle is not None:
            return env_cls._server_handle

        async with env_cls._server_lock:
            if env_cls._server_handle is not None:
                return env_cls._server_handle

            try:
                env_cls._server_handle = ray.get_actor("IsaacEnvServer")
                logger.info("Connected to existing IsaacEnvServer actor.")
            except ValueError as exc:
                logger.info("IsaacEnvServer actor not found. It should be started by the main script.")
                raise RuntimeError(
                    "IsaacEnvServer actor not found. Ensure it is started as a detached actor."
                ) from exc

            return env_cls._server_handle

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

    def _select_ground_truth_path(self, seed: int) -> Path | None:
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
    ) -> list[Image.Image]:
        return [Image.new("RGB", self.config.image_size, color) for _ in range(max(0, count))]

    async def _render_env_images(self) -> list[Image.Image]:
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
            n_images = len(self._dataset_images_cache) or 5
            fallback = self._make_fallback_images(count=n_images, color=(50, 50, 50))
            self._latest_scene_images = list(fallback)
            return fallback

    def _build_placement_obs_text(
        self,
        coordinate: dict[str, int],
        placement_result: PlacementRewardResult,
        termination_status: TerminationStatus,
    ) -> str:
        cam_labels = ["Top", "Front", "Side", "Iso", "Iso2"]
        label_lines = [
            f"{cam_labels[idx] if idx < len(cam_labels) else f'Cam{idx}'}: {self.config.image_placeholder}"
            for idx in range(len(self._latest_scene_images))
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
            f"{cam_labels[idx] if idx < len(cam_labels) else f'Cam{idx}'}: {self.config.image_placeholder}"
            for idx in range(len(self._latest_scene_images))
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

    def _scan_dataset(self, root: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        root_path = Path(root)
        if not root_path.exists():
            logger.warning("Dataset root does not exist: %s", root)
            return entries
        img_suffixes = ["_top", "_front", "_side", "_iso", "_iso2"]
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue
            stem = subdir.name
            imgs = [subdir / f"{stem}{suffix}.png" for suffix in img_suffixes]
            json_path = subdir / f"{stem}_data.json"
            if all(path.exists() for path in imgs) and json_path.exists():
                entries.append({"dir": subdir, "stem": stem, "imgs": imgs, "json": json_path})
        return entries

    def _load_dataset_images(self, seed: int) -> list[Image.Image]:
        if not self._dataset_entries:
            return []
        entry = self._dataset_entries[seed % len(self._dataset_entries)]
        images: list[Image.Image] = []
        for img_path in entry["imgs"]:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", img_path, exc)
                images.append(Image.new("RGB", self.config.image_size, (0, 0, 0)))
        return images

    def _load_target_description(self, seed: int) -> str:
        _ = seed
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

    def _build_system_prompt(self) -> str:
        try:
            from ..utils.prompt import get_checked_system_prompt
        except Exception:
            formatted = format_prompt(
                n_cameras=self.config.n_cameras,
                add_example=self.config.use_example_in_sys_prompt,
            )
            return system_prompt(n_cameras=self.config.n_cameras) + "\n" + formatted

        return get_checked_system_prompt(
            n_cameras=self.config.n_cameras,
            add_example=self.config.use_example_in_sys_prompt,
        )

    def _make_multi_image_obs(
        self,
        obs_str: str,
        images: list[Image.Image],
        action_str: str = "",
    ) -> dict[str, Any]:
        vision_start_tag = "<|vision_start|>"
        hallucinated_tags = action_str.count(vision_start_tag)

        if len(images) == 0 and hallucinated_tags == 0:
            return {"obs_str": obs_str}

        processed_images: list[Image.Image] = []
        for _ in range(hallucinated_tags):
            processed_images.append(Image.new("RGB", self.config.image_size, (0, 0, 0)))

        for image in images:
            if image.size != self.config.image_size:
                image = image.resize(self.config.image_size, Image.Resampling.LANCZOS)
            processed_images.append(image)

        return {
            "obs_str": obs_str,
            "multi_modal_input": {
                self.config.image_placeholder: processed_images,
            },
        }

    def _build_reset_observation(self, all_images: list[Image.Image], target_desc: str) -> dict[str, Any]:
        cam_labels = ["Top view", "Front view", "Side view", "Iso view", "Iso2 view"]
        img_placeholders = "\n".join(self.config.image_placeholder for _ in all_images)
        obs_text = (target_desc + "\n" if target_desc else "") + init_observation_template(
            img_placeholders=img_placeholders,
            camera_labels=cam_labels[: len(all_images)],
        )
        return self._make_multi_image_obs(obs_text, all_images)

    def _build_invalid_action_observation(self, action_str: str) -> dict[str, Any]:
        cam0_images = self._make_fallback_images(count=1, color=(30, 30, 30))
        msg = (
            'Could not parse your action. Valid formats:\n'
            '  Place a brick: {"x": 2, "y": 3, "z": 0}\n'
            "  Submit: submit"
        )
        return self._make_multi_image_obs(
            action_template(
                action_result=msg,
                img_placeholder=self.config.image_placeholder,
            ),
            cam0_images,
            action_str=action_str,
        )

    def _load_ground_truth_entries(self) -> None:
        self._dataset_entries = self._scan_dataset(self.config.dataset_root)
        self._ground_truth_entries = scan_ground_truth_entries(self.config.ground_truth_root)
        self._ground_truth_by_stem = {path.stem: path for path in self._ground_truth_entries}
        if not self._dataset_entries:
            logger.warning("Dataset is empty or not found at: %s", self.config.dataset_root)
        else:
            logger.info("Loaded %d dataset entries from %s", len(self._dataset_entries), self.config.dataset_root)
        if not self._ground_truth_entries:
            logger.warning("Ground truth JSONs are empty or not found at: %s", self.config.ground_truth_root)
        else:
            logger.info(
                "Loaded %d ground truth JSON files from %s",
                len(self._ground_truth_entries),
                self.config.ground_truth_root,
            )
