from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


STACK_CAMERA_NAMES = ("camera", "camera_front", "camera_side", "camera_iso", "camera_iso2")
STACK_CAMERA_RESOLUTION = (224, 224)
_LEGACY_CAMERA_NAMES = ("table_cam", "table_high_cam", "robot_cam", "cam_default")


def _stack_camera_layout(cube_size: float) -> dict[str, dict[str, tuple[float, ...]]]:
    table_height = 1.03
    camera_height = 0.7
    return {
        "camera": {"pos": (0.0, 0.0, table_height + camera_height + 0.5), "rot": (0.707, 0.0, 0.707, 0.0)},
        "camera_front": {"pos": (camera_height, 0.0, table_height + cube_size * 2), "rot": (0.0, 0.0, 0.0, 1.0)},
        "camera_side": {
            "pos": (0.0, camera_height, table_height + cube_size * 2),
            "rot": (0.707, 0.0, 0.0, -0.707),
        },
        "camera_iso": {
            "pos": (-camera_height / np.sqrt(2), camera_height / np.sqrt(2), table_height + camera_height),
            "rot": (0.85355, 0.14645, 0.35355, -0.35355),
        },
        "camera_iso2": {
            "pos": (camera_height / np.sqrt(2), -camera_height / np.sqrt(2), table_height + camera_height),
            "rot": (0.36, -0.33, 0.14, 0.85),
        },
    }


def configure_stack_scene_cameras(
    scene_cfg,
    enable_cameras: bool,
    cube_size: float,
    use_tiled: bool | None = None,
) -> None:
    """Configure assembling cameras on scene cfg.

    This function is idempotent and can be called multiple times.
    """
    for cam_name in _LEGACY_CAMERA_NAMES:
        if hasattr(scene_cfg, cam_name):
            setattr(scene_cfg, cam_name, None)

    for cam_name in STACK_CAMERA_NAMES:
        if hasattr(scene_cfg, cam_name):
            setattr(scene_cfg, cam_name, None)

    if not enable_cameras:
        return

    if use_tiled is None:
        use_tiled = os.getenv("VAGEN_USE_TILED", "1") != "0"
    cam_cfg_type = TiledCameraCfg if use_tiled else CameraCfg
    offset_cfg_type = getattr(cam_cfg_type, "OffsetCfg", TiledCameraCfg.OffsetCfg)
    height, width = STACK_CAMERA_RESOLUTION
    table_init_pos = (0.5, 0.0, -1.03)

    for cam_name, cam_cfg in _stack_camera_layout(cube_size=float(cube_size)).items():
        pos = cam_cfg["pos"]
        pos = (pos[0] + table_init_pos[0], pos[1] + table_init_pos[1], pos[2] + table_init_pos[2])
        rot = cam_cfg["rot"]
        setattr(
            scene_cfg,
            cam_name,
            cam_cfg_type(
                prim_path=f"{{ENV_REGEX_NS}}/{cam_name}",
                update_period=0.0,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 1000.0),
                ),
                offset=offset_cfg_type(pos=pos, rot=(rot[0], rot[1], rot[2], rot[3]), convention="world"),
            ),
        )


def _empty_image(env: ManagerBasedEnv, data_type: str, normalize: bool) -> torch.Tensor:
    device = torch.device(env.device)
    height, width = STACK_CAMERA_RESOLUTION
    if data_type == "rgb":
        dtype = torch.float32 if normalize else torch.uint8
        return torch.zeros((env.num_envs, height, width, 3), device=device, dtype=dtype)
    if data_type == "rgba":
        dtype = torch.float32 if normalize else torch.uint8
        return torch.zeros((env.num_envs, height, width, 4), device=device, dtype=dtype)
    return torch.zeros((env.num_envs, height, width, 1), device=device, dtype=torch.float32)


def camera_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """Safe camera image observation that tolerates missing camera sensors."""
    if not hasattr(env.scene, sensor_cfg.name):
        return _empty_image(env, data_type=data_type, normalize=normalize)
    sensor = env.scene[sensor_cfg.name]
    if sensor is None:
        return _empty_image(env, data_type=data_type, normalize=normalize)

    sensor_data = getattr(sensor, "data", None)
    if sensor_data is None:
        return _empty_image(env, data_type=data_type, normalize=normalize)
    output_dict = getattr(sensor_data, "output", None)
    if not isinstance(output_dict, dict) or data_type not in output_dict:
        return _empty_image(env, data_type=data_type, normalize=normalize)

    images = output_dict[data_type]

    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        intrinsic = getattr(sensor_data, "intrinsic_matrices", None)
        if intrinsic is not None:
            images = math_utils.orthogonalize_perspective_depth(images, intrinsic)

    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images = images.clone()
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()

