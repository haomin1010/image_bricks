# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.markers.config import FRAME_MARKER_CFG

from .cfg_override import (
    ASSEMBLING_MAX_CUBES,
    DEFAULT_GRID_CELL_SIZE,
    DEFAULT_GRID_LINE_THICKNESS,
    DEFAULT_GRID_ORIGIN,
    DEFAULT_GRID_SIZE,
    FRANKA_ARM_JOINT_NAMES,
    FRANKA_ARM_ONLY_CFG,
    AssemblingCfgOverride,
)
from . import mdp

def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _build_default_ee_frame_cfg() -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Base scene: robot + table + ground + light."""

    robot: ArticulationCfg = FRANKA_ARM_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    ee_frame: FrameTransformerCfg = _build_default_ee_frame_cfg()

    def __post_init__(self):
        super().__post_init__()
        if _env_flag("VAGEN_ENABLE_GRID_VISUALS", default=True):
            grid_origin, grid_size, cell_size, line_thickness = self.get_grid_spec()
            self.configure_grid_visuals(
                grid_origin=grid_origin,
                grid_size=grid_size,
                cell_size=cell_size,
                line_thickness=line_thickness,
            )

    def get_grid_spec(self) -> tuple[list[float], int, float, float]:
        return (
            list(DEFAULT_GRID_ORIGIN),
            int(DEFAULT_GRID_SIZE),
            float(DEFAULT_GRID_CELL_SIZE),
            float(DEFAULT_GRID_LINE_THICKNESS),
        )

    def configure_grid_visuals(
        self,
        *,
        grid_origin,
        grid_size: int,
        cell_size: float,
        line_thickness: float,
    ) -> None:
        """Attach grid visualization assets directly on the scene config."""
        half_width = grid_size * cell_size / 2
        origin = grid_origin

        for i in range(grid_size + 1):
            suffix = f"_{i}"
            y_pos = origin[1] - half_width + i * cell_size
            setattr(
                self,
                f"grid_h{suffix}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/grid_h{suffix}",
                    spawn=sim_utils.CuboidCfg(
                        size=(grid_size * cell_size + line_thickness, line_thickness, 0.0002),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                        semantic_tags=[("class", "grid")],
                        collision_props=None,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0], y_pos, origin[2])),
                ),
            )

            x_pos = origin[0] - half_width + i * cell_size
            setattr(
                self,
                f"grid_v{suffix}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/grid_v{suffix}",
                    spawn=sim_utils.CuboidCfg(
                        size=(line_thickness, grid_size * cell_size + line_thickness, 0.0002),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                        semantic_tags=[("class", "grid")],
                        collision_props=None,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(x_pos, origin[1], origin[2])),
                ),
            )

        setattr(
            self,
            "grid_x_axis",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/grid_x_axis",
                spawn=sim_utils.CuboidCfg(
                    size=(grid_size * cell_size + line_thickness, line_thickness * 3.0, 0.0004),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    semantic_tags=[("class", "grid_axis")],
                    collision_props=None,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0], origin[1] - half_width, origin[2])),
            ),
        )
        setattr(
            self,
            "grid_y_axis",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/grid_y_axis",
                spawn=sim_utils.CuboidCfg(
                    size=(line_thickness * 3.0, grid_size * cell_size + line_thickness, 0.0004),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    semantic_tags=[("class", "grid_axis")],
                    collision_props=None,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0] - half_width, origin[1], origin[2])),
            ),
        )
        setattr(
            self,
            "grid_origin_marker",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/grid_origin_marker",
                spawn=sim_utils.CuboidCfg(
                    size=(0.02, 0.02, 0.0006),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    semantic_tags=[("class", "grid_origin")],
                    collision_props=None,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(origin[0] - half_width, origin[1] - half_width, origin[2] + 0.0003)
                ),
            ),
        )

    def _resolve_local_isaac_assets_root(self) -> Path | None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "assets" / "Isaac"
            if candidate.exists():
                return candidate
        return None

    def resolve_asset_path(
        self,
        *,
        local_rel: str,
        nucleus_rel: str | None = None,
        isaac_assets_root: Path | str | None = None,
    ) -> str | None:
        if isaac_assets_root is None:
            isaac_assets_root = self._resolve_local_isaac_assets_root()
        elif isaac_assets_root is not None:
            isaac_assets_root = Path(isaac_assets_root)

        if isaac_assets_root is not None:
            local_path = isaac_assets_root / local_rel
            if local_path.exists():
                return str(local_path)

        if nucleus_rel is not None:
            return f"{ISAAC_NUCLEUS_DIR}/{nucleus_rel}"
        return None

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.1034],
        ),
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation terms shared by assembling tasks."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        root_pos = ObsTerm(func=mdp.root_pos_w)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_pos = ObsTerm(
            func=mdp.all_cube_positions_in_world_frame,
            params={"max_cubes": ASSEMBLING_MAX_CUBES},
        )
        env_origin = ObsTerm(func=mdp.env_origin)
        cube_quat = ObsTerm(
            func=mdp.all_cube_orientations_in_world_frame,
            params={"max_cubes": ASSEMBLING_MAX_CUBES},
        )
        ee_pos = ObsTerm(func=mdp.ee_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        ee_quat = ObsTerm(func=mdp.ee_quat, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        gripper_closed = ObsTerm(func=mdp.gripper_closed_flag)
        grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_1"),
                "diff_threshold": 0.06,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        camera = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "normalize": False},
        )
        camera_front = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "rgb", "normalize": False},
        )
        camera_side = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("camera_side"), "data_type": "rgb", "normalize": False},
        )
        camera_iso = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("camera_iso"), "data_type": "rgb", "normalize": False},
        )
        camera_iso2 = ObsTerm(
            func=mdp.camera_image,
            params={"sensor_cfg": SceneEntityCfg("camera_iso2"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    # Will be populated by downstream task configs when needed.
    subtask_terms = None


@configclass
class TerminationsCfg:
    """Termination placeholder; concrete tasks add terms later."""
    pass


@configclass
class RewardsCfg:
    """Reward placeholder terms for assembling tasks."""

    placeholder = RewTerm(func=mdp.placeholder_reward, weight=1.0)


@configclass
class EventsCfg:
    """Event terms shared by assembling tasks."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    # No suction event: grasping is done by the native Franka parallel gripper.


@configclass
class AssemblingEnvCfg(ManagerBasedRLEnvCfg):
    """Base assembling environment config."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    commands = None
    curriculum = None
    gripper_joint_names = ["panda_finger_.*"]
    gripper_open_val = 0.04
    gripper_threshold = 0.005

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        AssemblingCfgOverride.from_env().apply(self, arm_joint_names=FRANKA_ARM_JOINT_NAMES)
