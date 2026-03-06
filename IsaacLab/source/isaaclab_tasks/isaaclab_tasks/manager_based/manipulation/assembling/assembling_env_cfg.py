# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp


FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
ASSEMBLING_EE_BODY_NAME = "panda_hand"
ASSEMBLING_MAX_CUBES = int(getattr(mdp, "DEFAULT_MAX_CUBES", 8))
DEFAULT_GRID_ORIGIN: tuple[float, float, float] = (0.5, 0.0, 0.001)
DEFAULT_GRID_SIZE = 8
DEFAULT_GRID_LINE_THICKNESS = 0.001
DEFAULT_GRID_CELL_SIZE = 0.055 + DEFAULT_GRID_LINE_THICKNESS
DEFAULT_CUBE_SIZE = 0.0203 * 2.0

def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _resolve_local_franka_usd() -> str | None:
    rel_path = Path("Robots/FrankaRobotics/FrankaPanda/franka.usd")
    for parent in Path(__file__).resolve().parents:
        assets_root = parent / "assets" / "Isaac"
        candidate = assets_root / rel_path
        if candidate.exists():
            return str(candidate)
    return None


def _build_franka_osc_cfg() -> ArticulationCfg:
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.5,
        "panda_joint3": 0.0,
        "panda_joint4": -2.0,
        "panda_joint5": 0.0,
        "panda_joint6": 1.5,
        "panda_joint7": 0.7,
        "panda_finger_joint.*": 0.04,
    }
    robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
    robot_cfg.actuators["panda_shoulder"].damping = 0.0
    robot_cfg.actuators["panda_forearm"].stiffness = 0.0
    robot_cfg.actuators["panda_forearm"].damping = 0.0
    robot_cfg.spawn.rigid_props.disable_gravity = True

    local_franka_usd = _resolve_local_franka_usd()
    if local_franka_usd is not None:
        robot_cfg.spawn.usd_path = local_franka_usd
    return robot_cfg


FRANKA_ARM_ONLY_CFG = _build_franka_osc_cfg()


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
    arm_action: OperationalSpaceControllerActionCfg = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=100.0,
            motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        nullspace_joint_pos_target="center",
        position_scale=1.0,
        orientation_scale=1.0,
        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
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
            "default_pose": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.0400, 0.0400],
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
        # Hardcoded runtime settings (do not route through cfg_override).
        self.cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.cube_mass_props = MassPropertiesCfg(mass=0.12)
        self.cube_scale = (1.0, 1.0, 1.0)
        mdp.configure_stack_scene_cameras(scene_cfg=self.scene, enable_cameras=True, cube_size=DEFAULT_CUBE_SIZE)

        if isinstance(self.actions.arm_action, OperationalSpaceControllerActionCfg):
            self.actions.arm_action.joint_names = list(FRANKA_ARM_JOINT_NAMES)
            self.actions.arm_action.body_name = ASSEMBLING_EE_BODY_NAME
            if self.actions.arm_action.controller_cfg is not None:
                self.actions.arm_action.controller_cfg.nullspace_control = "position"
                self.actions.arm_action.controller_cfg.gravity_compensation = False
                self.actions.arm_action.controller_cfg.motion_control_axes_task = (1, 1, 1, 1, 1, 0)
                self.actions.arm_action.controller_cfg.nullspace_stiffness = 10.0
                self.actions.arm_action.controller_cfg.nullspace_damping_ratio = 1.0
            self.actions.arm_action.nullspace_joint_pos_target = "center"

        policy_obs = getattr(self.observations, "policy", None)
        if policy_obs is not None:
            cube_pos = getattr(policy_obs, "cube_pos", None)
            if cube_pos is not None and hasattr(cube_pos, "params"):
                cube_pos.params["max_cubes"] = ASSEMBLING_MAX_CUBES
            cube_quat = getattr(policy_obs, "cube_quat", None)
            if cube_quat is not None and hasattr(cube_quat, "params"):
                cube_quat.params["max_cubes"] = ASSEMBLING_MAX_CUBES
            ee_pos = getattr(policy_obs, "ee_pos", None)
            if ee_pos is not None and hasattr(ee_pos, "params"):
                ee_pos.params.pop("ee_body_name", None)
            ee_quat = getattr(policy_obs, "ee_quat", None)
            if ee_quat is not None and hasattr(ee_quat, "params"):
                ee_quat.params.pop("ee_body_name", None)
            grasped = getattr(policy_obs, "grasped", None)
            if grasped is not None and hasattr(grasped, "params"):
                grasped.params.pop("ee_body_name", None)

        self.decimation = 5
        self.episode_length_s = 600.0
        self.sim.dt = 0.01
        self.sim.render_interval = 5
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        if hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
            self.sim.physx.enable_external_forces_every_iteration = True
