# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


ASSEMBLING_MAX_CUBES = int(getattr(mdp, "DEFAULT_MAX_CUBES", 8))
DEFAULT_GRID_ORIGIN: tuple[float, float, float] = (0.5, 0.0, 0.001)
DEFAULT_GRID_SIZE = 8
DEFAULT_GRID_LINE_THICKNESS = 0.001
DEFAULT_GRID_CELL_SIZE = 0.051
DEFAULT_CUBE_SIZE = 0.05


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Base scene: table + ground + light."""

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

    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(1.0, 1.0, 1.0)
        )
    )

    side_light = AssetBaseCfg(
        prim_path="/World/side_light",
        spawn=sim_utils.DistantLightCfg(
            intensity=500.0,
            angle=0.0,
            color=(0.9, 0.9, 0.9)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.5, 0.5, 0.0, 0.0)
        )
    )

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
                        size=(grid_size * cell_size + line_thickness, line_thickness, 0.001),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
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
                        size=(line_thickness, grid_size * cell_size + line_thickness, 0.001),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
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
                    size=(grid_size * cell_size + line_thickness, line_thickness * 4.0, 0.003),
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
                    size=(line_thickness * 4.0, grid_size * cell_size + line_thickness, 0.003),
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
                    size=(0.018, 0.018, 0.006),
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
    pass


@configclass
class ObservationsCfg:
    """Observation terms shared by assembling tasks."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        cube_pos = ObsTerm(
            func=mdp.all_cube_positions_in_world_frame,
            params={"max_cubes": ASSEMBLING_MAX_CUBES},
        )
        env_origin = ObsTerm(func=mdp.env_origin)
        cube_quat = ObsTerm(
            func=mdp.all_cube_orientations_in_world_frame,
            params={"max_cubes": ASSEMBLING_MAX_CUBES},
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
    """Termination terms consumed from runtime-side packaged signals."""

    server_done = DoneTerm(func=mdp.server_done)
    done_submit = DoneTerm(func=mdp.done_submit)
    done_max_attempts = DoneTerm(func=mdp.done_max_attempts)
    done_teleport_failed = DoneTerm(func=mdp.done_teleport_failed)
    done_isaac_done = DoneTerm(func=mdp.done_isaac_done)
    done_repeat_coordinate = DoneTerm(func=mdp.done_repeat_coordinate)


@configclass
class RewardsCfg:
    """Reward placeholder terms for assembling tasks."""

    placeholder = RewTerm(func=mdp.placeholder_reward, weight=1.0)


@configclass
class EventsCfg:
    """Event terms shared by assembling tasks."""
    pass


@configclass
class AssemblingEnvCfg(ManagerBasedRLEnvCfg):
    """Base assembling environment config."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    commands = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    # Optional override in derived env configs.
    runtime_builder = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 600.0
        self.sim.dt = 0.01
        # Render every sim-step for crisper camera outputs in evaluation.
        self.sim.render_interval = int(os.getenv("VAGEN_RENDER_INTERVAL", "1"))
        if self.sim.render_interval < 1:
            self.sim.render_interval = 1

        # Prefer render modes without temporal accumulation to avoid
        # semi-transparent trails / ghosting on fast-moving teleported cubes.
        # Typical options: DLAA / TAA / OFF (depends on Isaac build).
        render_cfg = getattr(self.sim, "render", None)
        if render_cfg is not None:
            aa_mode = os.getenv("VAGEN_RENDER_AA_MODE", "OFF")
            if hasattr(render_cfg, "antialiasing_mode"):
                render_cfg.antialiasing_mode = aa_mode
            # Disable denoiser by default for sharper edges in this task.
            if hasattr(render_cfg, "enable_denoiser"):
                render_cfg.enable_denoiser = os.getenv("VAGEN_RENDER_ENABLE_DENOISER", "0") in {"1", "true", "True"}
            # Motion blur can produce "semi-transparent" trails after fast pose changes.
            if hasattr(render_cfg, "enable_motion_blur"):
                render_cfg.enable_motion_blur = os.getenv("VAGEN_RENDER_ENABLE_MOTION_BLUR", "0") in {"1", "true", "True"}

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        if hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
            self.sim.physx.enable_external_forces_every_iteration = True

    def build_server_runtime(
        self,
        *,
        env,
        cube_names,
        cube_size: float,
        max_tasks: int,
        grid_origin,
        cell_size: float,
        grid_size: int,
    ):
        """Unified runtime entrypoint used by server side."""
        builder = getattr(self, "runtime_builder", None)
        if callable(builder):
            return builder(
                env=env,
                cube_names=cube_names,
                cube_size=cube_size,
                max_tasks=max_tasks,
                grid_origin=grid_origin,
                cell_size=cell_size,
                grid_size=grid_size,
            )
        return mdp.build_franka_runtime(
            env=env,
            cube_names=cube_names,
            cube_size=cube_size,
            max_tasks=max_tasks,
            grid_origin=grid_origin,
            cell_size=cell_size,
            grid_size=grid_size,
        )
