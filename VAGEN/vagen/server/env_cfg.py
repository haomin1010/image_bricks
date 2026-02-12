import torch
import copy
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

def get_stack_cube_env_cfg(task_name, device, num_envs, enable_cameras=True):
    """
    Generate and modify the environment configuration for the stacking task.
    Adapted from stack_cube_sm.py.
    """
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    
    # --- Performance and Scene Setup ---
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.replicate_physics = (num_envs > 1)
    env_cfg.scene.lazy_sensor_update = False 
    
    # Disable internal Suction and Action Managers if present to use Magic Suction
    if hasattr(env_cfg.scene, "surface_gripper"):
        env_cfg.scene.surface_gripper = None
    
    for action_key in ["gripper_action", "suction_gripper", "gripper"]:
        if hasattr(env_cfg.actions, action_key):
            setattr(env_cfg.actions, action_key, None)

    if hasattr(env_cfg.observations, "policy"):
        if hasattr(env_cfg.observations.policy, "gripper_pos"):
            env_cfg.observations.policy.gripper_pos = None
    
    if hasattr(env_cfg.observations, "subtask_terms"):
        env_cfg.observations.subtask_terms = None

    # Fix UR10 IK offset
    if "UR10" in task_name and hasattr(env_cfg.actions, "arm_action"):
        env_cfg.actions.arm_action.body_offset.pos = (0.159, 0.0, 0.0)

    # Grid Constants
    grid_origin = [0.5, 0.0, 0.001]
    grid_size = 6
    line_thickness = 0.001
    cell_size = 0.055 + line_thickness 
    half_width = (grid_size * cell_size) / 2
    cube_size = 0.045
    
    # Spawning Cubes (8 cubes)
    max_cubes = 8
    blue_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
    cube_names = [f"cube_{i+1}" for i in range(max_cubes)]
    source_pick_pos_x = 0.3
    source_pick_pos_y = -0.2
    
    aligned_poses = []
    for i in range(max_cubes):
        if i == 0:
            aligned_poses.append([source_pick_pos_x, source_pick_pos_y, cube_size / 2.0, 1.0, 0.0, 0.0, 0.0])
        else:
            aligned_poses.append([-0.5 - (i * 0.1), 0.0, -1.0, 1.0, 0.0, 0.0, 0.0])

    # Clean existing cubes
    for default_name in ["cube_1", "cube_2", "cube_3"]:
        if hasattr(env_cfg.scene, default_name):
            getattr(env_cfg.scene, default_name).init_state.pos = (0.0, 0.0, -10.0)

    # Configure Blue Cubes
    for i, name in enumerate(cube_names):
        pos = aligned_poses[i]
        if i < 3: 
            cube_cfg = getattr(env_cfg.scene, name)
            cube_cfg.spawn.usd_path = blue_usd
            cube_cfg.init_state.pos = (pos[0], pos[1], pos[2])
            cube_cfg.init_state.rot = (pos[3], pos[4], pos[5], pos[6])
        else: 
            new_cube = copy.deepcopy(env_cfg.scene.cube_1)
            new_cube.prim_path = f"{{ENV_REGEX_NS}}/Cube_{i+1}"
            new_cube.spawn.semantic_tags = [("class", name)]
            new_cube.init_state.pos = (pos[0], pos[1], pos[2])
            new_cube.init_state.rot = (pos[3], pos[4], pos[5], pos[6])
            setattr(env_cfg.scene, name, new_cube)

    # Terminations and Episode logic
    if hasattr(env_cfg, "terminations"):
        for term_name in list(env_cfg.terminations.__dict__.keys()):
            if "cube" in term_name or "success" in term_name:
                setattr(env_cfg.terminations, term_name, None)
    env_cfg.episode_length_s = 600.0

    # Camera Setup
    default_cams = ["table_cam", "table_high_cam", "robot_cam", "cam_default"]
    for d_cam in default_cams:
        if hasattr(env_cfg.scene, d_cam):
            setattr(env_cfg.scene, d_cam, None)

    if enable_cameras:
        from isaaclab.sensors import TiledCameraCfg
        camera_ring_configs = {
            "cam_front_left": ([-0.1, 0.5, 0.6], [0.0, 0.7, -0.7]), 
            "cam_front_right": ([-0.1, -0.5, 0.6], [0.0, 0.7, 0.7]),
            "cam_back_left": ([1.1, 0.5, 0.6], [0.0, 0.7, -2.4]),
        }
        for cam_name, (pos, euler) in camera_ring_configs.items():
            q = math_utils.quat_from_euler_xyz(
                torch.tensor([euler[0]], device=device),
                torch.tensor([euler[1]], device=device),
                torch.tensor([euler[2]], device=device)
            ).tolist()[0]
            setattr(env_cfg.scene, cam_name, TiledCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{cam_name}",
                update_period=0.0,
                height=224, 
                width=224,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, 
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 1000.0),
                ),
                offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=(q[0], q[1], q[2], q[3]), convention="world"),
            ))

    # Grid Lines
    origin = grid_origin
    for i in range(grid_size + 1):
        suffix = f"_{i}"
        y_pos = origin[1] - half_width + i * cell_size
        setattr(env_cfg.scene, f"grid_h{suffix}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_h{suffix}",
            spawn=sim_utils.CuboidCfg(
                size=(grid_size * cell_size + line_thickness, line_thickness, 0.0002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0], y_pos, origin[2]))
        ))
        x_pos = origin[0] - half_width + i * cell_size
        setattr(env_cfg.scene, f"grid_v{suffix}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_v{suffix}",
            spawn=sim_utils.CuboidCfg(
                size=(line_thickness, grid_size * cell_size + line_thickness, 0.0002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x_pos, origin[1], origin[2]))
        ))

    return env_cfg, cube_names, aligned_poses
