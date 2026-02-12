# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to spawn a simple scene with one cube and a camera, then take a snapshot.
"""
# python one_cube_gen.py --device cuda:0 --headless --enable_cameras

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Snapshot a single cube.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import os
import torch
import numpy as np
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# =============================================================================
# 1. 配置场景 (Configuration)
# =============================================================================
TABLE_HEIGHT = 1.03  # Seattle Lab Table 的桌面高度大约是 1.03米
CUBE_SIZE = 0.045


@configclass
class SnapshotSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a cube and camera."""

    # 1. 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    )

    # 2. 光照 (调暗一点，防止白色桌子过曝看不清)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            intensity=1500.0,  # 降低亮度
            color=(0.9, 0.9, 0.9),
            angle=30.0 # 稍微调整角度产生阴影
        )
    )

    # 3. 桌子 
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), # 桌子中心在 (0.55, 0), 底座在地面 Z=0
            rot=(1.0, 0.0, 0.0, 0.0), 
        ),
    )

    # 4. 蓝色方块
    target_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        # 固定位置：(0.5, 0.0) 是网格中心附近
        # Z=0.025 大约是标准方块的一半高度，保证放在地面上而不是地底下
        # 必须把方块放在桌面上，而不是地板上
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, TABLE_HEIGHT + CUBE_SIZE/2), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # 5. 相机
    # 放在方块左后方 (X=0.8, Z=0.5)，俯视看向方块
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0, # 0.0 = 每帧更新
        height=512,
        width=512,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # 位置：X=-0.2 (在方块前面), Z=0.6 (高处俯视)
            pos=(0.0, 0.0, 1.7), 
            rot=(0.866, 0.0, 0.5, 0.0),
            convention="world",
        ),
    )

# =============================================================================
# 2. 主函数
# =============================================================================

def main():
    # 使用 CUDA
    if args_cli.device == "cpu":
        args_cli.device = "cuda:0"

    # 1. 设置仿真上下文
    sim_cfg = SimulationCfg(
        device=args_cli.device, 
        dt=0.01,          # 物理步长
        render_interval=1 # 强制每步都渲染
    )
    sim = SimulationContext(sim_cfg)

    # 2. 创建场景配置实例
    scene_cfg = SnapshotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    # --- 动态添加网格线 (Grid) ---
    grid_origin = [0.5, 0.0, TABLE_HEIGHT + 0.001]
    grid_size = 6
    cell_size = 0.056
    line_thickness = 0.002
    half_width = (grid_size * cell_size) / 2
    
    for i in range(grid_size + 1):
        suffix = f"_{i}"
        # 横线
        y_pos = grid_origin[1] - half_width + i * cell_size
        setattr(scene_cfg, f"grid_h{suffix}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_h{suffix}",
            spawn=sim_utils.CuboidCfg(
                size=(grid_size * cell_size + line_thickness, line_thickness, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(grid_origin[0], y_pos, 0.001))
        ))
        # 竖线
        x_pos = grid_origin[0] - half_width + i * cell_size
        setattr(scene_cfg, f"grid_v{suffix}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_v{suffix}",
            spawn=sim_utils.CuboidCfg(
                size=(line_thickness, grid_size * cell_size + line_thickness, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x_pos, grid_origin[1], 0.001))
        ))

    # 3. 创建场景
    scene = InteractiveScene(scene_cfg)

    # 4. 重置并播放仿真
    print("[INFO]: Resetting simulation...")
    sim.reset()

    # 5. 预热渲染 (Warmup)
    # 必须跑几十步，让光照渲染和物理位置（方块落地）稳定下来
    print("[INFO]: Warming up physics and rendering...")
    for _ in range(50): 
        sim.step()
        # 必须更新场景状态，否则相机获取不到最新的物体位置
        scene.update(dt=sim_cfg.dt)

    # 6. 拍照
    print("[INFO]: Capturing image...")
    camera: Camera = scene["camera"]
    
    # 获取 RGB 数据 [num_envs, H, W, 3]
    rgb_tensor = camera.data.output["rgb"]
    
    # 7. 保存图片
    output_dir = "output_snapshots"
    os.makedirs(output_dir, exist_ok=True)

    img_data = rgb_tensor[0]
    
    if isinstance(img_data, torch.Tensor):
        img_np = img_data.cpu().numpy()
    else:
        img_np = img_data
    
    # 去除 Alpha 通道
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    save_path = os.path.join(output_dir, "one_cube_final.png")
    Image.fromarray(img_np.astype(np.uint8)).save(save_path)
    print(f"[SUCCESS]: Image saved to {save_path}")

    # ================= 修改开始 =================
    # 如果是为了调试观看，不要直接关闭 app，而是进入循环
    print("[INFO]: Simulation is paused for viewing. Press Ctrl+C to exit.")
    
    while simulation_app.is_running():
        # 继续进行物理步进和渲染，这样你在画面里还能看到东西在动（如果有物理）
        # 如果只想静止，可以只调 scene.update() 或 simulation_app.update()
        sim.step()
        scene.update(dt=sim_cfg.dt)
    # ================= 修改结束 =================

    simulation_app.close()

if __name__ == "__main__":
    main()