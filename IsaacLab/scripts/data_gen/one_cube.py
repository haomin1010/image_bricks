# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
脚本功能：在固定位置生成单个蓝色方块，创建相机并拍摄照片保存。
使用说明：
    ./isaaclab.sh -p scripts/tutorials/cube_photo_capture.py --device cuda:0
"""

import argparse
import os
import torch
from PIL import Image

from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationCfg, SimulationContext, sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import Camera, CameraCfg

# 命令行参数解析
parser = argparse.ArgumentParser(description="生成单个方块并使用相机拍照")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--save_path", type=str, default="./cube_photo.png", help="照片保存路径")
parser.add_argument("--cube_pos", nargs=3, type=float, default=[0.5, 0.0, 0.0203], help="方块位置 (x y z)")
args_cli = parser.parse_args()

# 启动仿真应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    """核心逻辑：创建仿真场景、生成方块、配置相机、拍摄并保存照片"""
    # 1. 初始化仿真上下文（修复sim_cfg.RenderCfg的语法错误）
    sim_cfg = SimulationCfg(
        dt=0.01,
        device=args_cli.device,  # 支持指定设备（cuda/cpu）
        render=sim_utils.RenderCfg(
            enable_cameras=True,  # 启用相机渲染（关键：必须开启才能捕获图像）
            antialiasing_mode="DLAA"  # 抗锯齿，提升画质
        )
    )
    # 初始化仿真上下文，指定backend为torch
    sim = SimulationContext(sim_cfg, backend="torch")
    # 设置主视角（仅可视化用，不影响拍照相机）
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # 2. 定义地面（可选：让方块有支撑，避免悬空）
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg, translation=(0.0, 0.0, 0.0))

    # 3. 定义方块配置（单个蓝色方块）
    cube_properties = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        disable_gravity=False,  # 启用重力，让方块落在地面
        max_depenetration_velocity=5.0,  # 增加防穿透参数，提升物理稳定性
    )
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",  # 方块在场景中的USD路径
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=args_cli.cube_pos,  # 从命令行接收的固定位置
            rot=[1.0, 0.0, 0.0, 0.0]  # 无旋转（四元数）
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
        ),
    )

    # 4. 生成方块到场景中（修复初始化方式）
    cube = RigidObject(cube_cfg)
    cube.spawn(sim, "/World")  # 生成USD prim
    cube.initialize(sim)  # 初始化物理句柄

    # 5. 配置拍照相机（固定视角对准方块）
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",  # 相机在场景中的路径
        update_period=0.0,  # 每帧更新（0=无延迟）
        height=512,  # 照片高度
        width=512,   # 照片宽度
        data_types=["rgb"],  # 仅捕获RGB图像
        spawn=sim_utils.PinholeCameraCfg(  # 修复SpawnCfg的正确引用
            focal_length=24.0,
            focus_distance=1.0,  # 对焦距离（方块在0.5m位置，1m足够）
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),  # 裁剪范围（近/远裁剪面）
        ),
        offset=CameraCfg.OffsetCfg(
            pos=[0.8, 0.0, 0.5],  # 相机位置（方块右前方，略高于方块）
            rot=[0.7071, 0.0, 0.0, 0.7071],  # 旋转（看向方块），四元数格式
            convention="world"  # 以世界坐标系为基准
        ),
    )

    # 6. 生成相机并初始化
    camera = Camera(camera_cfg)
    camera.spawn(sim, "/World")  # 生成相机prim
    camera.initialize(sim)  # 初始化相机传感器

    # 7. 启动仿真并稳定物理
    sim.reset()  # 重置仿真（触发物理初始化）
    # 运行几帧让物理稳定（防止方块刚生成时位置未完全稳定）
    for _ in range(20):  # 增加帧数，确保方块落地稳定
        sim.step()  # 步进仿真
        camera.update(sim)  # 每帧更新相机数据

    # 8. 捕获相机数据并保存照片（核心步骤）
    # 强制更新相机最新数据
    camera.update(sim)
    # 获取RGB数据：格式为 (H, W, 3)，值范围[0,1]
    rgb_data = camera.data.output["rgb"]
    # 处理数据格式：从torch tensor转numpy，缩放至[0,255]并转uint8
    if isinstance(rgb_data, torch.Tensor):
        rgb_img = (rgb_data.cpu().numpy() * 255).astype("uint8")
    else:
        rgb_img = (rgb_data * 255).astype("uint8")
    # 转换为PIL图像并保存
    img = Image.fromarray(rgb_img)
    # 确保保存目录存在（处理路径包含子目录的情况）
    os.makedirs(os.path.dirname(args_cli.save_path), exist_ok=True)
    # 保存图片
    img.save(args_cli.save_path)
    print(f"✅ 照片已成功保存至: {os.path.abspath(args_cli.save_path)}")

    # 9. 关闭仿真（优雅退出）
    sim.stop()
    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        simulation_app.close()