# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to generate snapshots of a table with a grid and blocks based on JSON input.
Usage:
    python batch_gen.py --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Snapshot a table with grid.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--json_file", type=str, default="convex_01.json", help="Name of the JSON file to process")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import os
import json
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
# 0. 配置输入文件路径 (Configuration)
# =============================================================================

# 指定 JSON 文件的路径
JSON_FOLDER = "convex_json_batch"
JSON_FILENAME = args_cli.json_file
JSON_PATH = os.path.join(JSON_FOLDER, JSON_FILENAME)

# 提取文件名（不含扩展名），用于命名输出文件夹和图片
FILE_ID = os.path.splitext(JSON_FILENAME)[0]  # e.g., "convex_01"

# =============================================================================
# 1. 定义常量 (Constants)
# =============================================================================

# Seattle Lab Table 的桌面高度通常约为 1.03m
TABLE_HEIGHT = 1.03 
CUBE_SIZE = 0.05
GRID_ORIGIN = [-0.24, -0.24, TABLE_HEIGHT + 0.001]  # 网格原点(0,0)位置 - 左下角
CAMERA_HEIGHT = 0.7  # 相机离桌面的高度
GRID_SIZE = 8  # 网格行列数 (8x8 坐标系统：A-H, 1-8)
CENTRAL_HEIGHT = CUBE_SIZE * GRID_SIZE / 2.0


def load_block_data(file_path):
    """从JSON文件加载方块坐标"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find JSON file at: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    coords = []
    # 遍历 blocks 列表提取 x, y, z
    if "blocks" in data:
        for block in data["blocks"]:
            # JSON中的坐标直接对应网格坐标
            coords.append((block['x'], block['y'], block['z']))
    
    print(f"[INFO]: Loaded {len(coords)} blocks from {file_path}")
    return coords, data

# 加载坐标
BLOCK_COORDS, RAW_JSON_DATA = load_block_data(JSON_PATH)


@configclass
class SnapshotSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a table, grid, cube and camera."""

    # 1. 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    )

    # 2. 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.9, 0.9, 0.9),
            angle=30.0
        )
    )

    light2 = AssetBaseCfg(
        prim_path="/World/light2",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.9, 0.9, 0.9),
            angle=60.0
        )
    )

    # 3. 球形灯光 - 局部补充照明
    sphere_light = AssetBaseCfg(
        prim_path="/World/sphere_light",
        spawn=sim_utils.SphereLightCfg(
            intensity=1500.0,
            color=(1.0, 0.9, 0.7),  # 温暖的黄色调
            radius=0.1,  # 灯光的半径
            enable_color_temperature=True,
            color_temperature=3000.0,  # 温暖的色温
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.5, TABLE_HEIGHT + 1.0),  # 在桌子上方侧面
        ),
    )


    # 3. 桌子 (Seattle Lab Table)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT), 
            rot=(1.0, 0.0, 0.0, 0.0), 
        ),
    )

    # 4. 相机配置
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT + CAMERA_HEIGHT),
            rot=(0.707, 0.0, 0.707, 0.0), # Top View
            convention="world",
        ),
    )

    camera_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFront",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(CAMERA_HEIGHT, 0.0, TABLE_HEIGHT + CUBE_SIZE * 2), 
            rot=(0.0, 0.0, 0.0, 1.0), # Front View
            convention="world",
        ),
    )

    camera_side = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraSide",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, CAMERA_HEIGHT, TABLE_HEIGHT + CUBE_SIZE * 2), 
            rot=(0.707, 0.0, 0.0, -0.707), # Side View
            convention="world",
        ),
    )

    camera_iso = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraIso",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-CAMERA_HEIGHT / np.sqrt(2), CAMERA_HEIGHT / np.sqrt(2), TABLE_HEIGHT + CAMERA_HEIGHT), 
            rot=(0.85355, 0.14645, 0.35355, -0.35355),
            convention="world",
        ),
    )

    camera_iso2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraIso2",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(CAMERA_HEIGHT / np.sqrt(2), -CAMERA_HEIGHT / np.sqrt(2), TABLE_HEIGHT + CAMERA_HEIGHT), 
           
            rot=( 0.36, -0.33, 0.14, 0.85),
            convention="world",
        ),
    )



def add_blocks_from_coords(scene_cfg_cls, coords_list, origin):
    """动态向场景配置添加方块"""
    
    def grid_to_world_pos(grid_x, grid_y, grid_z):
        """将网格坐标转换为世界坐标"""
        # 注意：这里假设 JSON 中的 x, y 也是对应网格索引
        world_x = origin[0] + grid_x * 0.051 + 0.0255
        world_y = origin[1] + grid_y * 0.051 + 0.0255
        world_z = TABLE_HEIGHT + CUBE_SIZE/2 + grid_z * CUBE_SIZE
        return (world_x, world_y, world_z)
    
    for i, (gx, gy, gz) in enumerate(coords_list):
        world_pos = grid_to_world_pos(gx, gy, gz)
        
        # 使用蓝色方块
        cube_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Block_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=world_pos,
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )
        setattr(scene_cfg_cls, f"block_{i}", cube_cfg)


def add_blocks_from_coords(scene_cfg_cls, coords_list, origin):
    """动态向场景配置添加方块"""
    
    def grid_to_world_pos(grid_x, grid_y, grid_z):
        """将网格坐标转换为世界坐标"""
        # 注意：这里假设 JSON 中的 x, y 也是对应网格索引
        world_x = origin[0] + grid_x * 0.051 + 0.0255
        world_y = origin[1] + grid_y * 0.051 + 0.0255
        world_z = TABLE_HEIGHT + CUBE_SIZE/2 + grid_z * CUBE_SIZE
        return (world_x, world_y, world_z)
    
    for i, (gx, gy, gz) in enumerate(coords_list):
        world_pos = grid_to_world_pos(gx, gy, gz)
        
        # 使用蓝色方块
        cube_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Block_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=world_pos,
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )
        setattr(scene_cfg_cls, f"block_{i}", cube_cfg)


# 动态添加网格线 (Grid Injection)
# 我们在类定义之后、实例化之前，通过代码向 Config 类注入属性

def add_coordinate_grid_to_scene_config(scene_cfg_cls, origin, grid_size=8, cell_size=0.051, gap_width=0.001):
    """
    动态向 SceneCfg 类添加带坐标标注的网格系统。
    origin: [x, y, z] 网格右下角原点 (0,0) 位置
    grid_size: 网格大小 (8x8 方格，需要9条线)
    cell_size: 每个方格的大小（对应方块大小）
    gap_width: 方块间隙宽度（等于线条宽度）
    """
    
    # 定义材质
    normal_line_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.0, roughness=1.0)
    x_axis_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.0, roughness=0.2)  # 鲜红色 X轴
    y_axis_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.0, roughness=0.2)  # 鲜绿色 Y轴
    coord_marker_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 1.0), metallic=0.0, roughness=1.0)  # 蓝色标记
    
    # === 网格线创建（从原点开始）===
    # 需要 grid_size+1 条线来分割 grid_size 个方格
    for i in range(grid_size + 1):
        
        # --- 横线 (平行于X轴，沿 Y 方向排列) ---
        y_pos = origin[1] + i * cell_size  # 从原点开始，Y坐标 0,1,2...8
        
        # Y=0 轴线(第一条横线)用绿色，其他用普通颜色
        material = y_axis_material if i == 0 else normal_line_material
        line_height = 0.003 if i == 0 else 0.001  # Y轴稍微厚一点
        
        h_line_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_h_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(grid_size * cell_size, gap_width, line_height),  # 线宽等于间隙宽度
                visual_material=material,
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(origin[0] + grid_size * cell_size / 2, y_pos, origin[2])  # 中心位置
            )
        )
        setattr(scene_cfg_cls, f"grid_h_{i}", h_line_cfg)

        # --- 竖线 (平行于Y轴，沿 X 方向排列) ---
        x_pos = origin[0] + i * cell_size  # 从原点开始，X坐标 0,1,2...8
        
        # X=0 轴线(第一条竖线)用红色，其他用普通颜色
        material = x_axis_material if i == 0 else normal_line_material
        line_height = 0.003 if i == 0 else 0.001  # X轴稍微厚一点
        
        v_line_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/grid_v_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(gap_width, grid_size * cell_size, line_height),  # 线宽等于间隙宽度
                visual_material=material,
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(x_pos, origin[1] + grid_size * cell_size / 2, origin[2])  # 中心位置
            )
        )
        setattr(scene_cfg_cls, f"grid_v_{i}", v_line_cfg)
    
    # === 原点标记创建 ===
    origin_marker_cfg = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/origin_marker",
        spawn=sim_utils.CuboidCfg(
            size=(0.018, 0.018, 0.006),  # 更响亮的原点标记
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.1, roughness=0.1),  # 黄色
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(origin[0], origin[1], origin[2] + 0.001)  # 稍微抬高以突出显示
        )
    )
    setattr(scene_cfg_cls, "origin_marker", origin_marker_cfg)


# 执行注入: 传入从 JSON 读取的 BLOCK_COORDS
add_blocks_from_coords(SnapshotSceneCfg, BLOCK_COORDS, GRID_ORIGIN)
add_coordinate_grid_to_scene_config(SnapshotSceneCfg, GRID_ORIGIN)


# 主函数
def main():
    if args_cli.device == "cpu":
        args_cli.device = "cuda:0"

    sim_cfg = SimulationCfg(device=args_cli.device, dt=0.01)
    sim = SimulationContext(sim_cfg)

    scene_cfg = SnapshotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    print("[INFO]: Resetting simulation...")
    sim.reset()

    print(f"[INFO]: Warming up for {FILE_ID}...")
    for _ in range(30): # 稍微增加一点预热时间，因为方块多了
        sim.step()
        scene.update(dt=sim_cfg.dt)

    print("[INFO]: Capturing images...")
    
    # 获取所有相机数据
    camera_map = {
        "top": scene["camera"],
        "iso": scene["camera_iso"],
        "front": scene["camera_front"],
        "side": scene["camera_side"],
        "iso2": scene["camera_iso2"]
    }

    # 设置输出目录
    base_output_dir = "output_snapshots"
    # 使用 JSON 文件名作为子目录名 (e.g. "output_snapshots/convex_01")
    shape_output_dir = os.path.join(base_output_dir, FILE_ID)
    os.makedirs(shape_output_dir, exist_ok=True)
    
    image_filenames = []
    saved_views = []

    for view_name, camera_obj in camera_map.items():
        # 提取图像数据
        rgb_data = camera_obj.data.output["rgb"][0].cpu().numpy()
        
        if rgb_data.shape[-1] == 4:
            rgb_data = rgb_data[..., :3]
            
        # 格式化文件名: {FILE_ID}_{VIEW}.png -> e.g., convex_01_top.png
        filename = f"{FILE_ID}_{view_name}.png"
        filepath = os.path.join(shape_output_dir, filename)
        
        Image.fromarray(rgb_data.astype(np.uint8)).save(filepath)
        image_filenames.append(filename)
        saved_views.append(view_name)
        print(f"[SUCCESS]: Saved {filepath}")

    # 计算世界坐标用于记录
    def grid_to_world_pos(grid_x, grid_y, grid_z):
        world_x = GRID_ORIGIN[0] + grid_x * 0.051 + 0.0255
        world_y = GRID_ORIGIN[1] + grid_y * 0.051 + 0.0255
        world_z = TABLE_HEIGHT + CUBE_SIZE/2 + grid_z * CUBE_SIZE
        return {"world_x": world_x, "world_y": world_y, "world_z": world_z}

    # 创建完整的输出 JSON
    output_data = {
        "source_file": JSON_FILENAME,
        "shape_id": FILE_ID,
        "total_blocks": len(BLOCK_COORDS),
        # 包含原始的 JSON 数据
        "original_data": RAW_JSON_DATA, 
        # 添加生成的图片信息
        "generated_images": image_filenames,
        "camera_views": saved_views,
        "world_coordinates": [
            grid_to_world_pos(c[0], c[1], c[2]) for c in BLOCK_COORDS
        ],
        "grid_info": {
            "origin": GRID_ORIGIN,
            "cell_size": 0.051
        }
    }
    
    # 保存结果 JSON
    result_json_path = os.path.join(shape_output_dir, f"{FILE_ID}_data.json")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS]: Data and images saved to {shape_output_dir}/")


    print(f"[INFO]: Finished processing {FILE_ID}. Closing application...")
    # simulation_app.close()

    os._exit(0)
if __name__ == "__main__":
    main()