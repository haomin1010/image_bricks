数据集生成
整个工作涉及到模型对于积木空间结构的理解与搭建的长程规划，以及机械臂在非结构化环境下的灵巧操作与错误恢复，涉及到包括空间推理，物理直觉，运动规划，强化学习等多方面话题，在短期内，我们的目标是：
实现基于仿真环境的VLM交互框架，VLM基于多轮交互控制底层IK算法，实现结构化相同积木的搭建。总共分为两部分工作：
1. 搭建结构多样的评测数据集，并且对于现有开闭源模型对于物理搭建的开环规划能力进行详细评测与分析。
2. 实现基于仿真环境的多轮交互训练与评测框架，评测模型在实际物理世界交互的空间推理与规划能力，并且提出可行的训练范式，利用与仿真环境的交互提高模型对于空间的理解能力。
任务：给定仿真中的积木结构，通过提供三视图、轴测图等信息，模型自主规划搭建与移动顺序，搭建对应三维结构。
难度：积木为完全相同的正方体，整齐排列，结构整体符合物理规律。
目标：模型应当通过开环推理或与仿真环境的逐步交互，推理出符合物理规律与空间依赖关系的搭建序列，并且最终搭建的结构应当与实际结构保持一致。
使用NVIDIA Isaac Sim 5.1.0
Isaac Sim 的优势，适合用来测试 VLM
1. 物理真实性 (PhysX)：自带物理引擎，让积木自然掉落堆叠，或者使用脚本摆放并检测碰撞。这天然保证了“物理可搭建”和“重力稳定”。
2. 光影真实性 (RTX)：支持光线追踪（Ray Tracing），可以生成逼真的阴影、遮挡、反射和材质纹理。这对于测试 VLM 在真实世界的感知能力至关重要（避免 VLM 只是在“做几何题”，而是在“看图”）。
3. 数据标注自动化 (Replicator)：可以通过 Python 脚本自动获取每一个方块的精确坐标完全不需要人工标注。
实施路线图
1. 场景搭建 (World Setup)
- 积木设计：创建一个标准的 Cube（立方体）作为预制体（Asset）。
- 颜色/纹理：Currently, we use the same color and mesh texture
- 环境：设置一个简单的地面和 HDRI 光照环境。建议使用纯色或简单的网格地面，减少背景干扰
2. 程序化生成 (Procedural Generation)
- 复用之前的逻辑：把之前写的 generate_stable_convex 算法逻辑移植进去
```
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os

class CustomSizeBuilder:
    def __init__(self):
        # === 核心要求：坐标系边界固定为 8x8x8 ===
        self.L = 8
        self.W = 8
        self.H = 8
        self.grid = np.zeros((8, 8, 8), dtype=bool)

    def generate_object(self, init_size=(8, 8, 8), scale_mode="medium"):
        """
        :param init_size: (lx, ly, lz) 原始未切割方块的大小 (必须 <= 8)
        :param scale_mode: 'small', 'medium', 'large'
        """
        # 1. 参数校验
        ix, iy, iz = init_size
        ix, iy, iz = min(ix, 8), min(iy, 8), min(iz, 8)
        
        print(f"⚙️  初始化: 画布 8x8x8 | 初始方块 {ix}x{iy}x{iz} | 模式: {scale_mode}")

        # 2. 设定参数
        if scale_mode == "small":
            num_cuts = 10; cut_offset_range = 0.5
        elif scale_mode == "medium":
            num_cuts = 5; cut_offset_range = 1.0
        else: # "large"
            num_cuts = 3; cut_offset_range = 2.0

        # 3. 初始化高度图
        height_map = np.zeros((self.L, self.W))
        start_x = (self.L - ix) // 2
        start_y = (self.W - iy) // 2
        height_map[start_x : start_x+ix, start_y : start_y+iy] = float(iz)

        # 4. 执行削山法 (带“防毁灭”校验)
        center_x, center_y, center_z = self.L / 2, self.W / 2, iz / 2

        for i in range(num_cuts):
            # 随机生成平面
            angle = random.uniform(0, 2 * np.pi)
            tilt = random.uniform(0.5, 2.0) # 稍微减小最大倾斜度，防止切太狠
            nx, ny, nz = np.cos(angle)*tilt, np.sin(angle)*tilt, 1.0
            
            offset_factor = random.uniform(0.5, 1.5) * cut_offset_range
            px = center_x + (random.uniform(-1, 1) * ix/2 * offset_factor)
            py = center_y + (random.uniform(-1, 1) * iy/2 * offset_factor)
            
            # 【优化点1】提高最低切割高度，保护底座
            # 之前是 iz * 0.3，现在提高到 iz * 0.5，保证至少保留一半高度的中心点
            pz = center_z + (random.uniform(-1, 1) * iz/2 * offset_factor)
            pz = max(pz, iz * 0.5) 

            # 计算平面截距
            x_idx, y_idx = np.indices((self.L, self.W))
            plane_z = pz - (nx/nz)*(x_idx - px) - (ny/nz)*(y_idx - py)
            
            # 【优化点2：核心修复】试算一下，看看切完剩多少
            temp_height_map = np.minimum(height_map, plane_z)
            
            # 检查：如果切完后，最高点连 1.0 都不到（说明全没了），或者体积太小
            # np.max(temp_height_map) 获取当前最高的高度
            if np.max(temp_height_map) < 1.0:
                # print(f"  ⚠️ 第 {i+1} 刀太狠了，跳过 (会导致物体消失)")
                continue # 放弃这一刀，进入下一次循环
            
            # 如果检查通过，才真的应用切割
            height_map = temp_height_map

        # 5. 转为体素
        self.grid = np.zeros((self.L, self.W, self.H), dtype=bool)
        for x in range(self.L):
            for y in range(self.W):
                h = int(np.clip(height_map[x, y], 0, self.H))
                if h > 0:
                    self.grid[x, y, :h] = True
        
        block_count = np.sum(self.grid)
        print(f"✅ 生成完成！最终方块数: {block_count}")
        
        # 【兜底机制】如果万一还是0 (极小概率)，递归重试一次
        if block_count == 0:
            print("⚠️ 结果异常，正在自动重试...")
            self.generate_object(init_size, scale_mode)

    # ================= 以下是绘图辅助功能 (保持不变) =================
    def save_coordinates(self, filename="voxel_coordinates.csv"):
        if not np.any(self.grid): return
        coords = np.argwhere(self.grid)
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                writer.writerows(coords)
            print(f"💾 坐标已保存至: {os.path.abspath(filename)}")
        except Exception: pass

    def _draw_axis_panel(self, ax, data_bool, xlabel, ylabel, title, arrow_color):
        rows, cols = data_bool.shape
        X, Y = np.meshgrid(np.arange(cols + 1), np.arange(rows + 1))
        cmap = plt.cm.viridis.copy(); cmap.set_under('white')
        ax.pcolormesh(X, Y, data_bool.astype(float), cmap=cmap, vmin=0.1, vmax=1.0, edgecolors='k', linewidth=1.5, shading='flat')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold', color=arrow_color)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(0, cols + 1, 1)); ax.set_yticks(np.arange(0, rows + 1, 1))
        ax.grid(True, linestyle=':', alpha=0.5)

    def _draw_3d_overview(self, ax):
        ax.voxels(self.grid, edgecolor='k', linewidth=0.8, alpha=0.9, cmap='viridis')
        # Arrows
        cx, cy, cz = 4, 4, 10
        ax.quiver(cx, cy, cz, 0, 0, -2, color='red', linewidth=2, arrow_length_ratio=0.3)
        ax.text(cx, cy, cz, "Top", color='red', ha='center', va='bottom')
        ax.quiver(cx, -2, 4, 0, 2, 0, color='green', linewidth=2, arrow_length_ratio=0.3)
        ax.text(cx, -2, 4, "Front", color='green', ha='center', va='top')
        ax.quiver(10, cy, 4, -2, 0, 0, color='blue', linewidth=2, arrow_length_ratio=0.3)
        ax.text(10, cy, 4, "Side", color='blue', ha='center', va='bottom')
        # Axis
        ax.set_box_aspect((8, 8, 8))
        ax.set_xticks(np.arange(0, 9, 1)); ax.set_yticks(np.arange(0, 9, 1)); ax.set_zticks(np.arange(0, 9, 1))
        ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_zlim(0, 8)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title("3D View (8x8x8 Canvas)", fontweight='bold')
        ax.view_init(elev=25, azim=-55)

    def save_all_images(self, output_dir="custom_output"):
        if not np.any(self.grid): return
        os.makedirs(output_dir, exist_ok=True)
        top_view = np.any(self.grid, axis=2).T
        front_view = np.any(self.grid, axis=1).T
        side_view = np.any(self.grid, axis=0).T
        
        # Save Combined
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0]); self._draw_axis_panel(ax1, top_view, "X", "Y", "Top View", "red")
        ax2 = fig.add_subplot(gs[0, 1], projection='3d'); self._draw_3d_overview(ax2)
        ax3 = fig.add_subplot(gs[1, 0]); self._draw_axis_panel(ax3, front_view, "X", "Z", "Front View", "green")
        ax4 = fig.add_subplot(gs[1, 1]); self._draw_axis_panel(ax4, side_view, "Y", "Z", "Side View", "blue")
        plt.savefig(os.path.join(output_dir, "combined.png"), dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    builder = CustomSizeBuilder()
    
    # ---------------- 用户设置区域 ----------------
    # 1. 设置原始未切割方块大小 (长, 宽, 高)，最大不超过 8
    initial_block_size = (8, 8, 8)  
    
    # 2. 设置生成模式: 'small', 'medium', 'large'
    target_scale = "medium"
    # ---------------------------------------------
    
    builder.generate_object(init_size=initial_block_size, scale_mode=target_scale)
    builder.save_coordinates("coords.csv")
    builder.save_all_images("custom_output")

```

3. 相机配置 (Camera Setup)
固定内参。
在 Isaac Sim 中，创建一个 Camera Prim，并设置以下参数以模拟真实相机：
- Focal Length (焦距)
- Horizontal Aperture (传感器宽度)
- Resolution (分辨率)：例如 224*224。
- Position (外参)：脚本化设置 5 个位置：
  
  1. Top (俯视)
  2. Front (正视)
  3. Side (侧视)
  "Telephoto 45° High-Angle View" (长焦45度高俯角视图)
  - 视角： 俯仰角 45°，方位角 45°。
  4. Iso_1 (右前上)
  5. Iso_2 (左后上)
4. Ground Truth 导出 (JSON)
在拍摄图像的同时，遍历场景中所有的积木 Prim，读取它们的属性。