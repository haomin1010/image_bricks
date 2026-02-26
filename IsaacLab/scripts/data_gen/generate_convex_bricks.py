import numpy as np
import random
import csv
import os
import json

class FinalViewBuilder:
    def __init__(self, length=8, width=8, height=8):
        self.L = length
        self.W = width
        self.H = height
        self.grid = np.zeros((length, width, height), dtype=bool)

    def generate_stable_convex(self, num_cuts=12):
        """生成严格物理稳定且凸的物体"""
        height_map = np.full((self.L, self.W), float(self.H))

        # 削山法生成凸物体
        for i in range(num_cuts):
            angle = random.uniform(0, 2 * np.pi)
            tilt = random.uniform(0.5, 2.5)
            nx, ny, nz = np.cos(angle)*tilt, np.sin(angle)*tilt, 1.0
            
            px = self.L/2 + random.uniform(-1, 1)
            py = self.W/2 + random.uniform(-1, 1)
            pz = random.uniform(self.H * 0.5, self.H * 1.5)

            x_idx, y_idx = np.indices((self.L, self.W))
            plane_z = pz - (nx/nz)*(x_idx - px) - (ny/nz)*(y_idx - py)
            height_map = np.minimum(height_map, plane_z)

        self.grid = np.zeros((self.L, self.W, self.H), dtype=bool)
        for x in range(self.L):
            for y in range(self.W):
                h = int(np.clip(height_map[x, y], 0, self.H))
                if h > 0:
                    self.grid[x, y, :h] = True

    def save_coordinates(self, filename="voxel_coordinates.csv"):
        """保存坐标到CSV文件"""
        if not np.any(self.grid):
            return
        coords = np.argwhere(self.grid)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(coords)

    def save_coordinates_json(self, filename="voxel_data.json"):
        """保存坐标到JSON文件"""
        if not np.any(self.grid):
            return
        coords = np.argwhere(self.grid).tolist()
        
        data = {
            "dimensions": {
                "length": self.L,
                "width": self.W,
                "height": self.H
            },
            "total_blocks": len(coords),
            "blocks": [
                {"id": i+1, "x": coord[0], "y": coord[1], "z": coord[2]}
                for i, coord in enumerate(coords)
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 使用 8x8x8 尺寸
    builder = FinalViewBuilder(length=8, width=8, height=8)
    
    # 生成凸形物体
    builder.generate_stable_convex(num_cuts=10)
    
    # 导出坐标到CSV
    builder.save_coordinates("coordinates.csv")
    
    # 导出坐标到JSON
    builder.save_coordinates_json("coordinates.json")