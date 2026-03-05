import numpy as np
import random
import csv
import os
import json

class SmallBricksBuilder:
    def __init__(self, length=8, width=8, height=8):
        self.L = length
        self.W = width
        self.H = height
        self.grid = np.zeros((length, width, height), dtype=bool)

    def generate_random_stable_bricks(self, num_blocks):
        """生成严格物理稳定且连通的物体，可以是凸的或非凸的"""
        self.grid = np.zeros((self.L, self.W, self.H), dtype=bool)
        
        # Ensure we have at least 1 block
        if num_blocks < 1:
            return

        # Start with a random position on the ground (z=0)
        start_x = random.randint(0, self.L - 1)
        start_y = random.randint(0, self.W - 1)
        self.grid[start_x, start_y, 0] = True
        current_blocks = 1
        
        while current_blocks < num_blocks:
            valid_positions = []
            # Find all positions that are adjacent to the existing structure
            # and satisfy the stability condition
            for x in range(self.L):
                for y in range(self.W):
                    for z in range(self.H):
                        if not self.grid[x, y, z]:
                            # 1. Must be strictly stable: z == 0 or supported by a block directly below
                            is_stable = (z == 0) or self.grid[x, y, z-1]
                            if not is_stable:
                                continue
                            
                            # 2. Must be connected to at least one existing block in 6 directions
                            is_connected = False
                            directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
                            for dx, dy, dz in directions:
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if 0 <= nx < self.L and 0 <= ny < self.W and 0 <= nz < self.H:
                                    if self.grid[nx, ny, nz]:
                                        is_connected = True
                                        break
                                        
                            if is_connected:
                                valid_positions.append((x, y, z))
                                
            if not valid_positions:
                break
                
            # Randomly pick a valid position and add a block
            px, py, pz = random.choice(valid_positions)
            self.grid[px, py, pz] = True
            current_blocks += 1

    def save_coordinates(self, filename="small_voxel_coordinates.csv"):
        """保存坐标到CSV文件"""
        if not np.any(self.grid):
            return
        coords = np.argwhere(self.grid).tolist()
        # 按 Z X Y 升序排序
        coords.sort(key=lambda c: (c[2], c[0], c[1]))
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(coords)

    def save_coordinates_json(self, filename="small_voxel_data.json"):
        """保存坐标到JSON文件"""
        if not np.any(self.grid):
            return
        coords = np.argwhere(self.grid).tolist()
        # 按 Z X Y 升序排序
        coords.sort(key=lambda c: (c[2], c[0], c[1]))
        
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
    # 使用 8x8x8 尺寸保证坐标范围在[0, 7]
    builder = SmallBricksBuilder(length=8, width=8, height=8)
    
    # 方块数量在[1, 10]之间
    num_blocks = random.randint(1, 10)
    print(f"Generating {num_blocks} blocks...")
    
    # 生成物体
    builder.generate_random_stable_bricks(num_blocks=num_blocks)
    
    # 导出坐标到CSV
    builder.save_coordinates("small_coordinates.csv")
    
    # 导出坐标到JSON
    builder.save_coordinates_json("small_coordinates.json")
    
    print("Generation complete.")
