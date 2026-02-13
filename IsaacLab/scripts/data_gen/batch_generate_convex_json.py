import os
import sys
import shutil

# 保证可以import到generate_convex_bricks.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_convex_bricks import FinalViewBuilder

output_dir = "convex_json_batch"
os.makedirs(output_dir, exist_ok=True)

for i in range(21, 100):
    builder = FinalViewBuilder(length=8, width=8, height=8)
    builder.generate_stable_convex(num_cuts=10)
    json_path = os.path.join(output_dir, f"convex_{i:02d}.json")
    builder.save_coordinates_json(json_path)

print(f"已批量生成79个json文件，保存在: {os.path.abspath(output_dir)}")
