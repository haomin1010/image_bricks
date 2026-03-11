"""
[File Description]
This script is STEP 1 of the data generation pipeline.
It triggers the algorithms to generate a batch of random block coordinates,
and saves them as purely coordinate-based `.json` files in the intermediate_json folder.
Original name: batch_generate_bricks.py
Usage: python step1_generate_json_batch.py --type <category> --num <count>
"""

import os
import sys
import glob
import argparse
import random

# 保证可以import到生成相关的脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generator_alg_large import FinalViewBuilder
from generator_alg_small import SmallBricksBuilder

def get_next_id(category_dir):
    """获取该目录下下一个可用的序号 ID"""
    os.makedirs(category_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(category_dir, "*.json"))
    
    max_id = 0
    for f in existing_files:
        basename = os.path.basename(f)
        # Assuming filename format is "00001.json"
        try:
            file_id = int(os.path.splitext(basename)[0])
            if file_id > max_id:
                max_id = file_id
        except ValueError:
            pass  # Ignore files that don't match the numeric naming
            
    return max_id + 1

def generate_batch(category, num_samples, output_base="../../../assets/dataset/intermediate_json"):
    """批量生成指定数量和类别的数据"""
    category_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), output_base, category))
    os.makedirs(category_dir, exist_ok=True)
    
    start_id = get_next_id(category_dir)
    print(f"Preparing to generate {num_samples} samples for '{category}' category.")
    print(f"Starting at ID: {start_id:05d}")
    print(f"Output directory: {category_dir}")
    print("-" * 40)
    
    success_count = 0
    
    for i in range(num_samples):
        current_id = start_id + i
        file_name = f"{current_id:05d}.json"
        
        if category == "small":
            # For small: 1 to 10 blocks
            builder = SmallBricksBuilder(length=8, width=8, height=8)
            num_blocks = random.randint(1, 10)
            builder.generate_random_stable_bricks(num_blocks=num_blocks)
            json_path = os.path.join(category_dir, file_name)
            builder.save_coordinates_json(json_path)
            actual_blocks = len(builder.grid.nonzero()[0])
            
        elif category == "small_size4":
            # For small size4: 1 to 10 blocks in 4x4x4 grid
            builder = SmallBricksBuilder(length=4, width=4, height=4)
            num_blocks = random.randint(1, 10)
            builder.generate_random_stable_bricks(num_blocks=num_blocks)
            json_path = os.path.join(category_dir, file_name)
            builder.save_coordinates_json(json_path)
            actual_blocks = len(builder.grid.nonzero()[0])
            
        elif category == "large":
            # For large: we can use the convex generation or random generation with >10 blocks
            # Here we'll stick to the existing convex method as it naturally generates larger blocks
            # You can adapt this to use SmallBricksBuilder with random.randint(11, max_val) if you prefer non-convex too
            builder = FinalViewBuilder(length=8, width=8, height=8)
            builder.generate_stable_convex(num_cuts=random.randint(5, 12)) 
            json_path = os.path.join(category_dir, file_name)
            builder.save_coordinates_json(json_path)
            actual_blocks = len(builder.grid.nonzero()[0])
            
            # Since FinalViewBuilder might randomly generate <=10 blocks sometimes, 
            # we should technically enforce the >10 rule here if we want strict categories.
            while actual_blocks <= 10:
                builder.generate_stable_convex(num_cuts=random.randint(5, 12))
                actual_blocks = len(builder.grid.nonzero()[0])
            
            # Overwrite with the enforced block count
            builder.save_coordinates_json(json_path)
            
        else:
            print(f"Unknown category: {category}")
            return
            
        print(f"Generated [{current_id:05d}] - Blocks: {actual_blocks}")
        success_count += 1
        
    print("-" * 40)
    print(f"Successfully generated {success_count} '{category}' JSON files at {category_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch generate block structure JSON files.")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["small", "small_size4", "large"], 
        required=True,
        help="The category of data to generate ('small' = 1-10 blocks, 'small_size4' = 1-10 blocks in 4x4x4 grid, 'large' = >10 blocks)."
    )
    parser.add_argument(
        "--num", 
        type=int, 
        default=5, 
        help="Number of samples to generate (default: 5)."
    )
    
    args = parser.parse_args()
    
    generate_batch(category=args.type, num_samples=args.num)

if __name__ == "__main__":
    main()
