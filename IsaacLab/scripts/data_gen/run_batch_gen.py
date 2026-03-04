import os
import subprocess
import time
import glob
import argparse

def run_batch(category):
    # 1. 设置路径
    json_folder = os.path.join("ground_truth", category)
    script_name = "batch_gen.py"
    
    # 2. 获取所有 JSON 文件
    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    
    if not json_files:
        print(f"Error: No JSON files found in {json_folder}")
        return

    print(f"Found {len(json_files)} files to process in {category}.")

    # 3. 循环执行
    for i, json_path in enumerate(json_files):
        filename = os.path.basename(json_path)
        
        print(f"\n>>> [{i+1}/{len(json_files)}] Processing: {filename}")
        
        # 构建命令：python batch_gen.py --enable_cameras --type small --json_file 00001.json
        cmd = [
            "python", script_name,
            "--enable_cameras",
            "--type", category,
            "--json_file", filename
        ]
        
        try:
            # 启动子进程，并等待它结束（窗口关闭）
            subprocess.run(cmd, check=True)
            # 暂停 2 秒让显存释放
            time.sleep(2)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch generation script.")
    parser.add_argument("--type", type=str, choices=["small", "large"], required=True, 
                        help="Data category to process ('small' or 'large').")
    args_cli = parser.parse_args()
    
    run_batch(args_cli.type)