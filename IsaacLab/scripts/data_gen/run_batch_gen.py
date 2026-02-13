# 文件名: run_batch.py
import os
import subprocess
import time
import glob

def run_batch():
    # 1. 设置路径
    json_folder = "convex_json_batch"
    script_name = "batch_gen.py"
    
    # 2. 获取所有 JSON 文件
    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    
    if not json_files:
        print(f"Error: No JSON files found in {json_folder}")
        return

    print(f"Found {len(json_files)} files to process.")

    # 3. 循环执行
    for i, json_path in enumerate(json_files):
        filename = os.path.basename(json_path)
        
        print(f"\n>>> [{i+1}/{len(json_files)}] Processing: {filename}")
        
        # 构建命令：python batch_gen.py --enable_cameras --json_file convex_01.json
        cmd = [
            "python", script_name,
            "--enable_cameras",
            "--json_file", filename  # 传递文件名给主脚本
        ]
        
        try:
            # 启动子进程，并等待它结束（窗口关闭）
            subprocess.run(cmd, check=True)
            # 暂停 2 秒让显存释放
            time.sleep(2)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
        
if __name__ == "__main__":
    run_batch()