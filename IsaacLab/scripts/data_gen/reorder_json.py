import os
import json
import glob

def main():
    base_dir = "/Users/gc311/Desktop/26Spring/image_bricks/IsaacLab/scripts/data_gen"
    input_dir = os.path.join(base_dir, "convex_json_batch")
    output_dir = os.path.join(base_dir, "ground_truth")

    os.makedirs(output_dir, exist_ok=True)

    # Find all json files in the input directory and sort them alphanumerically
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    for idx, file_path in enumerate(json_files, start=1):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sort blocks by z, x, y ascending
        blocks = data.get("blocks", [])
        blocks.sort(key=lambda b: (b["z"], b["x"], b["y"]))
        
        # Update IDs to be sequential after sorting
        for i, b in enumerate(blocks, start=1):
            b["id"] = i
            
        data["blocks"] = blocks
        
        # Generate new filename like 0001.json
        new_filename = f"{idx:04d}.json"
        output_path = os.path.join(output_dir, new_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    print(f"✅ Processed {len(json_files)} files. Ground truth saved to {output_dir}/")

if __name__ == "__main__":
    main()
