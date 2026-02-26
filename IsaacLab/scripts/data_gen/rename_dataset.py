import os
import glob
import shutil

def main():
    base_dir = "/Users/gc311/Desktop/26Spring/image_bricks/IsaacLab/scripts/data_gen"
    json_batch_dir = os.path.join(base_dir, "convex_json_batch")
    snapshots_dir = os.path.join(base_dir, "output_snapshots")

    # 1. Rename files in convex_json_batch
    print("Renaming files in convex_json_batch...")
    json_files = sorted(glob.glob(os.path.join(json_batch_dir, "*.json")))
    
    for idx, file_path in enumerate(json_files, start=1):
        dirname = os.path.dirname(file_path)
        old_filename = os.path.basename(file_path)
        new_filename = f"{idx:04d}.json"
        
        # Avoid renaming if it's already in the correct format (e.g., if script is run twice)
        if old_filename == new_filename:
            continue
            
        new_file_path = os.path.join(dirname, new_filename)
        os.rename(file_path, new_file_path)
        print(f"  {old_filename} -> {new_filename}")

    # 2. Rename directories and files in output_snapshots
    print("\nRenaming directories and files in output_snapshots...")
    # Gather directories (filter out non-directories if any)
    all_items = sorted(os.listdir(snapshots_dir))
    dir_items = [d for d in all_items if os.path.isdir(os.path.join(snapshots_dir, d))]
    
    for idx, old_dirname in enumerate(dir_items, start=1):
        old_dirpath = os.path.join(snapshots_dir, old_dirname)
        new_dirname = f"{idx:04d}"
        new_dirpath = os.path.join(snapshots_dir, new_dirname)
        
        # Rename files inside the directory first
        for item in os.listdir(old_dirpath):
            item_path = os.path.join(old_dirpath, item)
            if os.path.isfile(item_path):
                # Replace the old directory prefix (e.g. convex_01) with the new one (e.g. 0001)
                if item.startswith(old_dirname):
                    new_item_name = item.replace(old_dirname, new_dirname, 1)
                    new_item_path = os.path.join(old_dirpath, new_item_name)
                    os.rename(item_path, new_item_path)
                    print(f"    {item} -> {new_item_name}")

        # Rename the directory itself
        if old_dirname != new_dirname:
            os.rename(old_dirpath, new_dirpath)
            print(f"  {old_dirname}/ -> {new_dirname}/")

    print("\n✅ Renaming complete.")

if __name__ == "__main__":
    main()
