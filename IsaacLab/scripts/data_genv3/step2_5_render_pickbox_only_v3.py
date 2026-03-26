#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser(
    description="Step2.5: generate full-size images with only the pick bounding box overlay from existing step metadata."
)
parser.add_argument("--output_root", type=str, default="", help="Default: assets/dataset_v3/smallsize_4")
parser.add_argument(
    "--pickbox_dirname",
    type=str,
    default="pickbox_overlay",
    help="Folder name created beside views/ for pickbox-only overlay images.",
)
parser.add_argument(
    "--sample_ids",
    nargs="*",
    default=None,
    help="Optional sample ids to process. Default: all samples under sft_train.",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=0,
    help="Optional cap on processed samples after filtering. 0 means no cap.",
)
parser.add_argument(
    "--max_images",
    type=int,
    default=0,
    help="Optional cap on newly saved images. 0 means no cap.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing pickbox-only images. Default: disabled.",
)
args_cli = parser.parse_args()


def _root() -> str:
    if args_cli.output_root:
        return os.path.abspath(os.path.expanduser(args_cli.output_root))
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/dataset_v3/smallsize_4")
    )


def _dash(draw: ImageDraw.ImageDraw, p0, p1, color, lw, dash_len, gap_len):
    x0, y0, x1, y1 = float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    if length <= 1.0e-6:
        return
    ux, uy = dx / length, dy / length
    t = 0.0
    while t < length:
        t2 = min(length, t + dash_len)
        draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * t2, y0 + uy * t2)], fill=color, width=lw)
        t += dash_len + gap_len


def _overlay_pickbox(rgb: np.ndarray, pickbox: dict | None) -> np.ndarray:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    if pickbox is None:
        return np.array(image)

    line_width = max(5, int(round(float(min(rgb.shape[0], rgb.shape[1])) * 0.0066)))
    dash_len = max(10, int(round(float(line_width) * 2.4)))
    gap_len = max(6, int(round(float(line_width) * 1.3)))
    color = (70, 140, 255)

    x0 = int(pickbox["x_min"])
    y0 = int(pickbox["y_min"])
    x1 = int(pickbox["x_max"])
    y1 = int(pickbox["y_max"])
    _dash(draw, (x0, y0), (x1, y0), color, line_width, dash_len, gap_len)
    _dash(draw, (x1, y0), (x1, y1), color, line_width, dash_len, gap_len)
    _dash(draw, (x1, y1), (x0, y1), color, line_width, dash_len, gap_len)
    _dash(draw, (x0, y1), (x0, y0), color, line_width, dash_len, gap_len)
    return np.array(image)


def _sample_json_paths(root: str, selected_ids: list[str] | None, max_samples: int) -> list[str]:
    sft_root = os.path.join(root, "sft_train")
    if not os.path.isdir(sft_root):
        raise FileNotFoundError(f"sft_train folder not found: {sft_root}")

    if selected_ids:
        sample_ids = [str(x) for x in selected_ids]
    else:
        sample_ids = sorted(name for name in os.listdir(sft_root) if os.path.isdir(os.path.join(sft_root, name)))

    out = []
    for sample_id in sample_ids:
        sample_json = os.path.join(sft_root, sample_id, f"{sample_id}_data.json")
        if os.path.isfile(sample_json):
            out.append(sample_json)
    if int(max_samples) > 0:
        out = out[: int(max_samples)]
    return out


def _pickbox_output_path(sample_dir: str, sample_id: str, step_idx: int, view_name: str, dirname: str) -> str:
    filename = f"{sample_id}_step{step_idx:05d}_{view_name}_pickbox_overlay.png"
    return os.path.join(sample_dir, dirname, filename)


def _step_view_image_map(step: dict) -> dict[str, str]:
    view_names = list(step.get("camera_views", []))
    image_paths = list(step.get("generated_images", []))
    return {view_name: image_paths[idx] for idx, view_name in enumerate(view_names) if idx < len(image_paths)}


def _iter_pending_images(sample_json: str, dirname: str, overwrite: bool):
    data = json.load(open(sample_json, "r", encoding="utf-8"))
    sample_id = str(data.get("sample_id") or os.path.basename(sample_json).replace("_data.json", ""))
    sample_dir = os.path.dirname(sample_json)
    for step in data.get("steps", []):
        bbox_meta = step.get("annotation_multiview_bounding_box", {})
        if not bool(bbox_meta.get("has_next_action", False)):
            continue
        step_idx = int(step["step_index"])
        view_to_image_rel = _step_view_image_map(step)
        for view_name in step.get("camera_views", []):
            view_meta = bbox_meta.get("views", {}).get(view_name, {})
            pickbox = view_meta.get("selected_bbox_xyxy")
            if pickbox is None:
                continue
            src_rel = view_to_image_rel.get(view_name)
            if not src_rel:
                continue
            src_path = os.path.join(sample_dir, src_rel)
            dst_path = _pickbox_output_path(sample_dir, sample_id, step_idx, view_name, dirname)
            if not overwrite and os.path.isfile(dst_path):
                continue
            if not os.path.isfile(src_path):
                continue
            yield {
                "sample_id": sample_id,
                "sample_dir": sample_dir,
                "step_idx": step_idx,
                "view_name": view_name,
                "src_path": src_path,
                "dst_path": dst_path,
                "pickbox": pickbox,
            }


def _render_one(entry: dict):
    os.makedirs(os.path.dirname(entry["dst_path"]), exist_ok=True)
    rgb = np.array(Image.open(entry["src_path"]).convert("RGB"))
    overlay = _overlay_pickbox(rgb, entry["pickbox"])
    Image.fromarray(overlay).save(entry["dst_path"])
    print(f"[SAVE]: {entry['dst_path']}")


def main():
    output_root = _root()
    sample_jsons = _sample_json_paths(output_root, args_cli.sample_ids, int(args_cli.max_samples))
    if not sample_jsons:
        raise RuntimeError("No sample json files found to process.")

    max_images = int(args_cli.max_images)
    saved = 0
    pending = 0

    print(f"[INFO]: output_root={output_root}")
    print(f"[INFO]: sample_count={len(sample_jsons)}")
    print(f"[INFO]: pickbox_dirname={args_cli.pickbox_dirname}")
    print(f"[INFO]: overwrite={bool(args_cli.overwrite)}")
    print(f"[INFO]: max_images={max_images}")

    for sample_json in sample_jsons:
        for entry in _iter_pending_images(sample_json, args_cli.pickbox_dirname, bool(args_cli.overwrite)):
            pending += 1
            if max_images > 0 and saved >= max_images:
                continue
            _render_one(entry)
            saved += 1

    print(f"[INFO]: pending_images_before_run={pending}")
    print(f"[INFO]: saved_now={saved}")
    print("[INFO]: Finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()
