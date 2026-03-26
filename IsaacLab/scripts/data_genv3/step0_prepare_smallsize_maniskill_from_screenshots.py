#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


CAMERA_SPECS = [
    {"source_name": "base_camera", "canonical_name": "oblique_main"},
    {"source_name": "left_side_camera", "canonical_name": "oblique_side"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert assets/screenshots into a smallsize_4-compatible dataset layout."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="assets/screenshots",
        help="Directory containing ep*_camera.png, ep*_camera_boxed.png, and ep*_boxes_with_corners.json.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="assets/dataset_v3/smallsize_maniskill",
        help="Output dataset root to create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild output_root if it already exists.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _xyxy(corners: dict) -> dict:
    return {
        "x_min": int(corners["top_left"][0]),
        "y_min": int(corners["top_left"][1]),
        "x_max": int(corners["bottom_right"][0]),
        "y_max": int(corners["bottom_right"][1]),
        "width": int(corners["bottom_right"][0] - corners["top_left"][0] + 1),
        "height": int(corners["bottom_right"][1] - corners["top_left"][1] + 1),
    }


def _sample_id_for_episode(episode: int) -> str:
    return f"{episode:05d}"


def _episode_indices(source_dir: Path) -> List[int]:
    indices: List[int] = []
    for path in sorted(source_dir.glob("ep*_boxes_with_corners.json")):
        name = path.stem
        if not name.startswith("ep") or not name.endswith("_boxes_with_corners"):
            continue
        episode_str = name[len("ep") : -len("_boxes_with_corners")]
        indices.append(int(episode_str))
    return indices


def _camera_payload(camera_data: dict, sample_id: str, canonical_name: str) -> tuple[dict, str, str]:
    current_image_rel = f"views/{sample_id}_step00000_{canonical_name}.png"
    overlay_rel = f"bbox_overlay/{sample_id}_step00000_{canonical_name}_gt_bbox_overlay.png"
    target_image_rel = f"struct/{sample_id}/{sample_id}_{canonical_name}.png"
    view_meta = {
        "image_size_hw": [512, 512],
        "bbox_size_px": 30,
        "target_bbox_xyxy": _xyxy(camera_data["goal_box_corners"]),
        "selected_bbox_xyxy": _xyxy(camera_data["init_box_corners"]),
        "bbox_files": {
            "bbox_overlay_image": overlay_rel,
        },
    }
    return view_meta, current_image_rel, target_image_rel


def _build_sample_payload(sample_id: str, episode: int, source_name: str, payload: dict) -> dict:
    views_meta: Dict[str, dict] = {}
    generated_images: List[str] = []
    target_images_by_view: Dict[str, str] = {}

    for camera_spec in CAMERA_SPECS:
        source_name = camera_spec["source_name"]
        canonical_name = camera_spec["canonical_name"]
        camera_data = payload["cameras"][source_name]
        view_meta, current_image_rel, target_image_rel = _camera_payload(camera_data, sample_id, canonical_name)
        views_meta[canonical_name] = view_meta
        generated_images.append(current_image_rel)
        target_images_by_view[canonical_name] = target_image_rel

    return {
        "source_file": source_name,
        "sample_id": sample_id,
        "struct_id": sample_id,
        "order_index": episode,
        "target_blocks_ordered": [{"id": 1}],
        "target_final_absolute_world_positions": [
            {
                "id": 1,
                "world_x": float(payload["goal_world_pos"][0]),
                "world_y": float(payload["goal_world_pos"][1]),
                "world_z": float(payload["goal_world_pos"][2]),
            }
        ],
        "block_colors": [],
        "structure": {
            "structure_folder": f"struct/{sample_id}",
            "target_images_by_view": target_images_by_view,
        },
        "steps": [
            {
                "step_index": 0,
                "total_steps": 1,
                "placed_count": 0,
                "scattered_count": 1,
                "placed_blocks": [],
                "scattered_blocks": [
                    {
                        "world_x": float(payload["cubeA_world_pos"][0]),
                        "world_y": float(payload["cubeA_world_pos"][1]),
                        "world_z": float(payload["cubeA_world_pos"][2]),
                    }
                ],
                "block_states": [
                    {
                        "id": 1,
                        "target_world_xyz": [float(v) for v in payload["goal_world_pos"]],
                        "current_world_xyz": [float(v) for v in payload["cubeA_world_pos"]],
                        "current_source": "scatter",
                        "is_placed": False,
                    }
                ],
                "generated_images": generated_images,
                "camera_views": [camera_spec["canonical_name"] for camera_spec in CAMERA_SPECS],
                "annotation_multiview_bounding_box": {
                    "enabled": True,
                    "type": "multi_view_bounding_box",
                    "bbox_source": "screenshots_boxes_with_corners",
                    "bbox_min_size_px": int(payload.get("box_size", 30)),
                    "bbox_base_size_px": int(payload.get("box_size", 30)),
                    "bbox_auto_scaled_with_resolution": False,
                    "save_bbox_images": True,
                    "has_next_action": True,
                    "next_block_index_0based": 0,
                    "next_block_id_1based": 1,
                    "next_target_world_xyz": [float(v) for v in payload["goal_world_pos"]],
                    "selected_block_world_xyz": [float(v) for v in payload["cubeA_world_pos"]],
                    "views": views_meta,
                },
            }
        ],
    }


def _build_structure_payload(sample_id: str, episode: int, payload: dict) -> dict:
    return {
        "rules": "maniskill_single_object_pick_place",
        "dimensions": {
            "length": 1,
            "width": 1,
            "height": 1,
        },
        "seed": episode,
        "total_blocks": 1,
        "blocks_canonical": [{"id": 0, "x": 0, "y": 0, "z": 0}],
        "struct_id": sample_id,
        "num_orders": 1,
        "goal_world_pos": [float(v) for v in payload["goal_world_pos"]],
        "camera_views": [camera_spec["canonical_name"] for camera_spec in CAMERA_SPECS],
        "target_image_mode": "boxed_reference_images",
    }


def _build_orders_payload(sample_id: str, episode: int, payload: dict) -> dict:
    return {
        "struct_id": sample_id,
        "orders": [
            {
                "order_index": episode,
                "sequence": [{"id": 1}],
                "goal_world_pos": [float(v) for v in payload["goal_world_pos"]],
            }
        ],
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    episode_indices = _episode_indices(source_dir)
    if not episode_indices:
        raise ValueError(f"No ep*_boxes_with_corners.json found in {source_dir}")

    print(f"[INFO] source_dir={source_dir}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] episode_count={len(episode_indices)}")

    for episode in episode_indices:
        sample_id = _sample_id_for_episode(episode)
        source_json = source_dir / f"ep{episode}_boxes_with_corners.json"
        payload = _load_json(source_json)

        sample_dir = output_root / "sft_train" / sample_id
        views_dir = sample_dir / "views"
        overlay_dir = sample_dir / "bbox_overlay"
        struct_dir = output_root / "struct" / sample_id

        for camera_spec in CAMERA_SPECS:
            source_name = camera_spec["source_name"]
            canonical_name = camera_spec["canonical_name"]
            plain_src = source_dir / f"ep{episode}_{source_name}.png"
            boxed_src = source_dir / f"ep{episode}_{source_name}_boxed.png"
            _copy(plain_src, views_dir / f"{sample_id}_step00000_{canonical_name}.png")
            _copy(boxed_src, overlay_dir / f"{sample_id}_step00000_{canonical_name}_gt_bbox_overlay.png")
            _copy(boxed_src, struct_dir / f"{sample_id}_{canonical_name}.png")

        sample_payload = _build_sample_payload(sample_id, episode, source_json.name, payload)
        structure_payload = _build_structure_payload(sample_id, episode, payload)
        orders_payload = _build_orders_payload(sample_id, episode, payload)

        _write_json(sample_dir / f"{sample_id}_data.json", sample_payload)
        _write_json(struct_dir / f"{sample_id}_structure.json", structure_payload)
        _write_json(struct_dir / f"{sample_id}_all_orders.json", orders_payload)

    print("[INFO] conversion_complete=true")


if __name__ == "__main__":
    main()
