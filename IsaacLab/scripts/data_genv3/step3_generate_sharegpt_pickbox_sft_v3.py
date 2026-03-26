#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


SYSTEM_PROMPT = (
    "You are a vision-based planning assistant for block-building tasks. Given target reference images and current "
    "scene images from multiple camera views, your goal is to decide which current object should be picked next so "
    "that the current structure moves closer to the target structure.\n\n"
    "Requirements:\n"
    "1. Predict a pickbox for the current object to grasp in each current camera view.\n"
    "2. The predicted boxes across different views must correspond to the same physical pick object.\n"
    "3. Bounding boxes must be expressed in image pixel coordinates, where (0,0) is the top-left corner.\n"
    "4. Each bounding box must use Qwen grounding format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>.\n\n"
    "Output format:\n"
    "Return exactly <thinking>...</thinking><text>...</text><answer>...</answer>.\n"
    "Inside <answer>, return one pickbox line per current-view in order.\n"
    "Use this exact line format:\n"
    "pickbox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
    "If chain-of-thought or extra text is unavailable, leave <thinking></thinking> and/or <text></text> empty.\n"
    "Do not include any additional text outside these tags."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dataset_v3 rendered samples into step-wise ShareGPT-style pickbox-only SFT data."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="assets/dataset_v3/smallsize_4",
        help="Dataset root containing sft_train and struct folders.",
    )
    parser.add_argument(
        "--pickbox_dirname",
        type=str,
        default="pickbox_overlay",
        help="Folder beside views/ that contains the current-view pickbox-only images.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Unified JSONL output path. Default: <dataset_root>/sharegpt_pickbox_grounding_qwen.jsonl",
    )
    parser.add_argument(
        "--write_per_sample",
        action="store_true",
        help="Also write one ShareGPT JSON file per actionable step beside each sample.",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default="",
        help="Optional comma-separated sample IDs to process, e.g. 00001,00003",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _pixel_box(box: dict | None) -> List[int]:
    if not box:
        raise ValueError("Missing pickbox for actionable step.")
    return [
        int(box["x_min"]),
        int(box["y_min"]),
        int(box["x_max"]),
        int(box["y_max"]),
    ]


def _relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sample_filter(sample_ids_raw: str) -> Set[str]:
    if not sample_ids_raw.strip():
        return set()
    return {item.strip() for item in sample_ids_raw.split(",") if item.strip()}


def _qwen_box_token(box: List[int]) -> str:
    x1, y1, x2, y2 = box
    return f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"


def _user_content_for_step(view_names: List[str], target_view_names: List[str]) -> str:
    lines: List[str] = ["Target reference images:"]
    for view_name in target_view_names:
        lines.append(f"Target view {view_name}: <image>")
    lines.append("Current scene images:")
    for view_name in view_names:
        lines.append(f"Current view {view_name}: <image>")
    lines.append(
        "Determine which current object should be picked next in each current view so that the structure progresses "
        "toward the target. Return all pickbox lines inside <answer>, using the exact format "
        "`pickbox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>`."
    )
    return "\n".join(lines)


def _assistant_text_for_step(step: dict) -> tuple[str, Dict[str, List[int]]]:
    bbox_meta = step["annotation_multiview_bounding_box"]
    pickbox: Dict[str, List[int]] = {}
    lines: List[str] = []
    for view_name in step["camera_views"]:
        view_meta = bbox_meta["views"][view_name]
        pickbox[view_name] = _pixel_box(view_meta["selected_bbox_xyxy"])
        lines.append(f"pickbox@{view_name}: {_qwen_box_token(pickbox[view_name])}")
    assistant = f"<thinking></thinking><text></text><answer>{chr(10).join(lines)}</answer>"
    return assistant, pickbox


def _discover_sample_dirs(sft_root: Path) -> List[Path]:
    return sorted([p for p in sft_root.iterdir() if p.is_dir()])


def _pickbox_image_path(sample_dir: Path, step_index: int, view_name: str, dirname: str) -> Path:
    sample_id = sample_dir.name
    filename = f"{sample_id}_step{step_index:05d}_{view_name}_pickbox_overlay.png"
    return sample_dir / dirname / filename


def _view_image_path(sample_dir: Path, step: dict, view_name: str) -> Path:
    view_to_image = {
        vn: sample_dir / rel_path for vn, rel_path in zip(step.get("camera_views", []), step.get("generated_images", []))
    }
    if view_name not in view_to_image:
        raise FileNotFoundError(f"Missing raw view image path for {sample_dir.name} view={view_name} step={step.get('step_index')}")
    return view_to_image[view_name]


def _build_records(sample_dir: Path, dataset_root: Path, pickbox_dirname: str) -> List[dict]:
    sample_id = sample_dir.name
    sample_json = sample_dir / f"{sample_id}_data.json"
    payload = _load_json(sample_json)

    steps = payload["steps"]
    actionable_steps = [step for step in steps if step["annotation_multiview_bounding_box"]["has_next_action"]]
    if not actionable_steps:
        raise ValueError(f"No actionable steps found in {sample_json}")

    target_images_by_view: Dict[str, str] = payload["structure"]["target_images_by_view"]
    target_view_names = list(target_images_by_view.keys())
    target_images = [
        _relative_to(dataset_root / target_images_by_view[view_name], dataset_root) for view_name in target_view_names
    ]

    records: List[dict] = []
    for step in actionable_steps:
        current_view_names = list(step["camera_views"])
        step_index = int(step["step_index"])
        current_image_paths = [_view_image_path(sample_dir, step, view_name) for view_name in current_view_names]
        for current_image_path in current_image_paths:
            if not current_image_path.is_file():
                raise FileNotFoundError(f"Missing current image for sample {sample_id} step {step_index}: {current_image_path}")
        current_images = [_relative_to(path, dataset_root) for path in current_image_paths]
        assistant_text, pickbox = _assistant_text_for_step(step)
        records.append(
            {
                "id": f"{sample_id}_step{step_index:05d}",
                "images": [*target_images, *current_images],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_content_for_step(current_view_names, target_view_names)},
                    {"role": "assistant", "content": assistant_text},
                ],
                "meta": {
                    "sample_id": payload["sample_id"],
                    "struct_id": payload["struct_id"],
                    "order_index": payload["order_index"],
                    "step_index": step_index,
                    "bbox_kind": "pickbox_only",
                    "answer_format": "qwen_box_tokens_pickbox",
                    "answer_line_format": "pickbox@{view_name}: <|box_start|>(x1,y1),(x2,y2)<|box_end|>",
                    "message_content_format": "string_with_image_placeholders",
                    "image_placeholder": "<image>",
                    "camera_views": current_view_names,
                    "target_views": target_view_names,
                    "box_targets_xyxy": {
                        "pickbox": pickbox,
                    },
                    "format_version": "qwen_box_tokens_pickbox_v3",
                },
            }
        )
    return records


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    sft_root = dataset_root / "sft_train"
    if not sft_root.is_dir():
        raise FileNotFoundError(f"sft_train not found under dataset_root: {sft_root}")

    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else dataset_root / "sharegpt_pickbox_grounding_qwen.jsonl"
    )

    sample_dirs = _discover_sample_dirs(sft_root)
    sample_filter = _sample_filter(args.sample_ids)
    if sample_filter:
        sample_dirs = [sample_dir for sample_dir in sample_dirs if sample_dir.name in sample_filter]

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] sample_count={len(sample_dirs)}")
    print(f"[INFO] pickbox_dirname={args.pickbox_dirname}")
    print(f"[INFO] output_jsonl={output_jsonl}")

    total = 0
    with output_jsonl.open("w", encoding="utf-8") as jsonl_f:
        for sample_dir in sample_dirs:
            records = _build_records(sample_dir, dataset_root, args.pickbox_dirname)
            for record in records:
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                if args.write_per_sample:
                    step_index = int(record["meta"]["step_index"])
                    per_sample_path = sample_dir / f"{sample_dir.name}_step{step_index:05d}_sharegpt_pickbox_qwen.json"
                    _write_json(per_sample_path, record)
    print(f"[INFO] wrote_records={total}")


if __name__ == "__main__":
    main()
CURRENT_IMAGE_MODE = "views"
