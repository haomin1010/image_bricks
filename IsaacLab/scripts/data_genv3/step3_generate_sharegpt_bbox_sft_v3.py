#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a visual grounding assistant for block-building data. "
    "Given the target reference images and the current scene images, predict the bounding box of the next placement "
    "target in each current camera view. "
    "Use the response template "
    "<thinking>...</thinking><text>...</text><answer>...</answer>. "
    "Return boxes inside <answer>, using Qwen grounding format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>. "
    "If chain-of-thought or extra text is unavailable, leave <thinking></thinking> and/or <text></text> empty."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dataset_v3 rendered samples into ShareGPT-style bbox grounding SFT data."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="assets/dataset_v3/smallsize_4",
        help="Dataset root containing sft_train and struct folders.",
    )
    parser.add_argument(
        "--bbox_kind",
        type=str,
        default="target",
        choices=["target", "selected"],
        help="Which bbox to export from the render annotations.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Unified JSONL output path. Default: <dataset_root>/sharegpt_bbox_grounding_qwen.jsonl",
    )
    parser.add_argument(
        "--write_per_sample",
        action="store_true",
        help="Also write one ShareGPT JSON file beside each sample.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_box(box: dict | None) -> str:
    if not box:
        raise ValueError("Missing bbox for actionable step.")
    return (
        f"<|box_start|>({int(box['x_min'])},{int(box['y_min'])}),"
        f"({int(box['x_max'])},{int(box['y_max'])})<|box_end|>"
    )


def _relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _user_text_initial(view_names: List[str], target_view_names: List[str]) -> str:
    lines = [
        "Target reference images:",
    ]
    for view_name in target_view_names:
        lines.append(f"Target view {view_name}: <image>")
    lines.append("Current scene images:")
    for view_name in view_names:
        lines.append(f"Current view {view_name}: <image>")
    lines.append(
        "Predict the next placement target bounding box in every current view. "
        "Return one box per line in the format `view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>`."
    )
    return "\n".join(lines)


def _user_text_followup(view_names: List[str]) -> str:
    lines = [
        "The placement has been executed. Here are the updated current scene images for the next step:",
    ]
    for view_name in view_names:
        lines.append(f"Current view {view_name}: <image>")
    lines.append(
        "Predict the next placement target bounding box in every current view. "
        "Return one box per line in the format `view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>`."
    )
    return "\n".join(lines)


def _assistant_text_for_step(step: dict, bbox_kind: str) -> str:
    bbox_meta = step["annotation_multiview_bounding_box"]
    lines: List[str] = []
    suffix = "target_bbox_xyxy" if bbox_kind == "target" else "selected_bbox_xyxy"
    for view_name in step["camera_views"]:
        view_meta = bbox_meta["views"][view_name]
        lines.append(f"{view_name}: {_format_box(view_meta[suffix])}")
    answer = "\n".join(lines)
    return f"<thinking></thinking><text></text><answer>{answer}</answer>"


def _discover_sample_dirs(sft_root: Path) -> List[Path]:
    return sorted([p for p in sft_root.iterdir() if p.is_dir()])


def _build_record(sample_dir: Path, dataset_root: Path, bbox_kind: str) -> dict:
    sample_id = sample_dir.name
    sample_json = sample_dir / f"{sample_id}_data.json"
    payload = _load_json(sample_json)

    steps = payload["steps"]
    actionable_steps = [step for step in steps if step["annotation_multiview_bounding_box"]["has_next_action"]]
    if not actionable_steps:
        raise ValueError(f"No actionable steps found in {sample_json}")

    target_images_by_view: Dict[str, str] = payload["structure"]["target_images_by_view"]
    target_view_names = list(target_images_by_view.keys())
    current_view_names = list(actionable_steps[0]["camera_views"])

    images: List[str] = []
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for view_name in target_view_names:
        images.append(_relative_to(dataset_root / target_images_by_view[view_name], dataset_root))

    first_step = actionable_steps[0]
    for rel_path in first_step["generated_images"]:
        images.append(_relative_to(sample_dir / rel_path, dataset_root))
    messages.append({"role": "user", "content": _user_text_initial(current_view_names, target_view_names)})

    for step_idx, step in enumerate(actionable_steps):
        messages.append({"role": "assistant", "content": _assistant_text_for_step(step, bbox_kind)})
        if step_idx + 1 >= len(actionable_steps):
            break
        next_step = actionable_steps[step_idx + 1]
        for rel_path in next_step["generated_images"]:
            images.append(_relative_to(sample_dir / rel_path, dataset_root))
        messages.append({"role": "user", "content": _user_text_followup(list(next_step["camera_views"]))})

    return {
        "id": sample_id,
        "images": images,
        "messages": messages,
        "meta": {
            "sample_id": payload["sample_id"],
            "struct_id": payload["struct_id"],
            "order_index": payload["order_index"],
            "bbox_kind": bbox_kind,
            "camera_views": current_view_names,
        },
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    sft_root = dataset_root / "sft_train"
    if not sft_root.is_dir():
        raise FileNotFoundError(f"sft_train not found under dataset_root: {sft_root}")

    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else dataset_root / "sharegpt_bbox_grounding_qwen.jsonl"
    )

    sample_dirs = _discover_sample_dirs(sft_root)
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] sample_count={len(sample_dirs)}")
    print(f"[INFO] bbox_kind={args.bbox_kind}")
    print(f"[INFO] output_jsonl={output_jsonl}")

    total = 0
    with output_jsonl.open("w", encoding="utf-8") as jsonl_f:
        for sample_dir in sample_dirs:
            record = _build_record(sample_dir, dataset_root, args.bbox_kind)
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            if args.write_per_sample:
                per_sample_path = sample_dir / f"{sample_dir.name}_sharegpt_bbox_qwen.json"
                _write_json(per_sample_path, record)
    print(f"[INFO] wrote_records={total}")


if __name__ == "__main__":
    main()
