#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


SFT_SYSTEM_PROMPT = (
    "You are a vision-based planning assistant for block-building tasks. Given target reference images and current "
    "scene images from multiple camera views, your goal is to decide which current object should be picked and where "
    "it should be placed so that the current structure moves closer to the target structure.\n\n"
    "Requirements:\n"
    "1. Predict a pickbox for the current object to grasp in each current camera view.\n"
    "2. Predict a placebox for the target placement location in each current camera view.\n"
    "3. The predicted boxes across different views must correspond to the same physical pick object and the same "
    "physical placement location.\n"
    "4. Bounding boxes must be expressed in image pixel coordinates, where (0,0) is the top-left corner.\n"
    "5. Each bounding box must use Qwen grounding format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>.\n\n"
    "Output format:\n"
    "Return exactly <thinking>...</thinking><text>...</text><answer>...</answer>.\n"
    "Inside <answer>, return one box per line. First output all pickbox lines in current-view order, then output all "
    "placebox lines in the same current-view order.\n"
    "Use this exact line format:\n"
    "pickbox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
    "placebox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
    "If chain-of-thought or extra text is unavailable, leave <thinking></thinking> and/or <text></text> empty.\n"
    "Do not include any additional text outside these tags."
)

CO_T_SYSTEM_PROMPT = (
    "You are a vision-based planning assistant for block-building tasks. Given target reference images and current "
    "scene images from multiple camera views, your goal is to decide which current object should be picked and where "
    "it should be placed so that the current structure moves closer to the target structure.\n\n"
    "Requirements:\n"
    "1. Predict a pickbox for the current object to grasp in each current camera view.\n"
    "2. Predict a placebox for the target placement location in each current camera view.\n"
    "3. The predicted boxes across different views must correspond to the same physical pick object and the same "
    "physical placement location.\n"
    "4. Bounding boxes must be expressed in image pixel coordinates, where (0,0) is the top-left corner.\n"
    "5. Each bounding box must use Qwen grounding format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>.\n"
    "6. Before giving the final answer, provide a short explanation inside <thinking>...</thinking> that:\n"
    "   - compares the target images with the current images,\n"
    "   - identifies which object should be picked and which target position is still missing,\n"
    "   - explicitly states which object the pickbox corresponds to,\n"
    "   - explicitly states which placement location the placebox corresponds to,\n"
    "   - explains why that visual location is the correct next placement,\n"
    "   - and states why this position can already be placed, especially whether the lower support is satisfied.\n"
    "7. The explanation must be based only on visible image evidence. Do not mention explicit pixel coordinates, grid "
    "coordinates, or simulator metadata inside <thinking>.\n"
    "8. The explanation should avoid phrases like \"the marked region\" or \"the marked bounding box\". Prefer a "
    "first-person decision sentence such as \"Therefore, I decide to output the pickbox of the ... block and the "
    "placebox of its target position.\"\n"
    "9. The explanation should contain exactly 3 sentences.\n"
    "10. Sentence 1 should state what the target has that the current scene is still missing.\n"
    "11. Sentence 2 should state which object the pickbox corresponds to and which location the placebox corresponds to.\n"
    "12. Sentence 3 should state why the placement is feasible now and conclude in first person that the model will "
    "output the pickbox and placebox.\n"
    "13. The explanation should keep the wording compact and avoid extra modifiers, repeated comparisons, or side remarks.\n"
    "14. The explanation should target roughly 60 to 110 tokens.\n\n"
    "Output format:\n"
    "Return exactly <thinking>...</thinking><text>...</text><answer>...</answer>.\n"
    "Inside <answer>, return one box per line. First output all pickbox lines in current-view order, then output all "
    "placebox lines in the same current-view order.\n"
    "Use this exact line format:\n"
    "pickbox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
    "placebox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
    "The <text></text> span may be empty.\n"
    "Do not include any additional text outside these tags."
)

THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
TEXT_PATTERN = re.compile(r"<text>(.*?)</text>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair existing step-wise bbox ShareGPT data to use Qwen box tokens and <image> placeholders."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="assets/dataset_v3/smallsize_4",
        help="Dataset root containing sft_train and sharegpt_bbox_grounding_qwen*.jsonl files.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pixel_box(box: dict | None) -> List[int]:
    if not box:
        raise ValueError("Missing bbox for actionable step.")
    return [
        int(box["x_min"]),
        int(box["y_min"]),
        int(box["x_max"]),
        int(box["y_max"]),
    ]


def _qwen_box_token(box: List[int]) -> str:
    x1, y1, x2, y2 = box
    return f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"


def _assistant_answer_lines(pickbox: Dict[str, List[int]], placebox: Dict[str, List[int]], view_names: List[str]) -> str:
    lines: List[str] = []
    for view_name in view_names:
        lines.append(f"pickbox@{view_name}: {_qwen_box_token(pickbox[view_name])}")
    for view_name in view_names:
        lines.append(f"placebox@{view_name}: {_qwen_box_token(placebox[view_name])}")
    return "\n".join(lines)


def _initial_user_text(target_view_names: List[str], current_view_names: List[str]) -> str:
    lines: List[str] = ["Target reference images:"]
    for view_name in target_view_names:
        lines.append(f"Target view {view_name}: <image>")
    lines.append("Current scene images:")
    for view_name in current_view_names:
        lines.append(f"Current view {view_name}: <image>")
    lines.append(
        "Determine which current object should be picked next and where it should be placed in each current view so "
        "that the structure progresses toward the target. Return all pickbox lines first and then all placebox lines "
        "inside <answer>, using the exact format `pickbox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>` and "
        "`placebox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>`."
    )
    return "\n".join(lines)


@lru_cache(maxsize=512)
def _load_sample_payload(dataset_root_str: str, sample_id: str) -> dict:
    dataset_root = Path(dataset_root_str)
    sample_json = dataset_root / "sft_train" / sample_id / f"{sample_id}_data.json"
    return _load_json(sample_json)


def _find_step(payload: dict, step_index: int) -> dict:
    for step in payload["steps"]:
        if int(step["step_index"]) == step_index:
            return step
    raise KeyError(f"step_index={step_index} not found in sample {payload.get('sample_id')}")


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(1) if match else ""


def _target_views(payload: dict) -> List[str]:
    return list(payload["structure"]["target_images_by_view"].keys())


def _step_boxes(step: dict) -> tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    bbox_meta = step["annotation_multiview_bounding_box"]
    pickbox: Dict[str, List[int]] = {}
    placebox: Dict[str, List[int]] = {}
    for view_name in step["camera_views"]:
        view_meta = bbox_meta["views"][view_name]
        pickbox[view_name] = _pixel_box(view_meta["selected_bbox_xyxy"])
        placebox[view_name] = _pixel_box(view_meta["target_bbox_xyxy"])
    return pickbox, placebox


def _is_cot_record(record: dict) -> bool:
    meta = record.get("meta", {})
    if "cot_style" in meta or "cot_model" in meta:
        return True
    assistant = record["messages"][2]["content"]
    return "<thinking>" in assistant


def _repair_step_record(record: dict, dataset_root: Path) -> dict:
    meta = record.get("meta", {})
    sample_id = str(meta["sample_id"])
    step_index = int(meta["step_index"])
    payload = _load_sample_payload(dataset_root.as_posix(), sample_id)
    step = _find_step(payload, step_index)

    current_view_names = list(step["camera_views"])
    target_view_names = _target_views(payload)
    pickbox, placebox = _step_boxes(step)
    assistant_old = str(record["messages"][2]["content"])
    thinking = _extract_tag(assistant_old, THINKING_PATTERN)
    text = _extract_tag(assistant_old, TEXT_PATTERN)

    record["messages"][0]["content"] = CO_T_SYSTEM_PROMPT if _is_cot_record(record) else SFT_SYSTEM_PROMPT
    record["messages"][1]["content"] = _initial_user_text(target_view_names, current_view_names)
    record["messages"][2]["content"] = (
        f"<thinking>{thinking}</thinking>"
        f"<text>{text}</text>"
        f"<answer>{_assistant_answer_lines(pickbox, placebox, current_view_names)}</answer>"
    )

    meta["bbox_kind"] = "both"
    meta["answer_format"] = "qwen_box_tokens_pickbox_placebox"
    meta["answer_line_format"] = "{box_kind}@{view_name}: <|box_start|>(x1,y1),(x2,y2)<|box_end|>"
    meta["message_content_format"] = "string_with_image_placeholders"
    meta["image_placeholder"] = "<image>"
    meta["camera_views"] = current_view_names
    meta["target_views"] = target_view_names
    meta["box_targets_xyxy"] = {
        "pickbox": pickbox,
        "placebox": placebox,
    }
    meta["format_version"] = "qwen_box_tokens_v3_repaired"
    return record


def _repair_json_file(path: Path, dataset_root: Path) -> bool:
    record = _load_json(path)
    meta = record.get("meta", {})
    if "step_index" not in meta or "sample_id" not in meta:
        return False
    _write_json(path, _repair_step_record(record, dataset_root))
    return True


def _repair_jsonl_file(path: Path, dataset_root: Path) -> int:
    rows = _read_jsonl(path)
    repaired_rows: List[dict] = []
    repaired_count = 0
    for row in rows:
        meta = row.get("meta", {})
        if "step_index" in meta and "sample_id" in meta:
            row = _repair_step_record(row, dataset_root)
            repaired_count += 1
        repaired_rows.append(row)
    _write_jsonl(path, repaired_rows)
    return repaired_count


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    step_json_paths = sorted((dataset_root / "sft_train").glob("*/*_step*_sharegpt_bbox_qwen.json"))
    cot_json_paths = sorted((dataset_root / "sft_train").glob("*/*_step*_sharegpt_bbox_qwen_cot.json"))
    jsonl_paths = [
        dataset_root / "sharegpt_bbox_grounding_qwen.jsonl",
        dataset_root / "sharegpt_bbox_grounding_qwen_cot_openai.jsonl",
    ]

    repaired_step_json = 0
    repaired_cot_json = 0
    repaired_jsonl_rows = 0

    for path in step_json_paths:
        if _repair_json_file(path, dataset_root):
            repaired_step_json += 1
    for path in cot_json_paths:
        if _repair_json_file(path, dataset_root):
            repaired_cot_json += 1
    for path in jsonl_paths:
        if path.exists():
            repaired_jsonl_rows += _repair_jsonl_file(path, dataset_root)

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] repaired_step_json={repaired_step_json}")
    print(f"[INFO] repaired_cot_json={repaired_cot_json}")
    print(f"[INFO] repaired_jsonl_rows={repaired_jsonl_rows}")


if __name__ == "__main__":
    main()
