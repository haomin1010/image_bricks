#!/usr/bin/env python3
"""
STEP 4: Generate partial-view SFT data (LlamaFactory/ShareGPT style) in batch.

This script scans dataset folders such as:
    assets/dataset_v2/small_size4/00001

For each sample folder, it generates:
    <shape_id>_sft_woCoT_partialview.json

The output format follows the currently used partial-view training data:
1) Every build step queries at least one camera view before placing one block.
2) Submit step (stepN+1) queries at least one camera view before submitting.

All target images are expected to be in the sample folder.
Each query result returns one image placeholder.

Additionally, this script also writes one unified JSONL file for batch training:
    <dataset_root>/sft_woCoT_partialview_all.jsonl
Each JSONL line is one sample with image paths relative to dataset_root.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CAMERA_VIEWS = ("top", "front", "side", "iso", "iso2")
CAMERA_GROUP_A_SEQ = (1, 2, 0)  # one from {0,1,2}
CAMERA_GROUP_B_SEQ = (4, 3)  # one from {3,4}
MAX_ATTEMPTS_FACTOR = 1.5

SYSTEM_PROMPT_TEXT = (
    "You are a robot arm controller. Your goal is to build a target block structure on a 8x8 tabletop grid.\n\n"
    "At reset you first see all 5 target camera views (IDs 0..4).\n"
    "Grid coordinates: x, y in {0..7}, z is the vertical layer (0 = bottom, 1 = one above, etc.).\n\n"
    "Each turn output exactly one action in this format:\n"
    "<thinking></thinking><action>...</action>\n\n"
    "Use the thinking section to briefly reason about the target views, the current partial structure, and the next best action before acting.\n"
    "Think step by step and keep the thinking concise and directly relevant to the next action.\n\n"
    "Valid action content inside <action> is exactly ONE of:\n\n"
    "1) Query one camera:\n"
    "{\"query\": [INT]}\n\n"
    "2) Place a cube:\n"
    "{\"x\": INT, \"y\": INT, \"z\": INT}\n\n"
    "3) When the structure is complete:\n"
    "submit\n\n"
    "Examples:\n"
    "  Query camera: <thinking></thinking><action>{\"query\": [2]}</action>\n"
    "  Place a brick: <thinking></thinking><action>{\"x\": 2, \"y\": 3, \"z\": 0}</action>\n"
    "  Submit: <thinking></thinking><action>submit</action>"
)

@dataclass(frozen=True)
class Block:
    x: int
    y: int
    z: int


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_blocks(payload: dict) -> List[Block]:
    blocks_raw = payload["original_data"]["blocks"]
    return [Block(x=int(item["x"]), y=int(item["y"]), z=int(item["z"])) for item in blocks_raw]


def _extract_dims(payload: dict) -> Tuple[int, int, int]:
    dims = payload["original_data"]["dimensions"]
    return (int(dims["length"]), int(dims["width"]), int(dims["height"]))


def _max_attempts(total_blocks: int, factor: float) -> int:
    return max(1, int(math.ceil(max(1, total_blocks) * factor)))


def _target_description(total_blocks: int, dims: Tuple[int, int, int], max_attempts: int) -> str:
    return (
        "Build the target block structure shown in the reference images. "
        "Use the target views to infer the correct shape and continue building from the current state."
    )


def _initial_target_user_text(desc: str) -> str:
    return (
        f"{desc}\n"
        "Target multi-view images:\n"
        "Target camera 0: <image>\n"
        "Target camera 1: <image>\n"
        "Target camera 2: <image>\n"
        "Target camera 3: <image>\n"
        "Target camera 4: <image>\n"
        "From the current state, you must query at least one camera before each placement or submit action.\n"
        "Query additional views if needed, then choose the next action that best advances the build toward the target."
    )


def _query_result_user_text(camera_id: int) -> str:
    return (
        f"Query result for camera {camera_id}.\n<image>\n"
        "You have queried a camera for this turn. Query another camera if needed, or place a cube / submit when ready."
    )


def _placement_applied_user_text() -> str:
    return (
        "Placement executed. The current state has been updated.\n"
        "You must query at least one camera before your next placement or submit action.\n"
        "You may query one camera, place a cube, or submit."
    )


def _submit_applied_user_text() -> str:
    return (
        "Submission executed.\n"
        "The current state has been checked against the target."
    )


def _step_data_path(sample_dir: Path, shape_id: str, step_idx: int) -> Path:
    return sample_dir / f"{shape_id}_step{step_idx:05d}_data.json"


def _target_image_name(shape_id: str, view: str) -> str:
    return f"{shape_id}_{view}.png"


def _step_image_name(shape_id: str, step_idx: int, view: str) -> str:
    return f"{shape_id}_step{step_idx:05d}_{view}.png"


def _camera_image_name_for_step(shape_id: str, step_idx: int, camera_id: int) -> str:
    return _step_image_name(shape_id, step_idx, CAMERA_VIEWS[camera_id])


def _query_pair_for_turn(turn_idx: int) -> Tuple[int, int]:
    cam_a = CAMERA_GROUP_A_SEQ[turn_idx % len(CAMERA_GROUP_A_SEQ)]
    cam_b = CAMERA_GROUP_B_SEQ[turn_idx % len(CAMERA_GROUP_B_SEQ)]
    return cam_a, cam_b


def _infer_last_placed_block(prev_blocks: Sequence[Block], curr_blocks: Sequence[Block]) -> Block:
    prev_set = {(b.x, b.y, b.z) for b in prev_blocks}
    new_blocks = [b for b in curr_blocks if (b.x, b.y, b.z) not in prev_set]
    if len(new_blocks) != 1:
        raise ValueError(
            f"Expected exactly one new block between steps, got {len(new_blocks)} "
            f"(prev={len(prev_blocks)}, curr={len(curr_blocks)})"
        )
    return new_blocks[0]


def _format_place_action(block: Block) -> str:
    action = json.dumps({"x": int(block.x), "y": int(block.y), "z": int(block.z)}, ensure_ascii=True)
    return _wrap_action(action)


def _format_query_action(camera_id: int) -> str:
    action = json.dumps({"query": [int(camera_id)]}, ensure_ascii=True)
    return _wrap_action(action)


def _format_submit_action() -> str:
    return _wrap_action("submit")


def _wrap_action(action_text: str) -> str:
    return f"<thinking></thinking><action>{action_text}</action>"


def _build_system_message() -> Dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT_TEXT}


def _discover_shape_dirs(dataset_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir():
            continue
        dirs.append(path)
    return dirs


def _discover_step_indices(sample_dir: Path, shape_id: str) -> List[int]:
    pattern = re.compile(rf"^{re.escape(shape_id)}_step(\d{{5}})_data\.json$")
    indices: List[int] = []
    for p in sample_dir.iterdir():
        if not p.is_file():
            continue
        match = pattern.match(p.name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        return []
    indices = sorted(set(indices))
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"Step indices are not contiguous for {shape_id}: found {indices}, expected {expected}"
        )
    return indices


def _verify_required_files(sample_dir: Path, shape_id: str, total_steps: int) -> None:
    missing: List[Path] = []
    for view in CAMERA_VIEWS:
        target_img = sample_dir / _target_image_name(shape_id, view)
        if not target_img.exists():
            missing.append(target_img)

    for step_idx in range(1, total_steps + 1):
        step_data = _step_data_path(sample_dir, shape_id, step_idx)
        if not step_data.exists():
            missing.append(step_data)
        for view in CAMERA_VIEWS:
            step_img = sample_dir / _step_image_name(shape_id, step_idx, view)
            if not step_img.exists():
                missing.append(step_img)

    if missing:
        preview = ", ".join(str(p.name) for p in missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} required files under {sample_dir}. First few: {preview}"
        )


def _build_samples_for_shape(sample_dir: Path) -> List[dict]:
    shape_id = sample_dir.name
    base_json_path = sample_dir / f"{shape_id}_data.json"
    if not base_json_path.exists():
        raise FileNotFoundError(f"Base json not found: {base_json_path}")

    base_payload = _load_json(base_json_path)
    dims = _extract_dims(base_payload)
    step_indices = _discover_step_indices(sample_dir, shape_id)
    if not step_indices:
        raise ValueError(f"No contiguous step data found in: {sample_dir}")
    total_steps = len(step_indices)

    _verify_required_files(sample_dir, shape_id, total_steps)

    step_blocks: List[List[Block]] = []
    for step_idx in step_indices:
        payload = _load_json(_step_data_path(sample_dir, shape_id, step_idx))
        blocks = _extract_blocks(payload)
        if not blocks:
            raise ValueError(f"Step data has no blocks: {_step_data_path(sample_dir, shape_id, step_idx)}")
        step_blocks.append(blocks)

    placements: List[Block] = []
    prev: List[Block] = []
    for curr in step_blocks:
        placed = _infer_last_placed_block(prev, curr)
        placements.append(placed)
        prev = list(curr)

    target_images = [_target_image_name(shape_id, view) for view in CAMERA_VIEWS]
    max_attempts = _max_attempts(total_steps, MAX_ATTEMPTS_FACTOR)
    desc = _target_description(total_steps, dims, max_attempts)

    samples: List[dict] = []
    system_msg = _build_system_message()

    # Build steps: 1..N
    for step_idx in range(1, total_steps + 1):
        sample_id = f"{shape_id}_build_step{step_idx:05d}"
        place_block = placements[step_idx - 1]
        messages: List[Dict[str, str]] = [system_msg]
        images = list(target_images)
        messages.append({"role": "user", "content": _initial_target_user_text(desc)})

        if step_idx == 1:
            cam_id = CAMERA_GROUP_A_SEQ[0]
            query_img = _target_image_name(shape_id, CAMERA_VIEWS[cam_id])
            images.append(query_img)

            messages.append({"role": "assistant", "content": _format_query_action(cam_id)})
            messages.append({"role": "user", "content": _query_result_user_text(cam_id)})
            messages.append({"role": "assistant", "content": _format_place_action(place_block)})
            messages.append({"role": "user", "content": _placement_applied_user_text()})
        else:
            turn_idx = step_idx - 2
            cam_1, cam_2 = _query_pair_for_turn(turn_idx)
            query_img_1 = _camera_image_name_for_step(shape_id, step_idx - 1, cam_1)
            query_img_2 = _camera_image_name_for_step(shape_id, step_idx - 1, cam_2)
            images.extend([query_img_1, query_img_2])

            messages.append({"role": "assistant", "content": _format_query_action(cam_1)})
            messages.append({"role": "user", "content": _query_result_user_text(cam_1)})
            messages.append({"role": "assistant", "content": _format_query_action(cam_2)})
            messages.append({"role": "user", "content": _query_result_user_text(cam_2)})
            messages.append({"role": "assistant", "content": _format_place_action(place_block)})
            messages.append({"role": "user", "content": _placement_applied_user_text()})

        samples.append({"id": sample_id, "images": images, "messages": messages})

    # Submit step: N+1
    submit_step_id = total_steps + 1
    sample_id = f"{shape_id}_build_step{submit_step_id:05d}_submit"
    cam_1, cam_2 = _query_pair_for_turn(total_steps - 1)
    submit_images = list(target_images)
    submit_images.extend(
        [
            _camera_image_name_for_step(shape_id, total_steps, cam_1),
            _camera_image_name_for_step(shape_id, total_steps, cam_2),
        ]
    )
    submit_messages: List[Dict[str, str]] = [
        system_msg,
        {"role": "user", "content": _initial_target_user_text(desc)},
        {"role": "assistant", "content": _format_query_action(cam_1)},
        {"role": "user", "content": _query_result_user_text(cam_1)},
        {"role": "assistant", "content": _format_query_action(cam_2)},
        {"role": "user", "content": _query_result_user_text(cam_2)},
        {"role": "assistant", "content": _format_submit_action()},
        {"role": "user", "content": _submit_applied_user_text()},
    ]
    samples.append({"id": sample_id, "images": submit_images, "messages": submit_messages})

    return samples


def _write_samples(samples: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _to_jsonl_record(sample: dict, shape_id: str) -> dict:
    return {
        "id": sample["id"],
        "images": [f"{shape_id}/{img_name}" for img_name in sample["images"]],
        "messages": sample["messages"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate partial-view SFT data. Only requires dataset root path."
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Path to dataset root containing shape folders, e.g. /.../assets/dataset_v2/small_size4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    shape_dirs = _discover_shape_dirs(dataset_root)

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] discovered shape folders: {len(shape_dirs)}")

    jsonl_path = dataset_root / "sft_woCoT_partialview_all.jsonl"
    print(f"[INFO] unified jsonl={jsonl_path}")

    generated = 0
    total_records = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for shape_dir in shape_dirs:
            shape_id = shape_dir.name
            output_path = shape_dir / f"{shape_id}_sft_woCoT_partialview.json"
            samples = _build_samples_for_shape(shape_dir)
            _write_samples(samples, output_path)

            for sample in samples:
                record = _to_jsonl_record(sample, shape_id)
                jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                total_records += 1

            print(f"[OK] {shape_id}: wrote {len(samples)} samples -> {output_path.name}")
            generated += 1

    print("-" * 72)
    print(f"[SUMMARY] generated={generated} total={len(shape_dirs)} records={total_records}")


if __name__ == "__main__":
    main()
