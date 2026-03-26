#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = "assets/dataset_v3/smallsize_4"
DEFAULT_INPUT_JSONL_NAME = "sharegpt_pickbox_grounding_qwen.jsonl"
DEFAULT_OUTPUT_JSONL_NAME = "sharegpt_pickbox_grounding_qwen_text.jsonl"
DEFAULT_ERROR_JSONL_NAME = "sharegpt_pickbox_grounding_qwen_text_errors.jsonl"
DEFAULT_MISSING_JSONL_NAME = "sharegpt_pickbox_grounding_qwen_text_missing.jsonl"
OPENAI_MODEL = "gemini-2.5-pro"
OPENAI_API_BASE = "https://api.chataiapi.com/v1"
OPENAI_API_KEY = "sk-NjSIBWC6tbyo6bD6eieZkn4hHxlBxW4yumd0FNRLxPwDywVD"
THREAD_LOCAL = threading.local()

ASSISTANT_PATTERN = re.compile(
    r"^\s*(?:<thinking>.*?</thinking>)?\s*<text>(.*?)</text>\s*<answer>(.*?)</answer>\s*$",
    re.DOTALL,
)

OPENAI_SYSTEM_PROMPT = """You describe the next block to pick in a block-building task.

You will see:
- two target reference images,
- two current scene images with pickbox overlays,
- and a color hint for the selected ground-truth block.

Return strict JSON only:
{"text": "Pick the ... block next."}

Rules:
- Output exactly one short sentence in very simple English.
- The sentence must say which block should be picked next.
- Prefer precise color words when needed, such as light blue, dark blue, teal, olive green, lime green, reddish brown, or purple.
- If colors are close or ambiguous, add a short relative-position cue to disambiguate, for example:
  "Pick the light blue block in the upper-left area next."
  "Pick the darker green block near the center-right area next."
- If two nearby blocks still look similar, use stronger comparative language such as darker, lighter, more saturated, less saturated, more greenish, more bluish, more purple, more orange, more pinkish.
- When needed, use relative phrases such as relative left, relative right, slightly upper, slightly lower, more upper-left, more lower-right, near center-left, near center-right, or more deeper inside the structure.
- Use coarse and relative spatial phrases instead of exact coordinates.
- Do not mention exact coordinates, view names, box tokens, overlays, pixels, or reasoning.
- Do not mention uncertainty.
"""

SFT_SYSTEM_PROMPT = """You are a vision-based planning assistant for block-building tasks. Given target reference images and current scene images from multiple camera views, your goal is to decide which current object should be picked next so that the current structure moves closer to the target structure.

Requirements:
1. Predict a pickbox for the current object to grasp in each current camera view.
2. The predicted boxes across different views must correspond to the same physical pick object.
3. Bounding boxes must be expressed in image pixel coordinates, where (0,0) is the top-left corner.
4. Each bounding box must use Qwen grounding format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>.

Output format:
Return exactly <text>...</text><answer>...</answer>.
Inside <text>, give one short sentence saying which block should be picked next.
Inside <answer>, return one pickbox line per current-view in order.
Use this exact line format:
pickbox@view_name: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
Do not include any additional text outside these tags."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pickbox-only ShareGPT data with short <text> labels.")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--input_jsonl", type=str, default="")
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--error_jsonl", type=str, default="")
    parser.add_argument("--missing_jsonl", type=str, default="")
    parser.add_argument("--model", type=str, default=OPENAI_MODEL)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--sample_ids", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_passes", type=int, default=3)
    parser.add_argument("--retry_sleep_seconds", type=float, default=3.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_per_sample", action="store_true", default=True)
    parser.add_argument("--no_write_per_sample", dest="write_per_sample", action="store_false")
    parser.add_argument("--allow_fallback_text", action="store_true", default=True)
    parser.add_argument("--no_allow_fallback_text", dest="allow_fallback_text", action="store_false")
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _sample_filter(sample_ids_raw: str) -> Set[str]:
    if not sample_ids_raw.strip():
        return set()
    return {item.strip() for item in sample_ids_raw.split(",") if item.strip()}


def _read_existing_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = payload.get("id")
            if record_id:
                ids.add(str(record_id))
    return ids


@lru_cache(maxsize=256)
def _image_to_data_url(path_str: str) -> str:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_rgb(rgb: List[float]) -> tuple[float, float, float]:
    return tuple(round(float(v), 6) for v in rgb)


def _base_color_name(rgb: List[float]) -> str:
    r, g, b = [float(v) for v in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c < 0.18:
        return "dark"
    if s < 0.18:
        if l < 0.3:
            return "dark gray"
        if l > 0.7:
            return "light gray"
        return "gray"

    h_deg = (h * 360.0) % 360.0
    if h_deg < 12 or h_deg >= 345:
        return "red"
    if h_deg < 28:
        return "orange"
    if h_deg < 45:
        return "yellow"
    if h_deg < 75:
        return "lime green"
    if h_deg < 105:
        return "green"
    if h_deg < 150:
        return "teal"
    if h_deg < 195:
        return "cyan"
    if h_deg < 250:
        return "blue"
    if h_deg < 290:
        return "purple"
    if h_deg < 325:
        return "magenta"
    if l < 0.42 and s < 0.55:
        return "brown"
    return "pink"


def _describe_palette_color(rgb: List[float], palette_rgbs: List[List[float]]) -> str:
    base_name = _base_color_name(rgb)
    r, g, b = [float(v) for v in rgb]
    _h, l, s = colorsys.rgb_to_hls(r, g, b)

    same_family = [item for item in palette_rgbs if _base_color_name(item) == base_name]
    if len(same_family) <= 1:
        if base_name == "red" and g > 0.12:
            return "reddish red"
        if base_name == "orange" and r < 0.48:
            return "brownish orange"
        if base_name == "pink" and l < 0.45:
            return "dark pink"
        return base_name

    lightness_values = sorted(colorsys.rgb_to_hls(*[float(v) for v in item])[1] for item in same_family)
    saturation_values = sorted(colorsys.rgb_to_hls(*[float(v) for v in item])[2] for item in same_family)
    darkest = lightness_values[0]
    lightest = lightness_values[-1]
    least_saturated = saturation_values[0]
    most_saturated = saturation_values[-1]

    prefix = ""
    if len(same_family) == 2:
        if abs(l - darkest) > 0.04 or abs(lightest - darkest) > 0.08:
            prefix = "dark " if l <= (darkest + lightest) / 2 else "light "
        elif abs(s - least_saturated) > 0.08 or abs(most_saturated - least_saturated) > 0.12:
            prefix = "less saturated " if s <= (least_saturated + most_saturated) / 2 else "more saturated "
    else:
        if abs(l - darkest) < 0.03:
            prefix = "darkest "
        elif abs(l - lightest) < 0.03:
            prefix = "lightest "
        elif l < sum(lightness_values) / len(lightness_values) - 0.04:
            prefix = "dark "
        elif l > sum(lightness_values) / len(lightness_values) + 0.04:
            prefix = "light "
        elif s < sum(saturation_values) / len(saturation_values) - 0.08:
            prefix = "less saturated "
        elif s > sum(saturation_values) / len(saturation_values) + 0.08:
            prefix = "more saturated "

    if base_name == "red" and g > 0.12:
        return f"{prefix}reddish red".strip()
    if base_name == "orange" and r < 0.48:
        return f"{prefix}brownish orange".strip()
    return f"{prefix}{base_name}".strip()


def _sample_palette_color_names(sample_payload: dict, sample_dir: Path) -> Dict[tuple[float, float, float], str]:
    color_scheme_path = sample_dir / f"{sample_payload['sample_id']}_color_scheme.json"
    palette_rgbs: List[List[float]]
    if color_scheme_path.is_file():
        color_scheme = _load_json(color_scheme_path)
        palette_rgbs = color_scheme.get("selected_colors_rgb") or []
    else:
        palette_rgbs = [item["color_rgb"] for item in sample_payload.get("block_colors", [])]

    normalized_palette = [_normalize_rgb(rgb) for rgb in palette_rgbs]
    color_name_by_rgb: Dict[tuple[float, float, float], str] = {}
    for rgb in palette_rgbs:
        color_name_by_rgb[_normalize_rgb(rgb)] = _describe_palette_color(rgb, palette_rgbs)

    for item in sample_payload.get("block_colors", []):
        norm_rgb = _normalize_rgb(item["color_rgb"])
        if norm_rgb not in color_name_by_rgb:
            color_name_by_rgb[norm_rgb] = _describe_palette_color(list(norm_rgb), [list(rgb) for rgb in normalized_palette] or [list(norm_rgb)])

    return color_name_by_rgb


def _fallback_text_from_sample(sample_payload: dict, step_index: int) -> str:
    step = next(step for step in sample_payload["steps"] if int(step["step_index"]) == step_index)
    block_id = int(step["annotation_multiview_bounding_box"]["next_block_id_1based"])
    color_entry = next(item for item in sample_payload["block_colors"] if int(item["id"]) == block_id)
    sample_dir = Path(sample_payload.get("_sample_dir", ""))
    color_name_by_rgb = _sample_palette_color_names(sample_payload, sample_dir) if sample_dir else {}
    color_name = color_name_by_rgb.get(_normalize_rgb(color_entry["color_rgb"]), _base_color_name(color_entry["color_rgb"]))
    return f"Pick the {color_name} block next."


def _assistant_answer_only(content: str) -> str:
    match = ASSISTANT_PATTERN.match(content)
    if not match:
        raise ValueError(f"Unexpected assistant content format: {content[:200]}")
    return match.group(2).strip()


def _per_sample_output_path(output_dataset_root: Path, sample_id: str, step_index: int) -> Path:
    return output_dataset_root / sample_id / f"{sample_id}_step{step_index:05d}_sharegpt_pickbox_qwen_text.json"


def _pickbox_overlay_paths(record: dict, dataset_root: Path) -> List[Path]:
    sample_id = str(record["meta"]["sample_id"])
    step_index = int(record["meta"]["step_index"])
    camera_views = list(record["meta"]["camera_views"])
    return [
        dataset_root / "sft_train" / sample_id / "pickbox_overlay" / f"{sample_id}_step{step_index:05d}_{view_name}_pickbox_overlay.png"
        for view_name in camera_views
    ]


def _build_openai_user_content(record: dict, dataset_root: Path, fallback_text: str) -> List[dict]:
    content: List[dict] = [
        {
            "type": "text",
            "text": (
                "Generate the short <text> sentence for this pick-only training sample. "
                f"Ground-truth color hint: {fallback_text}"
            ),
        }
    ]
    user_message = record["messages"][1]["content"]
    if not isinstance(user_message, str):
        raise TypeError("Expected original user content to be a string with <image> placeholders.")

    target_image_paths = [dataset_root / rel_path for rel_path in record["images"][:2]]
    current_image_paths = _pickbox_overlay_paths(record, dataset_root)
    for image_path in current_image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing pickbox overlay image for OpenAI prompt: {image_path}")
    image_paths = [*target_image_paths, *current_image_paths]
    labels = [
        "Target view oblique_main",
        "Target view oblique_side",
        "Current view oblique_main with pickbox overlay",
        "Current view oblique_side with pickbox overlay",
    ]
    for label, image_path in zip(labels, image_paths):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path.as_posix())}})
    content.append(
        {
            "type": "text",
            "text": (
                "Return JSON only. Keep the sentence short. "
                "If multiple blocks have similar colors or nearby positions, use finer color wording and stronger relative-position wording such as relative left, more upper-left, or more deeper inside the structure to disambiguate."
            ),
        }
    )
    return content


def _set_openai_client_class(client_class) -> None:
    global OPENAI_CLIENT_CLASS
    OPENAI_CLIENT_CLASS = client_class


def _get_thread_openai_client():
    client = getattr(THREAD_LOCAL, "openai_client", None)
    if client is None:
        if OPENAI_CLIENT_CLASS is None:
            raise RuntimeError("OpenAI client class has not been initialized.")
        client = OPENAI_CLIENT_CLASS(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        THREAD_LOCAL.openai_client = client
    return client


def _chat_completion_text(message_content: object) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        chunks: List[str] = []
        for item in message_content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        merged = "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()
        if merged:
            return merged
    raise ValueError(f"Unsupported chat completion content type: {type(message_content).__name__}")


def _call_openai_text(model: str, max_output_tokens: int, user_content: List[dict]) -> str:
    client = _get_thread_openai_client()
    max_retries = 3
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_output_tokens,
                temperature=0.0,
                timeout=60,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError("OpenAI response has no choices.")
            message = getattr(choices[0], "message", None)
            if message is None and isinstance(choices[0], dict):
                message = choices[0].get("message")
            if message is None:
                raise ValueError("OpenAI response choice has no message.")
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            content = _chat_completion_text(content)
            text = _extract_text_from_response_content(content)
            if not text:
                raise ValueError("OpenAI returned empty text.")
            return text
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                break
            sleep_sec = min(30, 2**attempt)
            print(f"[WARN] API call failed inside _call_openai_text attempt={attempt + 1}/{max_retries}, retrying in {sleep_sec}s: {exc}")
            time.sleep(sleep_sec)

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


def _extract_text_from_response_content(content: str) -> str:
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("OpenAI returned empty content.")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and "text" in payload:
        text = str(payload["text"]).strip()
        if text:
            return text

    match = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, re.DOTALL)
    if match:
        text = json.loads(f'"{match.group(1)}"').strip()
        if text:
            return text

    if "\n" not in cleaned and len(cleaned) <= 200:
        return cleaned.strip('"').strip()

    raise ValueError(f"Unable to extract text from OpenAI content: {cleaned[:200]}")


def _convert_record(base_record: dict, text: str, model_name: str, source: str) -> dict:
    record = copy.deepcopy(base_record)
    record["messages"][0]["content"] = SFT_SYSTEM_PROMPT
    answer_text = _assistant_answer_only(record["messages"][2]["content"])
    record["messages"][2]["content"] = f"<text>{text}</text><answer>{answer_text}</answer>"
    meta = record.setdefault("meta", {})
    meta["text_model"] = model_name
    meta["text_source"] = source
    meta["text_generated_utc"] = datetime.now(timezone.utc).isoformat()
    meta["format_version"] = "qwen_box_tokens_pickbox_text_v1"
    return record


def _process_record(record: dict, args: argparse.Namespace, dataset_root: Path, output_sft_root: Path) -> dict:
    record_id = str(record["id"])
    sample_id = str(record["meta"]["sample_id"])
    step_index = int(record["meta"]["step_index"])
    sample_dir = dataset_root / "sft_train" / sample_id
    sample_payload = _load_json(sample_dir / f"{sample_id}_data.json")
    sample_payload["_sample_dir"] = sample_dir.as_posix()
    fallback_text = _fallback_text_from_sample(sample_payload, step_index)
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            text = _call_openai_text(
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                user_content=_build_openai_user_content(record, dataset_root, fallback_text),
            )
            converted = _convert_record(record, text, args.model, "openai_chat_completions_api")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            return {
                "status": "processed",
                "id": record_id,
                "sample_id": sample_id,
                "step_index": step_index,
                "text_record": converted,
                "per_sample_path": _per_sample_output_path(output_sft_root, sample_id, step_index),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] id={record_id} attempt={attempt}/{args.max_retries} error={last_error}")
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds * attempt)
    if args.allow_fallback_text:
        converted = _convert_record(record, fallback_text, args.model, f"fallback_after_api_error: {last_error}")
        return {
            "status": "processed",
            "id": record_id,
            "sample_id": sample_id,
            "step_index": step_index,
            "text_record": converted,
            "per_sample_path": _per_sample_output_path(output_sft_root, sample_id, step_index),
        }
    return {
        "status": "failed",
        "id": record_id,
        "sample_id": sample_id,
        "step_index": step_index,
        "error": last_error or "unknown error",
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_sft_root = dataset_root / "sft_train"
    input_jsonl = Path(args.input_jsonl).resolve() if args.input_jsonl else dataset_root / DEFAULT_INPUT_JSONL_NAME
    output_jsonl = (
        Path(args.output_jsonl).resolve() if args.output_jsonl else dataset_root / DEFAULT_OUTPUT_JSONL_NAME
    )
    error_jsonl = (
        Path(args.error_jsonl).resolve() if args.error_jsonl else dataset_root / DEFAULT_ERROR_JSONL_NAME
    )
    missing_jsonl = (
        Path(args.missing_jsonl).resolve() if args.missing_jsonl else dataset_root / DEFAULT_MISSING_JSONL_NAME
    )

    rows = _read_jsonl(input_jsonl)
    sample_ids = _sample_filter(args.sample_ids)
    if sample_ids:
        rows = [row for row in rows if str(row["meta"]["sample_id"]) in sample_ids]
    if args.max_records > 0:
        rows = rows[: args.max_records]

    existing_ids = set()
    if output_jsonl.exists() and not args.overwrite:
        existing_ids = _read_existing_ids(output_jsonl)

    if args.overwrite:
        output_jsonl.unlink(missing_ok=True)
        error_jsonl.unlink(missing_ok=True)
    missing_jsonl.unlink(missing_ok=True)

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] output_sft_root={output_sft_root}")
    print(f"[INFO] input_jsonl={input_jsonl}")
    print(f"[INFO] output_jsonl={output_jsonl}")
    print(f"[INFO] error_jsonl={error_jsonl}")
    print(f"[INFO] missing_jsonl={missing_jsonl}")
    print(f"[INFO] base_record_count={len(rows)}")
    print(f"[INFO] existing_output_ids={len(existing_ids)}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] api_base={OPENAI_API_BASE}")
    print(f"[INFO] num_workers={args.num_workers}")
    print(f"[INFO] max_passes={args.max_passes}")
    print(f"[INFO] allow_fallback_text={args.allow_fallback_text}")

    selected_rows: List[dict] = []
    skipped = 0
    for row in rows:
        record_id = str(row["id"])
        if record_id in existing_ids:
            skipped += 1
            continue
        if args.max_records > 0 and len(selected_rows) >= args.max_records:
            break
        selected_rows.append(row)

    print(f"[INFO] selected={len(selected_rows)}")

    processed = 0
    failed_events = 0

    def persist_result(result: dict) -> None:
        nonlocal processed, failed_events
        if result["status"] == "processed":
            if args.write_per_sample:
                _write_json(result["per_sample_path"], result["text_record"])
            _append_jsonl(output_jsonl, result["text_record"])
            processed += 1
            return
        failed_events += 1
        _append_jsonl(
            error_jsonl,
            {
                "id": result["id"],
                "sample_id": result["sample_id"],
                "step_index": result["step_index"],
                "error": result["error"],
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    remaining_rows = list(selected_rows)
    pass_index = 0
    while remaining_rows and pass_index < args.max_passes:
        pass_index += 1
        print(f"[INFO] recovery_pass={pass_index} remaining={len(remaining_rows)}")
        if args.num_workers == 1:
            for row in remaining_rows:
                persist_result(_process_record(row, args, dataset_root, output_sft_root))
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
                future_map = {
                    executor.submit(_process_record, row, args, dataset_root, output_sft_root): row
                    for row in remaining_rows
                }
                for future in as_completed(future_map):
                    row = future_map[future]
                    try:
                        persist_result(future.result())
                    except Exception as exc:
                        persist_result(
                            {
                                "status": "failed",
                                "id": row.get("id"),
                                "sample_id": row.get("meta", {}).get("sample_id"),
                                "step_index": row.get("meta", {}).get("step_index"),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
        completed_ids = _read_existing_ids(output_jsonl)
        remaining_rows = [row for row in remaining_rows if str(row["id"]) not in completed_ids]

    missing_rows = [
        {
            "id": row["id"],
            "sample_id": row["meta"]["sample_id"],
            "step_index": int(row["meta"]["step_index"]),
        }
        for row in remaining_rows
    ]
    _write_jsonl(missing_jsonl, missing_rows)

    print(f"[INFO] processed={processed}")
    print(f"[INFO] skipped={skipped}")
    print(f"[INFO] failed_events={failed_events}")
    print(f"[INFO] remaining={len(remaining_rows)}")
    print(f"[INFO] recovery_passes_used={pass_index}")

    if args.require_complete and remaining_rows:
        raise RuntimeError(
            f"Generation incomplete: {len(remaining_rows)} records are still missing. "
            f"See {missing_jsonl} and {error_jsonl}."
        )


if __name__ == "__main__":
    main()
