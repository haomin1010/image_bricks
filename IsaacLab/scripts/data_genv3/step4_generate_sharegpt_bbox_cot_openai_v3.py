#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set


SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_INPUT_JSONL_NAME = "sharegpt_bbox_grounding_qwen.jsonl"
DEFAULT_OUTPUT_JSONL_NAME = "sharegpt_bbox_grounding_qwen_cot_openai.jsonl"
DEFAULT_ERROR_JSONL_NAME = "sharegpt_bbox_grounding_qwen_cot_openai_errors.jsonl"
DEFAULT_MISSING_JSONL_NAME = "sharegpt_bbox_grounding_qwen_cot_openai_missing.jsonl"
DEFAULT_SFT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "sharegpt_bbox_cot_sft_system_prompt.txt"
DEFAULT_OPENAI_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "sharegpt_bbox_cot_openai_system_prompt.txt"
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
THREAD_LOCAL = threading.local()
OPENAI_CLIENT_CLASS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT-augmented ShareGPT bbox grounding data by calling OpenAI multimodal models."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="assets/dataset_v3/smallsize_4",
        help="Dataset root containing sharegpt_bbox_grounding_qwen.jsonl, sft_train and struct folders.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="",
        help="Base step-wise bbox SFT JSONL. Default: <dataset_root>/sharegpt_bbox_grounding_qwen.jsonl",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="CoT JSONL output path. Default: <dataset_root>/sharegpt_bbox_grounding_qwen_cot_openai.jsonl",
    )
    parser.add_argument(
        "--error_jsonl",
        type=str,
        default="",
        help="Error log JSONL path. Default: <dataset_root>/sharegpt_bbox_grounding_qwen_cot_openai_errors.jsonl",
    )
    parser.add_argument(
        "--missing_jsonl",
        type=str,
        default="",
        help="Missing-record JSONL path written after all recovery passes.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="OpenAI multimodal model to call. Default: gpt-5.4",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for supported models. GPT-5.4 supports none, low, medium, high, and xhigh.",
    )
    parser.add_argument(
        "--image_detail",
        type=str,
        default="auto",
        choices=["auto", "low", "high"],
        help="Image detail level sent to the API.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=2048,
        help="Maximum output tokens for the explanation generation call.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Optional cap on processed records for smoke tests. 0 means no cap.",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default="",
        help="Optional comma-separated sample IDs to process, e.g. 00001,00003",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Sleep between successful API requests per worker.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel OpenAI requests. 1 means serial execution.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries per record on API failure.",
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=3,
        help="Maximum full recovery passes over records still missing after each pass.",
    )
    parser.add_argument(
        "--retry_sleep_seconds",
        type=float,
        default=5.0,
        help="Base sleep time between retries.",
    )
    parser.add_argument(
        "--write_per_sample",
        action="store_true",
        help="Also write one CoT ShareGPT JSON beside each step sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSONL instead of resuming and skipping existing IDs.",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Exit with an error if any selected records are still missing after all recovery passes.",
    )
    parser.add_argument(
        "--include_overlay_images",
        action="store_true",
        default=True,
        help="Include pre-rendered bbox overlay images in the OpenAI call.",
    )
    parser.add_argument(
        "--no_overlay_images",
        dest="include_overlay_images",
        action="store_false",
        help="Do not include bbox overlay images in the OpenAI call.",
    )
    parser.add_argument(
        "--sft_system_prompt_path",
        type=str,
        default=str(DEFAULT_SFT_SYSTEM_PROMPT_PATH),
        help="Path to the final SFT system prompt text file.",
    )
    parser.add_argument(
        "--openai_system_prompt_path",
        type=str,
        default=str(DEFAULT_OPENAI_SYSTEM_PROMPT_PATH),
        help="Path to the OpenAI generation system prompt text file.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_answer_json_text(assistant_content: str) -> str:
    match = ANSWER_PATTERN.search(assistant_content)
    if not match:
        raise ValueError(f"Assistant content missing <answer>: {assistant_content[:200]}")
    answer_text = match.group(1).strip()
    json.loads(answer_text)
    return answer_text


def _sample_filter(sample_ids_raw: str) -> Set[str]:
    if not sample_ids_raw.strip():
        return set()
    return {item.strip() for item in sample_ids_raw.split(",") if item.strip()}


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
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=256)
def _load_sample_payload(path_str: str) -> dict:
    return _load_json(Path(path_str))


def _find_step(payload: dict, step_index: int) -> dict:
    for step in payload["steps"]:
        if int(step["step_index"]) == step_index:
            return step
    raise KeyError(f"step_index={step_index} not found in sample {payload.get('sample_id')}")


def _per_sample_output_path(dataset_root: Path, sample_id: str, step_index: int) -> Path:
    return dataset_root / "sft_train" / sample_id / f"{sample_id}_step{step_index:05d}_sharegpt_bbox_qwen_cot.json"


def _overlay_images_by_view(step: dict, sample_dir: Path) -> Dict[str, str]:
    bbox_views = step["annotation_multiview_bounding_box"]["views"]
    rel_paths: Dict[str, str] = {}
    for view_name in step["camera_views"]:
        rel_path = bbox_views[view_name]["bbox_files"].get("bbox_overlay_image")
        if rel_path:
            rel_paths[view_name] = (sample_dir / rel_path).resolve().as_posix()
    return rel_paths


def _build_actual_user_content(
    record: dict,
    dataset_root: Path,
    image_detail: str,
    overlay_paths: Dict[str, str],
) -> List[dict]:
    user_message = record["messages"][1]["content"]
    if not isinstance(user_message, list):
        raise TypeError("Expected user content to be multimodal content list.")

    content: List[dict] = []
    content.append({"type": "input_text", "text": "Now generate the explanation for the following training example."})

    image_iter = iter(record["images"])
    for item in user_message:
        item_type = item["type"]
        if item_type == "text":
            content.append({"type": "input_text", "text": item["text"]})
        elif item_type == "image":
            rel_path = next(image_iter)
            abs_path = (dataset_root / rel_path).resolve()
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(abs_path.as_posix()),
                    "detail": image_detail,
                }
            )
        else:
            raise ValueError(f"Unsupported user content type: {item_type}")

    if overlay_paths:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "Additional reference: these overlay images show the annotated action regions in the current views. "
                    "When both source-object and destination annotations are visible, explicitly explain which object "
                    "the pickbox corresponds to and which target position the placebox corresponds to."
                ),
            }
        )
        for view_name, abs_path in overlay_paths.items():
            content.append({"type": "input_text", "text": f"Overlay view {view_name}:"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(abs_path),
                    "detail": image_detail,
                }
            )

    content.append(
        {
            "type": "input_text",
            "text": (
                "Explain briefly, using only visible evidence from the images, why this is the correct next pick-and-"
                "place action. Explicitly distinguish the pickbox object from the placebox location. Do not say "
                "'marked region' or 'marked bounding box'. End with a first-person decision such as deciding to output "
                "the pickbox of the missing object and the placebox of its target position. Do not mention any "
                "explicit coordinates, pixel values, grid indices, or hidden simulator facts. Return strict JSON only."
            ),
        }
    )
    return content


def _supports_reasoning_effort(model: str) -> bool:
    prefixes = ("gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o4")
    return model.startswith(prefixes)


def _parse_thinking_payload(text: str) -> str:
    stripped = CODE_FENCE_PATTERN.sub("", text.strip())
    payload = json.loads(stripped)
    thinking = str(payload["thinking"]).strip()
    if not thinking:
        raise ValueError("Model returned empty thinking.")
    return thinking


def _thinking_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "thinking": {"type": "string"},
        },
        "required": ["thinking"],
        "additionalProperties": False,
    }


def _set_openai_client_class(client_class) -> None:
    global OPENAI_CLIENT_CLASS
    OPENAI_CLIENT_CLASS = client_class


def _get_thread_openai_client():
    client = getattr(THREAD_LOCAL, "openai_client", None)
    if client is None:
        if OPENAI_CLIENT_CLASS is None:
            raise RuntimeError("OpenAI client class has not been initialized.")
        client = OPENAI_CLIENT_CLASS()
        THREAD_LOCAL.openai_client = client
    return client


def _build_cot_record(
    base_record: dict,
    sft_system_prompt: str,
    thinking: str,
    answer_json_text: str,
    model_name: str,
    overlay_paths: Dict[str, str],
) -> dict:
    record = copy.deepcopy(base_record)
    record["messages"][0]["content"] = sft_system_prompt
    record["messages"][2]["content"] = f"<thinking>{thinking}</thinking><answer>{answer_json_text}</answer>"
    record["meta"]["cot_model"] = model_name
    record["meta"]["cot_source"] = "openai_responses_api"
    record["meta"]["cot_style"] = "visible_rationale"
    record["meta"]["cot_generated_utc"] = datetime.now(timezone.utc).isoformat()
    if overlay_paths:
        record["meta"]["bbox_overlay_images"] = {
            view_name: Path(path_str).relative_to(Path.cwd().resolve()).as_posix()
            if Path(path_str).is_relative_to(Path.cwd().resolve())
            else path_str
            for view_name, path_str in overlay_paths.items()
        }
    return record


def _call_openai_reasoning(
    client,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    openai_system_prompt: str,
    user_content: List[dict],
) -> tuple[str, object]:
    request = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": openai_system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cot_reasoning",
                "strict": True,
                "schema": _thinking_json_schema(),
            }
        },
    }
    if _supports_reasoning_effort(model):
        request["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**request)
    output_text = getattr(response, "output_text", None)
    if not output_text:
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        raise ValueError(
            f"OpenAI response returned empty output_text. status={status} incomplete_details={incomplete_details}"
        )
    return output_text, response


def _is_max_output_tokens_incomplete(exc: Exception) -> bool:
    text = str(exc)
    return "incomplete_details" in text and "max_output_tokens" in text


def _process_record(
    record: dict,
    args: argparse.Namespace,
    dataset_root: Path,
    sft_system_prompt: str,
    openai_system_prompt: str,
) -> dict:
    record_id = record["id"]
    sample_id = record["meta"]["sample_id"]
    step_index = int(record["meta"]["step_index"])
    sample_dir = dataset_root / "sft_train" / sample_id
    sample_json = sample_dir / f"{sample_id}_data.json"
    payload = _load_sample_payload(sample_json.as_posix())
    step = _find_step(payload, step_index)
    overlay_paths = _overlay_images_by_view(step, sample_dir) if args.include_overlay_images else {}
    answer_json_text = _extract_answer_json_text(record["messages"][2]["content"])
    user_content = _build_actual_user_content(
        record=record,
        dataset_root=dataset_root,
        image_detail=args.image_detail,
        overlay_paths=overlay_paths,
    )

    last_error: str | None = None
    last_raw_text: str | None = None
    current_max_output_tokens = args.max_output_tokens

    for attempt in range(1, args.max_retries + 1):
        try:
            raw_text, _response = _call_openai_reasoning(
                client=_get_thread_openai_client(),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=current_max_output_tokens,
                openai_system_prompt=openai_system_prompt,
                user_content=user_content,
            )
            last_raw_text = raw_text
            thinking = _parse_thinking_payload(raw_text)
            cot_record = _build_cot_record(
                base_record=record,
                sft_system_prompt=sft_system_prompt,
                thinking=thinking,
                answer_json_text=answer_json_text,
                model_name=args.model,
                overlay_paths=overlay_paths,
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            return {
                "status": "processed",
                "id": record_id,
                "sample_id": sample_id,
                "step_index": step_index,
                "cot_record": cot_record,
                "per_sample_path": _per_sample_output_path(dataset_root, sample_id, step_index),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] id={record_id} attempt={attempt}/{args.max_retries} error={last_error}")
            if _is_max_output_tokens_incomplete(exc) and attempt < args.max_retries:
                current_max_output_tokens = min(current_max_output_tokens * 2, 8192)
                print(
                    f"[INFO] id={record_id} increasing_max_output_tokens={current_max_output_tokens} for retry"
                )
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds * attempt)

    return {
        "status": "failed",
        "id": record_id,
        "sample_id": sample_id,
        "step_index": step_index,
        "error": last_error,
        "raw_text_preview": (last_raw_text[:1000] if last_raw_text else ""),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    input_jsonl = (
        Path(args.input_jsonl).expanduser().resolve()
        if args.input_jsonl
        else dataset_root / DEFAULT_INPUT_JSONL_NAME
    )
    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else dataset_root / DEFAULT_OUTPUT_JSONL_NAME
    )
    error_jsonl = (
        Path(args.error_jsonl).expanduser().resolve()
        if args.error_jsonl
        else dataset_root / DEFAULT_ERROR_JSONL_NAME
    )
    missing_jsonl = (
        Path(args.missing_jsonl).expanduser().resolve()
        if args.missing_jsonl
        else dataset_root / DEFAULT_MISSING_JSONL_NAME
    )
    sft_system_prompt_path = Path(args.sft_system_prompt_path).expanduser().resolve()
    openai_system_prompt_path = Path(args.openai_system_prompt_path).expanduser().resolve()

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The `openai` package is required. Install it with `pip install openai`.") from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.max_passes < 1:
        raise ValueError("--max_passes must be >= 1")

    _set_openai_client_class(OpenAI)
    sft_system_prompt = _load_text(sft_system_prompt_path)
    openai_system_prompt = _load_text(openai_system_prompt_path)
    base_records = _read_jsonl(input_jsonl)

    sample_filter = _sample_filter(args.sample_ids)
    if sample_filter:
        base_records = [record for record in base_records if record["meta"]["sample_id"] in sample_filter]

    existing_ids = set()
    if output_jsonl.exists() and not args.overwrite:
        existing_ids = _read_existing_ids(output_jsonl)

    if args.overwrite:
        output_jsonl.unlink(missing_ok=True)
        error_jsonl.unlink(missing_ok=True)
    missing_jsonl.unlink(missing_ok=True)

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] input_jsonl={input_jsonl}")
    print(f"[INFO] output_jsonl={output_jsonl}")
    print(f"[INFO] error_jsonl={error_jsonl}")
    print(f"[INFO] missing_jsonl={missing_jsonl}")
    print(f"[INFO] base_record_count={len(base_records)}")
    print(f"[INFO] existing_output_ids={len(existing_ids)}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] image_detail={args.image_detail}")
    print(f"[INFO] include_overlay_images={args.include_overlay_images}")
    print(f"[INFO] num_workers={args.num_workers}")
    print(f"[INFO] max_passes={args.max_passes}")

    processed = 0
    skipped = 0
    failed_events = 0
    selected_records: List[dict] = []

    for record in base_records:
        record_id = record["id"]
        if record_id in existing_ids:
            skipped += 1
            continue
        if args.max_records > 0 and len(selected_records) >= args.max_records:
            break
        selected_records.append(record)

    selected = len(selected_records)
    print(f"[INFO] selected={selected}")

    def persist_result(result: dict) -> None:
        nonlocal processed, failed_events
        if result["status"] == "processed":
            if args.write_per_sample:
                _write_json(result["per_sample_path"], result["cot_record"])
            _append_jsonl(output_jsonl, result["cot_record"])
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
                "raw_text_preview": result["raw_text_preview"],
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    remaining_records = list(selected_records)
    pass_index = 0
    while remaining_records and pass_index < args.max_passes:
        pass_index += 1
        print(f"[INFO] recovery_pass={pass_index} remaining={len(remaining_records)}")
        for record in remaining_records:
            print(f"[INFO] generating_cot id={record['id']}")

        if args.num_workers == 1:
            for record in remaining_records:
                persist_result(
                    _process_record(
                        record=record,
                        args=args,
                        dataset_root=dataset_root,
                        sft_system_prompt=sft_system_prompt,
                        openai_system_prompt=openai_system_prompt,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_record = {
                    executor.submit(
                        _process_record,
                        record,
                        args,
                        dataset_root,
                        sft_system_prompt,
                        openai_system_prompt,
                    ): record
                    for record in remaining_records
                }
                for future in as_completed(future_to_record):
                    record = future_to_record[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "status": "failed",
                            "id": record["id"],
                            "sample_id": record["meta"]["sample_id"],
                            "step_index": int(record["meta"]["step_index"]),
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_text_preview": "",
                        }
                    persist_result(result)

        completed_ids = _read_existing_ids(output_jsonl)
        remaining_records = [record for record in remaining_records if record["id"] not in completed_ids]

    missing_rows = [
        {
            "id": record["id"],
            "sample_id": record["meta"]["sample_id"],
            "step_index": int(record["meta"]["step_index"]),
        }
        for record in remaining_records
    ]
    _write_jsonl(missing_jsonl, missing_rows)

    print(f"[INFO] processed={processed}")
    print(f"[INFO] skipped={skipped}")
    print(f"[INFO] failed_events={failed_events}")
    print(f"[INFO] remaining={len(remaining_records)}")
    print(f"[INFO] recovery_passes_used={pass_index}")

    if args.require_complete and remaining_records:
        raise RuntimeError(
            f"Generation incomplete: {len(remaining_records)} records are still missing. "
            f"See {missing_jsonl} and {error_jsonl}."
        )


if __name__ == "__main__":
    main()
