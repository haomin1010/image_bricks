#!/usr/bin/env python3
"""
Fill empty <thinking></thinking> spans in partial-view SFT data with text generated
by a local Qwen model.

This script is designed as a post-processing step after:
    step4_generate_partialview_sft_v2.py

It reads either:
1) a unified JSONL file such as:
   <dataset_root>/sft_woCoT_partialview_all.jsonl
2) a per-shape JSON file such as:
   <dataset_root>/<shape_id>/<shape_id>_sft_woCoT_partialview.json

For every assistant message in the form:
    <thinking></thinking><action>...</action>
it reconstructs the hidden build context from the dataset metadata, asks a local
Qwen model to write a concise rationale, and rewrites the message as:
    <thinking>...</thinking><action>...</action>

The generated thinking is intended for SFT supervision only. Since the local model
is text-only, the prompt uses structured metadata about the build state and the
fixed action instead of the raw images.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ASSISTANT_ACTION_RE = re.compile(
    r"^\s*<thinking>(?P<thinking>.*?)</thinking><action>(?P<action>.*?)</action>\s*$",
    re.DOTALL,
)
SAMPLE_ID_RE = re.compile(
    r"^(?P<shape_id>.+?)_build_step(?P<step_idx>\d{5})(?P<submit>_submit)?$"
)

THINKING_SYSTEM_PROMPT = """You are writing hidden reasoning traces for a robot block-building dataset.

Rules:
- Output plain text only.
- Output a coherent reasoning trace that can be inserted inside <thinking>...</thinking>.
- Do not emit XML tags, JSON, markdown, lists, labels, or code fences.
- Do not repeat the prompt, the task description, the target summary, or the action text verbatim.
- Do not mention "rationale", "reasoning", "task", "target summary", "current state", "camera query plan", or similar meta wording.
- Do not ask questions, do not start enumerating, and do not explain impossible contradictions.
- Think through the current partial structure before committing to the fixed next action.
- Keep it grounded in the current partial structure and the next action only.
- If the action is a camera query, explain what this view should clarify and why it matters for the next placement or submit decision.
- If the action is a placement, explain why this cube position is supported, how it extends the partial shape correctly, and why it is better than placing elsewhere.
- If the action is submit, explain why the visible structure now appears complete enough to verify.
- Prefer concise but complete reasoning over arbitrary length limits."""


THINKING_FEWSHOT_EXAMPLES = """Examples of good hidden thinking:

Example 1
Context:
- Placed blocks so far: Current structure is empty.
- Turn queries already used: none.
- Next action: query camera 1.
Good output:
I should inspect this view to confirm the footprint before placing the first cube. A wrong first placement would shift the whole build.

Example 2
Context:
- Placed blocks so far: 2 blocks placed. z=0: 2 blocks [(1,0), (2,0)]
- Turn queries already used: 1, 4.
- Next action: place cube at (3, 0, 0).
- Support status: supported.
- Adjacent same-layer neighbors: (2,0,0).
Good output:
This supported placement extends the same confirmed row and keeps the footprint aligned. Placing elsewhere would break the current continuation.

Example 3
Context:
- Placed blocks so far: 6 blocks placed. z=0: 4 blocks [(0,0), (1,0), (0,1), (1,1)] | z=1: 2 blocks [(0,0), (1,0)]
- Turn queries already used: 2, 3.
- Next action: submit.
Good output:
The visible layers already match the intended shape, so submission is appropriate now. I do not see a missing extension that would justify another action.
"""


@dataclass(frozen=True)
class Block:
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class ShapeMeta:
    shape_id: str
    dims: Tuple[int, int, int]
    target_blocks: Tuple[Block, ...]
    step_blocks: Tuple[Tuple[Block, ...], ...]

    @property
    def total_steps(self) -> int:
        return len(self.step_blocks)

    @property
    def total_blocks(self) -> int:
        return len(self.target_blocks)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _extract_blocks(payload: dict) -> Tuple[Block, ...]:
    blocks_raw = payload["original_data"]["blocks"]
    return tuple(Block(x=int(item["x"]), y=int(item["y"]), z=int(item["z"])) for item in blocks_raw)


def _extract_dims(payload: dict) -> Tuple[int, int, int]:
    dims = payload["original_data"]["dimensions"]
    return (int(dims["length"]), int(dims["width"]), int(dims["height"]))


def _discover_step_indices(sample_dir: Path, shape_id: str) -> List[int]:
    pattern = re.compile(rf"^{re.escape(shape_id)}_step(\d{{5}})_data\.json$")
    indices: List[int] = []
    for path in sample_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))

    indices = sorted(set(indices))
    if not indices:
        raise ValueError(f"No step data found under {sample_dir}")

    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        raise ValueError(f"Step indices are not contiguous for {shape_id}: {indices} != {expected}")
    return indices


def _load_shape_meta(dataset_root: Path, shape_id: str) -> ShapeMeta:
    sample_dir = dataset_root / shape_id
    base_payload = _load_json(sample_dir / f"{shape_id}_data.json")
    dims = _extract_dims(base_payload)
    target_blocks = _extract_blocks(base_payload)
    step_indices = _discover_step_indices(sample_dir, shape_id)

    step_blocks: List[Tuple[Block, ...]] = []
    for step_idx in step_indices:
        payload = _load_json(sample_dir / f"{shape_id}_step{step_idx:05d}_data.json")
        step_blocks.append(_extract_blocks(payload))

    return ShapeMeta(
        shape_id=shape_id,
        dims=dims,
        target_blocks=target_blocks,
        step_blocks=tuple(step_blocks),
    )


def _parse_sample_id(sample_id: str) -> Tuple[str, int, bool]:
    match = SAMPLE_ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Unsupported sample id format: {sample_id}")
    shape_id = match.group("shape_id")
    step_idx = int(match.group("step_idx"))
    is_submit = bool(match.group("submit"))
    return shape_id, step_idx, is_submit


def _parse_assistant_action(content: str) -> Optional[Tuple[str, str]]:
    match = ASSISTANT_ACTION_RE.match(content)
    if not match:
        return None
    return match.group("thinking"), match.group("action")


def _classify_action(action_text: str) -> Tuple[str, dict | str]:
    stripped = action_text.strip()
    if stripped == "submit":
        return "submit", stripped

    payload = json.loads(stripped)
    if "query" in payload:
        return "query", payload
    if {"x", "y", "z"} <= set(payload.keys()):
        return "place", payload
    raise ValueError(f"Unsupported action payload: {action_text}")


def _summarize_blocks(blocks: Sequence[Block]) -> str:
    if not blocks:
        return "Current structure is empty."

    by_layer: Dict[int, List[Tuple[int, int]]] = {}
    for block in blocks:
        by_layer.setdefault(block.z, []).append((block.x, block.y))

    parts: List[str] = []
    for z in sorted(by_layer):
        coords = sorted(by_layer[z])
        preview = ", ".join(f"({x},{y})" for x, y in coords[:8])
        if len(coords) > 8:
            preview += ", ..."
        parts.append(f"z={z}: {len(coords)} blocks [{preview}]")

    return f"{len(blocks)} blocks placed. " + " | ".join(parts)


def _current_blocks_for_sample(meta: ShapeMeta, step_idx: int, is_submit: bool) -> Tuple[Block, ...]:
    if is_submit:
        return meta.step_blocks[-1]
    if step_idx <= 1:
        return tuple()
    return meta.step_blocks[step_idx - 2]


def _action_summary(action_type: str, action_payload: dict | str) -> str:
    if action_type == "query":
        query_ids = action_payload["query"]
        return f"query camera {query_ids}"
    if action_type == "place":
        return f"place one cube at ({action_payload['x']}, {action_payload['y']}, {action_payload['z']})"
    return "submit"


def _build_prompt(
    meta: ShapeMeta,
    step_idx: int,
    is_submit: bool,
    current_blocks: Sequence[Block],
    queried_cameras: Sequence[int],
    action_type: str,
    action_payload: dict | str,
) -> str:
    queried_text = "none" if not queried_cameras else ", ".join(str(cam_id) for cam_id in queried_cameras)
    lines = ["Write a grounded first-person hidden thought for the fixed next action.", THINKING_FEWSHOT_EXAMPLES]
    lines.append("Current sample context:")
    lines.append(f"Build phase: {'submit turn' if is_submit else f'step {step_idx} of {meta.total_steps}'} on an 8x8 grid.")
    lines.append(f"Placed blocks so far: {_summarize_blocks(current_blocks)}")
    lines.append(f"Turn queries already used: {queried_text}.")

    if action_type == "place":
        block = Block(
            x=int(action_payload["x"]),
            y=int(action_payload["y"]),
            z=int(action_payload["z"]),
        )
        supported = block.z == 0 or any(
            existing.x == block.x and existing.y == block.y and existing.z == block.z - 1
            for existing in current_blocks
        )
        neighbors = []
        for existing in current_blocks:
            if existing.z == block.z and abs(existing.x - block.x) + abs(existing.y - block.y) == 1:
                neighbors.append(f"({existing.x},{existing.y},{existing.z})")
        neighbor_text = ", ".join(sorted(neighbors)) if neighbors else "none"
        lines.append(f"Next action: place cube at ({block.x}, {block.y}, {block.z}).")
        lines.append(f"Support status: {'supported' if supported else 'unsupported'}.")
        lines.append(f"Adjacent same-layer neighbors: {neighbor_text}.")
        lines.append("Focus on why this placement fits the partial structure and extends the shape correctly.")
    elif action_type == "query":
        query_ids = ",".join(str(cam_id) for cam_id in action_payload["query"])
        lines.append(f"Next action: query camera {query_ids}.")
        lines.append("Focus on what this view clarifies about footprint, height, alignment, or occlusion before acting.")
    else:
        lines.append("Next action: submit.")
        lines.append("Focus on why the visible structure now seems complete enough to verify.")

    lines.append(
        "Output rules: Start directly with the thought. Explain why this action is chosen now and what visual or structural evidence supports it. "
        "Reason naturally for as long as needed, but stay focused and do not mention the prompt or examples."
    )
    return "\n".join(lines)


def _clean_generated_thinking(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "I should make the next action that best matches the observed target and current structure."

    if "</thinking>" in cleaned:
        cleaned = cleaned.split("</thinking>", 1)[0]
    cleaned = cleaned.replace("<thinking>", "").replace("</thinking>", "")
    cleaned = cleaned.replace("<action>", "").replace("</action>", "")
    cleaned = cleaned.replace("<good_thinking>", "").replace("</good_thinking>", "")
    cleaned = cleaned.replace("<example>", "").replace("</example>", "")
    cleaned = cleaned.replace("<examples>", "").replace("</examples>", "")
    cleaned = cleaned.replace("<context>", "").replace("</context>", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    bad_prefixes = (
        "rationale:",
        "reasoning:",
        "the rationale",
        "the reasoning",
        "task:",
        "target summary:",
        "current state:",
        "build context:",
        "next action:",
        "placement support check:",
        "support status:",
        "camera queries already made",
        "turn queries already used:",
        "placed blocks so far:",
        "after this action:",
        "what is the",
        "how is this possible",
    )
    for _ in range(8):
        lowered = cleaned.lower().lstrip()
        matched = False
        for prefix in bad_prefixes:
            if lowered.startswith(prefix):
                parts = re.split(r"[:.]\s*", cleaned, maxsplit=1)
                cleaned = parts[1].strip() if len(parts) > 1 else ""
                matched = True
                break
        if not matched:
            break

    cleaned = re.sub(r"\b(The rationale should|Write one brief thought.*)$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\b(Task|Target summary|Current state|Build context|Next action)\b.*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\b(Good output|Example \d+|Context|Output rules)\b.*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0].strip()
    lowered = cleaned.lower()
    if cleaned and not lowered.startswith(("i ", "i'", "this ", "from ", "with ", "checking ", "querying ", "placing ", "submitting ")):
        cleaned = f"I should {cleaned[0].lower() + cleaned[1:]}" if len(cleaned) > 1 else f"I should {cleaned.lower()}"
    if cleaned and cleaned[-1].isalnum():
        cleaned += "."
    if not cleaned:
        return "I should make the next action that best matches the observed target and current structure."
    return cleaned


class QwenThinkingGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: str,
        device_map: str,
        max_new_tokens: int,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map[torch_dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        # Some Qwen checkpoints ship sampling-only defaults such as top_k in the
        # generation config. We use greedy decoding for stable SFT traces, so clear
        # those fields to avoid transformers warnings about ignored flags.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": THINKING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class VllmThinkingGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: str,
        max_new_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError("vLLM is not available but --backend=vllm was requested.") from exc

        dtype_map = {
            "auto": "auto",
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
        }

        llm_kwargs = {
            "model": model_name_or_path,
            "trust_remote_code": True,
            "dtype": dtype_map[torch_dtype],
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_new_tokens,
            stop=[
                "</thinking>",
                "<action>",
                "<good_thinking>",
                "</good_thinking>",
                "<example>",
                "</example>",
                "Rationale:",
                "Reasoning:",
                "Task:",
                "Good output:",
                "Example 1",
                "Example 2",
                "Example 3",
            ],
        )

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        outputs = self.llm.generate(list(prompts), self.sampling_params, use_tqdm=False)
        texts: List[str] = []
        for output in outputs:
            if not output.outputs:
                texts.append("")
            else:
                texts.append(output.outputs[0].text.strip())
        return texts


def _replace_assistant_messages(
    sample: dict,
    meta: ShapeMeta,
    generator: QwenThinkingGenerator,
    overwrite_existing: bool,
) -> dict:
    sample_id = str(sample["id"])
    _, step_idx, is_submit = _parse_sample_id(sample_id)
    current_blocks = list(_current_blocks_for_sample(meta, step_idx, is_submit))
    queried_cameras: List[int] = []

    new_messages: List[dict] = []
    for message in sample["messages"]:
        if message.get("role") != "assistant":
            new_messages.append(message)
            continue

        parsed = _parse_assistant_action(str(message.get("content", "")))
        if parsed is None:
            new_messages.append(message)
            continue

        old_thinking, action_text = parsed
        if old_thinking.strip() and not overwrite_existing:
            new_messages.append(message)
            action_type, action_payload = _classify_action(action_text)
        else:
            action_type, action_payload = _classify_action(action_text)
            prompt = _build_prompt(
                meta=meta,
                step_idx=step_idx,
                is_submit=is_submit,
                current_blocks=current_blocks,
                queried_cameras=queried_cameras,
                action_type=action_type,
                action_payload=action_payload,
            )
            generated = generator.generate(prompt)
            thinking_text = _clean_generated_thinking(generated)
            new_content = f"<thinking>{thinking_text}</thinking><action>{action_text}</action>"
            new_messages.append({"role": "assistant", "content": new_content})

        if action_type == "query":
            queried_cameras.extend(int(cam_id) for cam_id in action_payload["query"])
        elif action_type == "place":
            current_blocks.append(
                Block(
                    x=int(action_payload["x"]),
                    y=int(action_payload["y"]),
                    z=int(action_payload["z"]),
                )
            )
            queried_cameras = []
        elif action_type == "submit":
            queried_cameras = []

    new_sample = dict(sample)
    new_sample["messages"] = new_messages
    return new_sample


def _collect_prompt_jobs(
    sample: dict,
    meta: ShapeMeta,
    overwrite_existing: bool,
) -> List[dict]:
    sample_id = str(sample["id"])
    _, step_idx, is_submit = _parse_sample_id(sample_id)
    current_blocks = list(_current_blocks_for_sample(meta, step_idx, is_submit))
    queried_cameras: List[int] = []

    jobs: List[dict] = []
    for message_idx, message in enumerate(sample["messages"]):
        if message.get("role") != "assistant":
            continue

        parsed = _parse_assistant_action(str(message.get("content", "")))
        if parsed is None:
            continue

        old_thinking, action_text = parsed
        action_type, action_payload = _classify_action(action_text)

        if not old_thinking.strip() or overwrite_existing:
            prompt = _build_prompt(
                meta=meta,
                step_idx=step_idx,
                is_submit=is_submit,
                current_blocks=current_blocks,
                queried_cameras=queried_cameras,
                action_type=action_type,
                action_payload=action_payload,
            )
            jobs.append(
                {
                    "sample_id": sample_id,
                    "message_idx": message_idx,
                    "action_text": action_text,
                    "prompt": prompt,
                }
            )

        if action_type == "query":
            queried_cameras.extend(int(cam_id) for cam_id in action_payload["query"])
        elif action_type == "place":
            current_blocks.append(
                Block(
                    x=int(action_payload["x"]),
                    y=int(action_payload["y"]),
                    z=int(action_payload["z"]),
                )
            )
            queried_cameras = []
        elif action_type == "submit":
            queried_cameras = []

    return jobs


def _apply_generated_thinking(sample: dict, outputs_by_message_idx: Dict[int, str]) -> dict:
    new_messages: List[dict] = []
    for message_idx, message in enumerate(sample["messages"]):
        replacement = outputs_by_message_idx.get(message_idx)
        if replacement is None:
            new_messages.append(message)
            continue

        parsed = _parse_assistant_action(str(message.get("content", "")))
        if parsed is None:
            new_messages.append(message)
            continue

        _, action_text = parsed
        thinking_text = _clean_generated_thinking(replacement)
        new_messages.append(
            {
                "role": "assistant",
                "content": f"<thinking>{thinking_text}</thinking><action>{action_text}</action>",
            }
        )

    new_sample = dict(sample)
    new_sample["messages"] = new_messages
    return new_sample


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def _discover_per_shape_json_inputs(dataset_root: Path) -> List[Path]:
    inputs: List[Path] = []
    for shape_dir in sorted(dataset_root.iterdir()):
        if not shape_dir.is_dir():
            continue
        shape_id = shape_dir.name
        json_path = shape_dir / f"{shape_id}_sft_woCoT_partialview.json"
        if json_path.exists():
            inputs.append(json_path)
    return inputs


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".jsonl":
        if input_path.name.endswith("_all.jsonl"):
            return input_path.with_name(input_path.stem.replace("woCoT", "wCoT") + ".jsonl")
        return input_path.with_name(input_path.stem + "_wCoT.jsonl")
    if input_path.suffix == ".json":
        return input_path.with_name(input_path.stem.replace("woCoT", "wCoT") + ".json")
    return input_path.with_name(input_path.name + ".with_thinking")


def _detect_tensor_parallel_size(requested_size: int) -> int:
    if requested_size > 0:
        return requested_size

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip():
        parts = [part.strip() for part in visible_devices.split(",") if part.strip()]
        if parts:
            return len(parts)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if gpu_lines:
            return len(gpu_lines)
    except Exception:
        pass

    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill empty <thinking> spans in partial-view SFT data with Qwen.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=str,
        default="/mnt/data/lhm/image_bricks/assets/dataset_v2/small_size4",
        help="Dataset root containing shape folders and step JSON files.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Input JSON/JSONL path. Default: process all per-shape *_sft_woCoT_partialview.json files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSON/JSONL path. Only valid for single-file input. Default: derived from input path.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/mnt/data/Big_Model/Qwen3-235B",
        help="Local Qwen model path used to generate the thinking text.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Passed to transformers.from_pretrained(device_map=...).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens for each thinking span.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples for debugging.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate thinking even if the <thinking> span is already non-empty.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("transformers", "vllm"),
        default="vllm",
        help="Inference backend. vLLM is much faster for batched generation on large models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Prompt batch size used with the vLLM backend.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("VLLM_TP_SIZE", "0")),
        help="vLLM tensor parallel size. <=0 means auto-detect from visible GPUs.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization target.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if args.input_path is not None:
        input_paths = [Path(args.input_path).expanduser().resolve()]
    else:
        input_paths = _discover_per_shape_json_inputs(dataset_root)

    if not input_paths:
        raise FileNotFoundError(f"No input files found under {dataset_root}")

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.limit is not None:
        input_paths = input_paths[: args.limit]

    tensor_parallel_size = _detect_tensor_parallel_size(args.tensor_parallel_size)
    print(f"[INFO] backend={args.backend}")
    print(f"[INFO] tensor_parallel_size={tensor_parallel_size}")
    print(f"[INFO] input_files={len(input_paths)}")

    if args.output_path is not None and len(input_paths) != 1:
        raise ValueError("--output-path is only supported when processing a single input file.")

    if args.backend == "transformers":
        generator = QwenThinkingGenerator(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        generator = VllmThinkingGenerator(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    total_records = 0
    for file_index, input_path in enumerate(input_paths, start=1):
        output_path = (
            Path(args.output_path).expanduser().resolve()
            if args.output_path is not None
            else _default_output_path(input_path)
        )

        if input_path.suffix == ".jsonl":
            records = list(_iter_jsonl(input_path))
        elif input_path.suffix == ".json":
            records = _load_json(input_path)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array in {input_path}")
        else:
            raise ValueError(f"Unsupported input suffix: {input_path.suffix}")

        shape_cache: Dict[str, ShapeMeta] = {}
        for sample in records:
            shape_id, _, _ = _parse_sample_id(str(sample["id"]))
            if shape_id not in shape_cache:
                shape_cache[shape_id] = _load_shape_meta(dataset_root, shape_id)

        processed: List[dict] = []
        if args.backend == "transformers":
            for index, sample in enumerate(records, start=1):
                shape_id, _, _ = _parse_sample_id(str(sample["id"]))
                new_sample = _replace_assistant_messages(
                    sample=sample,
                    meta=shape_cache[shape_id],
                    generator=generator,
                    overwrite_existing=args.overwrite_existing,
                )
                processed.append(new_sample)
                print(f"[{file_index}/{len(input_paths)}] [{index}/{len(records)}] processed {sample['id']}")
        else:
            jobs_by_sample_id: Dict[str, List[dict]] = {}
            all_jobs: List[dict] = []
            for sample in records:
                shape_id, _, _ = _parse_sample_id(str(sample["id"]))
                jobs = _collect_prompt_jobs(
                    sample=sample,
                    meta=shape_cache[shape_id],
                    overwrite_existing=args.overwrite_existing,
                )
                jobs_by_sample_id[str(sample["id"])] = jobs
                all_jobs.extend(jobs)

            print(f"[INFO] file={input_path.name} vLLM prompt jobs: {len(all_jobs)}")
            generated_texts: List[str] = []
            for start in range(0, len(all_jobs), args.batch_size):
                batch_jobs = all_jobs[start : start + args.batch_size]
                batch_prompts = [job["prompt"] for job in batch_jobs]
                batch_outputs = generator.generate_batch(batch_prompts)
                generated_texts.extend(batch_outputs)
                print(
                    f"[INFO] file={input_path.name} generated "
                    f"{min(start + len(batch_jobs), len(all_jobs))}/{len(all_jobs)} prompts"
                )

            generated_iter = iter(generated_texts)
            for index, sample in enumerate(records, start=1):
                sample_outputs: Dict[int, str] = {}
                for job in jobs_by_sample_id[str(sample["id"])]:
                    sample_outputs[int(job["message_idx"])] = next(generated_iter)
                processed.append(_apply_generated_thinking(sample, sample_outputs))
                print(f"[{file_index}/{len(input_paths)}] [{index}/{len(records)}] processed {sample['id']}")

        if input_path.suffix == ".jsonl":
            _write_jsonl(output_path, processed)
        else:
            _write_json(output_path, processed)

        total_records += len(processed)
        print(f"[OK] input : {input_path}")
        print(f"[OK] output: {output_path}")
        print(f"[OK] records: {len(processed)}")
        print("-" * 72)

    print(f"[SUMMARY] files={len(input_paths)} total_records={total_records}")


if __name__ == "__main__":
    main()
