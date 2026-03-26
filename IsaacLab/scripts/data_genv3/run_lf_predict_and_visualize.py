#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


TEXT_PATTERN = re.compile(r"<text>(.*?)</text>", re.DOTALL)
ANSWER_LINE_PATTERN = re.compile(
    r"pickbox@(?P<view>[^:]+):\s*<\|box_start\|>\((?P<x1>\d+),(?P<y1>\d+)\),\((?P<x2>\d+),(?P<y2>\d+)\)<\|box_end\|>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LlamaFactory prediction and visualize generated text + boxes.")
    parser.add_argument(
        "--predict_config",
        type=str,
        default="/mnt/data/lhm/LlamaFactory/examples/inference/qwen2_5vl_lora_predict_pick_with_text_smallsize4.yaml",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/data/lhm/image_bricks/assets/dataset_pick_with_text/smallsize_4",
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default="",
        help="Optional explicit eval jsonl path. Default: infer from dataset_root val split.",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="",
        help="Optional explicit generated_predictions.jsonl path. Default: infer from predict yaml output_dir.",
    )
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--skip_predict", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Default: <dataset_root>/visualizations/predict_compare",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            value = match.group(1).split("#", 1)[0].strip()
            return value.strip("\"'")
    raise KeyError(f"{key} not found in {path}")


def _extract_text(content: str) -> str:
    match = TEXT_PATTERN.search(content)
    return match.group(1).strip() if match else ""


def _extract_boxes(content: str) -> Dict[str, Tuple[int, int, int, int]]:
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for match in ANSWER_LINE_PATTERN.finditer(content):
        boxes[match.group("view")] = (
            int(match.group("x1")),
            int(match.group("y1")),
            int(match.group("x2")),
            int(match.group("y2")),
        )
    return boxes


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], color: Tuple[int, int, int], width: int) -> None:
    draw.rectangle(box, outline=color, width=width)


def _draw_text_block(
    draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont, fill, bg_fill
) -> None:
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 6, top - 4, right + 6, bottom + 4), fill=bg_fill)
    draw.text(xy, text, fill=fill, font=font)


def _annotate_image(
    image_path: Path,
    title: str,
    pred_box: Tuple[int, int, int, int] | None,
    gt_box: Tuple[int, int, int, int] | None,
    pred_text: str | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = _font(26)
    body_font = _font(22)
    _draw_text_block(draw, (12, 12), title, title_font, (0, 0, 0), (255, 214, 10))

    if gt_box is not None:
        _draw_box(draw, gt_box, (40, 220, 80), 5)
        _draw_text_block(draw, (gt_box[0] + 8, max(8, gt_box[1] - 34)), "gt", body_font, (255, 255, 255), (20, 120, 40))
    if pred_box is not None:
        _draw_box(draw, pred_box, (255, 80, 80), 5)
        _draw_text_block(draw, (pred_box[0] + 8, min(image.height - 40, pred_box[3] + 8)), "pred", body_font, (255, 255, 255), (150, 30, 30))

    if pred_text:
        left, top, right, bottom = draw.textbbox((0, 0), pred_text, font=body_font)
        pad = 10
        x = 12
        y = image.height - (bottom - top) - 2 * pad - 12
        draw.rectangle((x, y, x + (right - left) + 2 * pad, y + (bottom - top) + 2 * pad), fill=(0, 0, 0))
        draw.text((x + pad, y + pad), pred_text, fill=(255, 255, 255), font=body_font)

    return image


def _make_grid(images: List[Image.Image], footer_lines: List[str]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    cell_w = max(widths)
    cell_h = max(heights)
    cols = 2
    rows = math.ceil(len(images) / cols)
    footer_h = 110
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + footer_h), (245, 245, 245))
    for idx, img in enumerate(images):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        canvas.paste(img, (x, y))

    draw = ImageDraw.Draw(canvas)
    footer_font = _font(22)
    draw.rectangle((0, rows * cell_h, canvas.width, canvas.height), fill=(30, 30, 30))
    y = rows * cell_h + 12
    for line in footer_lines:
        draw.text((20, y), line, fill=(255, 255, 255), font=footer_font)
        y += 30
    return canvas


def _run_predict(predict_config: Path) -> None:
    workdir = Path("/mnt/data/lhm/LlamaFactory")
    cmd = (
        "source /mnt/data/miniconda3/etc/profile.d/conda.sh && "
        "conda activate llama && "
        f"cd {workdir} && "
        "PYTHONPATH=/mnt/data/lhm/LlamaFactory/src:$PYTHONPATH "
        f"python -m llamafactory.cli train {predict_config}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)


def _visualize_record(record: dict, pred_row: dict, dataset_root: Path, output_dir: Path) -> Path:
    gt_content = record["messages"][-1]["content"]
    pred_content = str(pred_row["predict"])
    gt_text = _extract_text(gt_content)
    pred_text = _extract_text(pred_content)
    gt_boxes = _extract_boxes(gt_content)
    pred_boxes = _extract_boxes(pred_content)

    sample_id = str(record["meta"]["sample_id"])
    step_index = int(record["meta"]["step_index"])
    images = record["images"]
    target_views = record["meta"]["target_views"]
    current_views = record["meta"]["camera_views"]

    panels: List[Image.Image] = []
    for idx, view_name in enumerate(target_views):
        panels.append(
            _annotate_image(dataset_root / images[idx], f"target:{view_name}", None, None, None)
        )

    for offset, view_name in enumerate(current_views, start=len(target_views)):
        panels.append(
            _annotate_image(
                dataset_root / images[offset],
                f"current:{view_name}",
                pred_boxes.get(view_name),
                gt_boxes.get(view_name),
                pred_text or "(no <text> parsed)",
            )
        )

    footer_lines = [
        f"id: {record['id']}",
        f"pred_text: {pred_text or '(empty / unparsable)'}",
        f"gt_text: {gt_text or '(empty)'}",
    ]
    grid = _make_grid(panels, footer_lines)
    output_path = output_dir / sample_id / f"{sample_id}_step{step_index:05d}_predict_viz.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    predict_config = Path(args.predict_config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    eval_jsonl = (
        Path(args.eval_jsonl).resolve()
        if args.eval_jsonl
        else dataset_root / "sharegpt_pickbox_grounding_qwen_text_val.jsonl"
    )
    prediction_file = (
        Path(args.prediction_file).resolve()
        if args.prediction_file
        else Path(_parse_yaml_scalar(predict_config, "output_dir")).resolve() / "generated_predictions.jsonl"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_root / "visualizations" / "predict_compare"
    )

    if not args.skip_predict:
        _run_predict(predict_config)

    records = _read_jsonl(eval_jsonl)
    pred_rows = _read_jsonl(prediction_file)
    if len(records) != len(pred_rows):
        raise ValueError(
            f"Eval record count {len(records)} != prediction row count {len(pred_rows)}. "
            f"Check {eval_jsonl} and {prediction_file}."
        )

    if args.max_samples > 0:
        records = records[: args.max_samples]
        pred_rows = pred_rows[: args.max_samples]

    written: List[Path] = []
    for record, pred_row in zip(records, pred_rows):
        written.append(_visualize_record(record, pred_row, dataset_root, output_dir))

    print(f"wrote {len(written)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
