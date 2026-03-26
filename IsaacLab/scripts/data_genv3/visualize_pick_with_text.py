#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


TEXT_PATTERN = re.compile(r"<text>(.*?)</text>", re.DOTALL)
ANSWER_LINE_PATTERN = re.compile(
    r"pickbox@(?P<view>[^:]+):\s*<\|box_start\|>\((?P<x1>\d+),(?P<y1>\d+)\),\((?P<x2>\d+),(?P<y2>\d+)\)<\|box_end\|>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize text + pickbox annotations on dataset images.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/data/lhm/image_bricks/assets/dataset_pick_with_text/smallsize_4",
        help="Dataset root containing struct/, sft_train/ and json/jsonl files.",
    )
    parser.add_argument("--input_json", type=str, default="", help="Single *_sharegpt_pickbox_qwen_text.json file.")
    parser.add_argument("--input_jsonl", type=str, default="", help="JSONL file to visualize in batch.")
    parser.add_argument("--sample_ids", type=str, default="", help="Optional comma-separated sample ids for JSONL mode.")
    parser.add_argument("--max_samples", type=int, default=20, help="Max records to render in JSONL mode.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Default: <dataset_root>/visualizations/pick_with_text",
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


def _extract_text(assistant_content: str) -> str:
    match = TEXT_PATTERN.search(assistant_content)
    return match.group(1).strip() if match else ""


def _extract_boxes(assistant_content: str) -> Dict[str, Tuple[int, int, int, int]]:
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for match in ANSWER_LINE_PATTERN.finditer(assistant_content):
        boxes[match.group("view")] = (
            int(match.group("x1")),
            int(match.group("y1")),
            int(match.group("x2")),
            int(match.group("y2")),
        )
    return boxes


def _sample_filter(sample_ids_raw: str) -> set[str]:
    if not sample_ids_raw.strip():
        return set()
    return {item.strip() for item in sample_ids_raw.split(",") if item.strip()}


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 4, top - 2, right + 4, bottom + 2), fill=(255, 196, 0))
    draw.text((xy[0], xy[1]), text, fill=(0, 0, 0), font=font)


def _annotate_image(
    image_path: Path,
    panel_title: str,
    box: Tuple[int, int, int, int] | None,
    body_text: str | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    body_font = _font(24)

    if box is not None:
        draw.rectangle(box, outline=(255, 80, 80), width=6)
        _draw_label(draw, (box[0] + 8, max(8, box[1] - 32)), panel_title, title_font)
    else:
        _draw_label(draw, (12, 12), panel_title, title_font)

    if body_text:
        left, top, right, bottom = draw.textbbox((0, 0), body_text, font=body_font)
        pad = 10
        x = 12
        y = image.height - (bottom - top) - 2 * pad - 12
        draw.rectangle((x, y, x + (right - left) + 2 * pad, y + (bottom - top) + 2 * pad), fill=(0, 0, 0))
        draw.text((x + pad, y + pad), body_text, fill=(255, 255, 255), font=body_font)

    return image


def _make_grid(images: List[Image.Image], footer: str) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    cell_w = max(widths)
    cell_h = max(heights)
    cols = 2
    rows = math.ceil(len(images) / cols)
    footer_h = 72
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + footer_h), (245, 245, 245))
    for idx, img in enumerate(images):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        canvas.paste(img, (x, y))

    draw = ImageDraw.Draw(canvas)
    footer_font = _font(26)
    draw.rectangle((0, rows * cell_h, canvas.width, canvas.height), fill=(30, 30, 30))
    draw.text((20, rows * cell_h + 20), footer, fill=(255, 255, 255), font=footer_font)
    return canvas


def _visualize_record(record: dict, dataset_root: Path, output_dir: Path) -> Path:
    assistant_content = record["messages"][-1]["content"]
    text = _extract_text(assistant_content)
    boxes = _extract_boxes(assistant_content)
    sample_id = str(record["meta"]["sample_id"])
    step_index = int(record["meta"]["step_index"])
    struct_views = record["meta"]["target_views"]
    current_views = record["meta"]["camera_views"]

    images = record["images"]
    panel_images: List[Image.Image] = []

    for idx, view_name in enumerate(struct_views):
        image_path = dataset_root / images[idx]
        panel_images.append(_annotate_image(image_path, f"target:{view_name}", None, None))

    for offset, view_name in enumerate(current_views, start=len(struct_views)):
        image_path = dataset_root / images[offset]
        panel_images.append(_annotate_image(image_path, f"pickbox:{view_name}", boxes.get(view_name), text))

    footer = f"{record['id']} | {text}"
    grid = _make_grid(panel_images, footer)
    output_path = output_dir / sample_id / f"{sample_id}_step{step_index:05d}_viz.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_root / "visualizations" / "pick_with_text"
    )

    if args.input_json:
        records = [_load_json(Path(args.input_json).resolve())]
    else:
        input_jsonl = (
            Path(args.input_jsonl).resolve()
            if args.input_jsonl
            else dataset_root / "sharegpt_pickbox_grounding_qwen_text_val.jsonl"
        )
        records = _read_jsonl(input_jsonl)
        sample_filter = _sample_filter(args.sample_ids)
        if sample_filter:
            records = [row for row in records if str(row["meta"]["sample_id"]) in sample_filter]
        if args.max_samples > 0:
            records = records[: args.max_samples]

    written: List[Path] = []
    for record in records:
        written.append(_visualize_record(record, dataset_root, output_dir))

    print(f"wrote {len(written)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
