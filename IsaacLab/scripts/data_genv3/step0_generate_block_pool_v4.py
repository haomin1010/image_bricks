#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a pool of colored block USDA assets and metadata.")
    parser.add_argument("--output_root", type=str, required=True, help="Dataset root, e.g. assets/dataset_v4")
    parser.add_argument("--pool_size", type=int, default=100, help="Number of colored block assets to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for color pool generation.")
    return parser.parse_args()


def make_colors(n: int, seed: int) -> list[tuple[float, float, float]]:
    import random

    rng = random.Random(seed + 137)
    hues = [((i / float(max(1, n))) + rng.uniform(-0.03, 0.03)) % 1.0 for i in range(n)]
    rng.shuffle(hues)
    out = []
    for h in hues:
        s, v = rng.uniform(0.72, 0.96), rng.uniform(0.36, 0.66)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        mx = max(r, g, b)
        if mx > 0.72:
            scale = 0.72 / mx
            r, g, b = r * scale, g * scale, b * scale
        out.append((float(r), float(g), float(b)))
    return out


def main():
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    asset_dir = output_root / "block_pool"
    asset_dir.mkdir(parents=True, exist_ok=True)
    base_usda = (output_root.parent / "dataset_v3" / "bordered_blue_block.usda").resolve()
    if not base_usda.is_file():
        raise FileNotFoundError(f"Base USDA not found: {base_usda}")
    template = base_usda.read_text(encoding="utf-8")

    colors = make_colors(int(args.pool_size), int(args.seed))
    metadata = []
    for idx, (r, g, b) in enumerate(colors, start=1):
        asset_id = f"{idx:03d}"
        asset_name = f"block_{asset_id}.usda"
        asset_path = asset_dir / asset_name
        content = template.replace(
            "color3f inputs:diffuse_color_constant = (0, 0.75, 1)",
            f"color3f inputs:diffuse_color_constant = ({r:.6f}, {g:.6f}, {b:.6f})",
        )
        asset_path.write_text(content, encoding="utf-8")
        metadata.append(
            {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "asset_relpath": f"block_pool/{asset_name}",
                "color_rgb": [r, g, b],
            }
        )

    (asset_dir / "color_pool.json").write_text(
        json.dumps({"pool_size": len(metadata), "seed": int(args.seed), "assets": metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(asset_dir / "color_pool.json")


if __name__ == "__main__":
    main()
