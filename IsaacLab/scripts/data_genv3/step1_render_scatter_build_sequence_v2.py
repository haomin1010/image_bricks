#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np


def _default_output_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/dataset_v3/smallsize4")
    )


def _generate_structure(seed: int) -> tuple[list[tuple[int, int, int]], dict]:
    rng = random.Random(int(seed) + 17)
    num_blocks = rng.randint(1, 10)

    grid = np.zeros((4, 4, 4), dtype=bool)
    sx, sy = rng.randint(0, 3), rng.randint(0, 3)
    grid[sx, sy, 0] = True
    current = 1

    while current < num_blocks:
        candidates: list[tuple[int, int, int]] = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if grid[x, y, z]:
                        continue
                    if z > 0 and not grid[x, y, z - 1]:
                        continue
                    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and grid[nx, ny, nz]:
                            candidates.append((x, y, z))
                            break
        if not candidates:
            break
        px, py, pz = rng.choice(candidates)
        grid[px, py, pz] = True
        current += 1

    coords = [tuple(int(v) for v in p) for p in np.argwhere(grid).tolist()]
    coords.sort(key=lambda p: (p[2], p[0], p[1]))

    payload = {
        "rules": "small_size4_connected_stable",
        "dimensions": {"length": 4, "width": 4, "height": 4},
        "seed": int(seed),
        "total_blocks": int(len(coords)),
        "blocks_canonical": [{"id": int(i), "x": int(x), "y": int(y), "z": int(z)} for i, (x, y, z) in enumerate(coords)],
    }
    return coords, payload


def _all_build_orders(blocks: list[tuple[int, int, int]]) -> list[list[int]]:
    idx_to_block = {i: b for i, b in enumerate(blocks)}
    remaining = set(idx_to_block.keys())
    placed: set[int] = set()
    out: list[list[int]] = []

    def dfs(order: list[int]) -> None:
        if not remaining:
            out.append(list(order))
            return
        cands: list[int] = []
        for i in sorted(remaining):
            x, y, z = idx_to_block[i]
            if z == 0:
                cands.append(i)
            else:
                below_idx = None
                for j, (bx, by, bz) in idx_to_block.items():
                    if bx == x and by == y and bz == z - 1:
                        below_idx = j
                        break
                if below_idx is not None and below_idx in placed:
                    cands.append(i)
        for i in cands:
            remaining.remove(i)
            placed.add(i)
            order.append(i)
            dfs(order)
            order.pop()
            placed.remove(i)
            remaining.add(i)

    dfs([])
    return out


def _orders_to_coords(
    orders: list[list[int]],
    blocks: list[tuple[int, int, int]],
) -> list[list[dict[str, int]]]:
    out: list[list[dict[str, int]]] = []
    for order in orders:
        seq: list[dict[str, int]] = []
        for idx in order:
            x, y, z = blocks[int(idx)]
            seq.append({"x": int(x), "y": int(y), "z": int(z)})
        out.append(seq)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1: generate structures + all build orders into struct/.")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--output_count", type=int, default=100)
    parser.add_argument("--start_id", type=int, default=1, help="Starting struct id number, e.g. 1001.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for structure generation.")
    args = parser.parse_args()
    if int(args.output_count) <= 0:
        raise ValueError("output_count must be > 0")
    if int(args.start_id) <= 0:
        raise ValueError("start_id must be > 0")

    output_root = os.path.abspath(os.path.expanduser(args.output_root)) if args.output_root else _default_output_root()
    struct_root = os.path.join(output_root, "struct")
    os.makedirs(struct_root, exist_ok=True)

    for i in range(int(args.output_count)):
        struct_id = f"{int(args.start_id) + i:05d}"
        seed = int(args.base_seed) + i * 1000
        blocks, struct_payload = _generate_structure(seed)
        orders = _all_build_orders(blocks)
        struct_payload["struct_id"] = struct_id
        struct_payload["num_orders"] = int(len(orders))

        struct_dir = os.path.join(struct_root, struct_id)
        os.makedirs(struct_dir, exist_ok=True)
        struct_json = os.path.join(struct_dir, f"{struct_id}_structure.json")
        orders_json = os.path.join(struct_dir, f"{struct_id}_all_orders.json")

        with open(struct_json, "w", encoding="utf-8") as f:
            json.dump(struct_payload, f, indent=2, ensure_ascii=False)

        with open(orders_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "struct_id": struct_id,
                    "seed": int(seed),
                    "total_orders": int(len(orders)),
                    "orders_build_coords": _orders_to_coords(orders, blocks),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[OK] struct={struct_id} blocks={len(blocks)} orders={len(orders)}")


if __name__ == "__main__":
    main()
