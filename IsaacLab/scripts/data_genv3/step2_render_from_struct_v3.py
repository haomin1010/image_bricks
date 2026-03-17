#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import os
import random
import sys

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Step2: sample from struct/orders and render images + step json only.")
parser.add_argument("--output_root", type=str, default="", help="Default: assets/dataset_v3/smallsize4")
parser.add_argument("--output_count", type=int, default=100, help="How many rendered samples to generate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if int(args_cli.output_count) <= 0:
    raise ValueError("output_count must be > 0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from PIL import Image, ImageDraw

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import CameraCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass


def _root() -> str:
    if args_cli.output_root:
        return os.path.abspath(os.path.expanduser(args_cli.output_root))
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/dataset_v3/smallsize4")
    )


def _grid_to_world(gx: int, gy: int, gz: int) -> tuple[float, float, float]:
    return -0.204 + gx * 0.051 + 0.0255, -0.204 + gy * 0.051 + 0.0255, 0.025 + gz * 0.05


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    h = 0.5 * yaw
    return float(np.cos(h)), 0.0, 0.0, float(np.sin(h))


def _colors(n: int, seed: int) -> list[tuple[float, float, float]]:
    rng = random.Random(seed + 137)
    hues = [((i / float(max(1, n))) + rng.uniform(-0.06, 0.06)) % 1.0 for i in range(n)]
    rng.shuffle(hues)
    out = []
    for h in hues:
        s, v = rng.uniform(0.72, 0.96), rng.uniform(0.36, 0.66)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        mx = max(r, g, b)
        if mx > 0.72:
            k = 0.72 / mx
            r, g, b = r * k, g * k, b * k
        out.append((float(r), float(g), float(b)))
    return out


def _scatter(target_coords: list[tuple[int, int, int]], seed: int) -> list[dict]:
    rng = random.Random(seed)
    txy = [(_grid_to_world(x, y, z)[0], _grid_to_world(x, y, z)[1]) for (x, y, z) in target_coords]
    out, attempts = [], 0
    md, tc = 0.06, 0.055
    while len(out) < len(target_coords) and attempts < 25000:
        attempts += 1
        if attempts % 4000 == 0:
            md *= 0.92
            tc *= 0.95
        wx, wy = rng.uniform(-0.1785, 0.1785), rng.uniform(-0.1785, 0.1785)
        if any((wx - tx) ** 2 + (wy - ty) ** 2 < tc**2 for tx, ty in txy):
            continue
        if any((wx - s["world"][0]) ** 2 + (wy - s["world"][1]) ** 2 < md**2 for s in out):
            continue
        gx = max(0, min(7, int(round((wx + 0.204 - 0.0255) / 0.051))))
        gy = max(0, min(7, int(round((wy + 0.204 - 0.0255) / 0.051))))
        if 0 <= gx < 4 and 0 <= gy < 4:
            continue
        yaw = rng.uniform(-np.pi, np.pi)
        out.append({"grid": (gx, gy, 0), "world": (float(wx), float(wy), 0.025), "quat": _yaw_quat(yaw), "yaw_deg": float(np.degrees(yaw))})
    if len(out) < len(target_coords):
        raise RuntimeError("scatter failed")
    return out


def _usd() -> str:
    p3 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/dataset_v3/bordered_blue_block.usda"))
    p2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/dataset_v2/bordered_blue_block.usda"))
    if os.path.isfile(p3):
        return p3
    if os.path.isfile(p2):
        return p2
    raise FileNotFoundError("bordered_blue_block.usda not found")


def _cam_cfg(path: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> CameraCfg:
    return CameraCfg(
        prim_path=path,
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955 / (1280.0 / 720.0),
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
    )


def _look_at_quat(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    eye_v = np.array(eye, dtype=np.float64)
    target_v = np.array(target, dtype=np.float64)
    up_v = np.array(world_up, dtype=np.float64)

    forward = target_v - eye_v
    forward /= np.linalg.norm(forward)

    up_proj = up_v - np.dot(up_v, forward) * forward
    up_proj /= np.linalg.norm(up_proj)

    side = np.cross(up_proj, forward)
    side /= np.linalg.norm(side)

    rot = np.stack([forward, side, up_proj], axis=1)
    quat = math_utils.quat_from_matrix(torch.tensor(rot, dtype=torch.float32).unsqueeze(0))[0].cpu().numpy()
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _scene_cfg(n: int, cols: list[tuple[float, float, float]]):
    usd = _usd()

    @configclass
    class Cfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(size=(3.2, 3.2, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.91, 0.93), roughness=0.9), collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        )
        dome_light = AssetBaseCfg(prim_path="/World/dome_light", spawn=sim_utils.DomeLightCfg(intensity=1450.0, color=(0.98, 0.98, 1.0)))
        side_light = AssetBaseCfg(
            prim_path="/World/side_light",
            spawn=sim_utils.DistantLightCfg(intensity=580.0, angle=0.0, color=(0.95, 0.95, 0.95)),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.55, 0.45, 0.0, 0.0)),
        )
        wall_s = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/wall_s", spawn=sim_utils.CuboidCfg(size=(0.214, 0.010, 0.004), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.55), collision_props=None), init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.102, -0.209, 0.0021)))
        wall_n = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/wall_n", spawn=sim_utils.CuboidCfg(size=(0.214, 0.010, 0.004), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.55), collision_props=None), init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.102, 0.005, 0.0021)))
        wall_w = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/wall_w", spawn=sim_utils.CuboidCfg(size=(0.010, 0.214, 0.004), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.55), collision_props=None), init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.209, -0.102, 0.0021)))
        wall_e = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/wall_e", spawn=sim_utils.CuboidCfg(size=(0.010, 0.214, 0.004), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.55), collision_props=None), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.005, -0.102, 0.0021)))
        origin_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/origin_marker",
            spawn=sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.24, 0.16), roughness=0.28),
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.209, -0.209, 0.0085)),
        )
        camera_oblique_main = _cam_cfg(
            "{ENV_REGEX_NS}/CameraObliqueMain",
            (0.62, -0.08, 0.52),
            _look_at_quat((0.62, -0.08, 0.52), (-0.102, -0.102, 0.03)),
        )
        camera_oblique_side = _cam_cfg(
            "{ENV_REGEX_NS}/CameraObliqueSide",
            (-0.08, 0.62, 0.52),
            _look_at_quat((-0.08, 0.62, 0.52), (-0.102, -0.102, 0.03)),
        )

    for i in range(n):
        setattr(
            Cfg,
            f"block_{i}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Block_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd,
                    scale=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cols[i], metallic=0.0, roughness=0.62),
                    visual_material_path="random_material",
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.0, -5.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        )
    return Cfg


def _entries(scene: InteractiveScene) -> dict:
    return {
        "oblique_main": {"obj": scene["camera_oblique_main"], "prim": "/World/envs/env_0/CameraObliqueMain"},
        "oblique_side": {"obj": scene["camera_oblique_side"], "prim": "/World/envs/env_0/CameraObliqueSide"},
    }


def _set_layout(scene: InteractiveScene, target: list[tuple[int, int, int]], scatter: list[dict], step_idx: int):
    env_ids = torch.tensor([0], device=scene.device, dtype=torch.int32)
    zero = torch.zeros((1, 6), device=scene.device, dtype=torch.float32)
    for i in range(len(target)):
        if i < step_idx:
            wx, wy, wz = _grid_to_world(*target[i])
            quat = (1.0, 0.0, 0.0, 0.0)
        else:
            wx, wy, wz = scatter[i]["world"]
            quat = scatter[i]["quat"]
        pose = torch.tensor([[wx, wy, wz, quat[0], quat[1], quat[2], quat[3]]], device=scene.device, dtype=torch.float32)
        b = scene[f"block_{i}"]
        b.write_root_pose_to_sim(pose, env_ids=env_ids)
        b.write_root_velocity_to_sim(zero, env_ids=env_ids)


def _hide_blocks(scene: InteractiveScene, start_idx: int, total_blocks: int):
    env_ids = torch.tensor([0], device=scene.device, dtype=torch.int32)
    zero = torch.zeros((1, 6), device=scene.device, dtype=torch.float32)
    for i in range(int(start_idx), int(total_blocks)):
        pose = torch.tensor([[-5.0, -5.0, -5.0, 1.0, 0.0, 0.0, 0.0]], device=scene.device, dtype=torch.float32)
        b = scene[f"block_{i}"]
        b.write_root_pose_to_sim(pose, env_ids=env_ids)
        b.write_root_velocity_to_sim(zero, env_ids=env_ids)


def _proj(stage, camera_prim: str, xyz: tuple[float, float, float], w: int, h: int) -> dict | None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(camera_prim)
    if not prim.IsValid():
        return None
    fr = UsdGeom.Camera(prim).GetCamera().frustum
    p_view = fr.ComputeViewMatrix().Transform(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    p_ndc = fr.ComputeProjectionMatrix().Transform(p_view)
    nx, ny, nz = float(p_ndc[0]), float(p_ndc[1]), float(p_ndc[2])
    if not (np.isfinite(nx) and np.isfinite(ny) and np.isfinite(nz)):
        return None
    u = float((nx * 0.5 + 0.5) * (w - 1))
    v = float((1.0 - (ny * 0.5 + 0.5)) * (h - 1))
    return {"valid": bool(float(p_view[2]) < 0.0), "u_px": u, "v_px": v, "depth_view": float(p_view[2]), "inside_image": bool(0.0 <= u < w and 0.0 <= v < h), "method": "usd_view_projection"}


def _bbox(p: dict | None, w: int, h: int) -> dict | None:
    if p is None or not bool(p.get("valid", False)):
        return None
    half = max(2, int(round(27 * (float(min(w, h)) / 224.0)))) // 2
    cx, cy = int(round(float(p["u_px"]))), int(round(float(p["v_px"])))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w - 1, cx + half), min(h - 1, cy + half)
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "width": x1 - x0 + 1, "height": y1 - y0 + 1}


def _dash(draw: ImageDraw.ImageDraw, p0, p1, color, lw, d, g):
    x0, y0, x1, y1 = float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    L = float(np.hypot(dx, dy))
    if L <= 1.0e-6:
        return
    ux, uy = dx / L, dy / L
    t = 0.0
    while t < L:
        t2 = min(L, t + d)
        draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * t2, y0 + uy * t2)], fill=color, width=lw)
        t += d + g


def _overlay(rgb: np.ndarray, tb: dict | None, sb: dict | None) -> np.ndarray:
    im = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(im)
    lw = max(5, int(round(float(min(rgb.shape[0], rgb.shape[1])) * 0.0066)))
    d, g = max(10, int(round(float(lw) * 2.4))), max(6, int(round(float(lw) * 1.3)))
    for b, c in [(tb, (255, 80, 80)), (sb, (70, 140, 255))]:
        if b is None:
            continue
        x0, y0, x1, y1 = int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])
        _dash(dr, (x0, y0), (x1, y0), c, lw, d, g)
        _dash(dr, (x1, y0), (x1, y1), c, lw, d, g)
        _dash(dr, (x1, y1), (x0, y1), c, lw, d, g)
        _dash(dr, (x0, y1), (x0, y0), c, lw, d, g)
    return np.array(im)


def _discover_structs(struct_root: str) -> list[tuple[str, str, str]]:
    out = []
    for name in sorted(os.listdir(struct_root)):
        d = os.path.join(struct_root, name)
        if not os.path.isdir(d):
            continue
        s = os.path.join(d, f"{name}_structure.json")
        o = os.path.join(d, f"{name}_all_orders.json")
        if os.path.isfile(s) and os.path.isfile(o):
            out.append((name, s, o))
    return out


def _load_struct_canonical(struct_json: str) -> tuple[list[tuple[int, int, int]], dict]:
    s = json.load(open(struct_json, "r", encoding="utf-8"))
    canonical = [(int(b["x"]), int(b["y"]), int(b["z"])) for b in s["blocks_canonical"]]
    return canonical, s


def _load_struct_and_order(struct_json: str, orders_json: str, rng: random.Random) -> tuple[list[tuple[int, int, int]], dict, int]:
    s = json.load(open(struct_json, "r", encoding="utf-8"))
    o = json.load(open(orders_json, "r", encoding="utf-8"))
    all_orders = o["orders_build_coords"]
    order_idx = rng.randrange(len(all_orders))
    order = all_orders[order_idx]
    target = [(int(p["x"]), int(p["y"]), int(p["z"])) for p in order]
    return target, s, int(order_idx)


def _target_world_list(target: list[tuple[int, int, int]]) -> list[dict]:
    out = []
    for i, (gx, gy, gz) in enumerate(target):
        wx, wy, wz = _grid_to_world(gx, gy, gz)
        out.append({"id": int(i + 1), "grid_x": int(gx), "grid_y": int(gy), "grid_z": int(gz), "world_x": float(wx), "world_y": float(wy), "world_z": float(wz)})
    return out


def _color_list(cols: list[tuple[float, float, float]]) -> list[dict]:
    return [{"id": int(i + 1), "block_name": f"Block_{i}", "color_rgb": [float(r), float(g), float(b)]} for i, (r, g, b) in enumerate(cols)]


def _collect_step_record(step_idx: int, target: list[tuple[int, int, int]], scatter: list[dict], images: list[str], views: list[str], bbox: dict):
    placed = [{"x": int(x), "y": int(y), "z": int(z)} for (x, y, z) in target[:step_idx]]
    scattered = []
    for item in scatter[step_idx:]:
        gx, gy, gz = item["grid"]
        wx, wy, wz = item["world"]
        scattered.append({"grid_x": int(gx), "grid_y": int(gy), "grid_z": int(gz), "world_x": float(wx), "world_y": float(wy), "world_z": float(wz), "yaw_deg": float(item["yaw_deg"])})
    block_states = []
    for i, (gx, gy, gz) in enumerate(target):
        tx, ty, tz = _grid_to_world(gx, gy, gz)
        if i < step_idx:
            cx, cy, cz = tx, ty, tz
            source = "placed"
            yaw_deg = 0.0
        else:
            cx, cy, cz = scatter[i]["world"]
            source = "scatter"
            yaw_deg = float(scatter[i]["yaw_deg"])
        block_states.append(
            {
                "id": int(i + 1),
                "target_grid_xyz": [int(gx), int(gy), int(gz)],
                "target_world_xyz": [float(tx), float(ty), float(tz)],
                "current_world_xyz": [float(cx), float(cy), float(cz)],
                "current_source": source,
                "current_yaw_deg": yaw_deg,
                "is_placed": bool(i < step_idx),
            }
        )
    return {
        "step_index": int(step_idx),
        "total_steps": int(len(target)),
        "placed_count": int(len(placed)),
        "scattered_count": int(len(scattered)),
        "placed_blocks": placed,
        "scattered_blocks": scattered,
        "block_states": block_states,
        "generated_images": images,
        "camera_views": views,
        "annotation_multiview_bounding_box": bbox,
    }


def _write_sample_json(out_dir: str, sample_id: str, target: list[tuple[int, int, int]], cols: list[tuple[float, float, float]], struct_meta: dict, order_index: int, steps: list[dict], structure_target_images: dict):
    payload = {
        "source_file": "__from_struct_orders__",
        "sample_id": sample_id,
        "struct_id": struct_meta["struct_id"],
        "order_index": int(order_index),
        "target_blocks_ordered": [{"x": int(x), "y": int(y), "z": int(z)} for (x, y, z) in target],
        "target_final_absolute_world_positions": _target_world_list(target),
        "block_colors": _color_list(cols),
        "structure": {
            "structure_folder": os.path.join("struct", str(struct_meta["struct_id"])).replace("\\", "/"),
            "target_images_by_view": structure_target_images,
        },
        "steps": steps,
    }
    p = os.path.join(out_dir, f"{sample_id}_data.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[SAVE]: {p}")


def _cleanup_legacy_step_json(out_dir: str, sample_id: str):
    prefix = f"{sample_id}_step"
    suffix = "_data.json"
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            p = os.path.join(out_dir, name)
            try:
                os.remove(p)
                print(f"[CLEAN]: {p}")
            except Exception:
                pass


def _set_ortho_cameras(sim):
    # Keep all cameras as perspective (PinholeCameraCfg default).
    return


def _warmup(sim, scene, dt: float, steps: int):
    for _ in range(int(steps)):
        sim.step()
        scene.update(dt=dt)


def _build_runtime(device: str) -> dict:
    total_blocks = 10
    all_cols = _colors(total_blocks, seed=42)
    cfg = _scene_cfg(total_blocks, all_cols)
    sim_cfg = SimulationCfg(
        device=device,
        dt=0.01,
        render_interval=1,
        render=sim_utils.RenderCfg(antialiasing_mode="FXAA", enable_dlssg=False, enable_dl_denoiser=False),
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    _set_ortho_cameras(sim)
    _warmup(sim, scene, sim_cfg.dt, 30)
    return {"sim": sim, "scene": scene, "sim_cfg": sim_cfg, "entries": _entries(scene), "all_cols": all_cols, "total_blocks": total_blocks}


def _reset_episode(runtime: dict):
    sim = runtime["sim"]
    scene = runtime["scene"]
    sim_cfg = runtime["sim_cfg"]
    sim.reset()
    _warmup(sim, scene, sim_cfg.dt, 30)


def _render_one(sample_id: str, target: list[tuple[int, int, int]], struct_meta: dict, order_index: int, output_root: str, seed: int, runtime: dict):
    out_dir = os.path.join(output_root, "sft_train", sample_id)
    views_dir = os.path.join(out_dir, "views")
    bbox_dir = os.path.join(out_dir, "bbox_overlay")
    os.makedirs(out_dir, exist_ok=True)
    _cleanup_legacy_step_json(out_dir, sample_id)
    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    struct_dir = os.path.join(output_root, "struct", str(struct_meta["struct_id"]))
    os.makedirs(struct_dir, exist_ok=True)

    sim = runtime["sim"]
    scene = runtime["scene"]
    sim_cfg = runtime["sim_cfg"]
    entries = runtime["entries"]
    total_blocks = int(runtime["total_blocks"])
    cols = runtime["all_cols"][: len(target)]
    scatter = _scatter(target, seed=seed + 701)
    steps = []
    _reset_episode(runtime)
    for step_idx in range(len(target) + 1):
        _set_layout(scene, target, scatter, step_idx)
        _hide_blocks(scene, len(target), total_blocks)
        _warmup(sim, scene, sim_cfg.dt, 12)

        images, views, rgb_by_view = [], [], {}
        for vn, e in entries.items():
            rgb = e["obj"].data.output["rgb"][0].cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            rgb_u8 = rgb.astype(np.uint8)
            fn = f"{sample_id}_step{step_idx:05d}_{vn}.png"
            rel = os.path.join("views", fn).replace("\\", "/")
            p = os.path.join(views_dir, fn)
            Image.fromarray(rgb_u8).save(p)
            images.append(rel)
            views.append(vn)
            rgb_by_view[vn] = rgb_u8
            print(f"[SAVE]: {p}")
            if step_idx == len(target):
                target_name = f"{struct_meta['struct_id']}_{vn}.png"
                target_path = os.path.join(struct_dir, target_name)
                Image.fromarray(rgb_u8).save(target_path)
                print(f"[SAVE]: {target_path}")

        has_next = step_idx < len(target)
        next_idx = int(step_idx) if has_next else None
        tw = _grid_to_world(*target[step_idx]) if has_next else None
        sw = tuple(float(v) for v in scatter[step_idx]["world"]) if has_next else None
        vmeta = {}
        for vn, e in entries.items():
            h, w = int(rgb_by_view[vn].shape[0]), int(rgb_by_view[vn].shape[1])
            tp = _proj(sim.stage, e["prim"], tw, w, h) if has_next else None
            sp = _proj(sim.stage, e["prim"], sw, w, h) if has_next else None
            tb, sb = _bbox(tp, w, h), _bbox(sp, w, h)
            ov = f"{sample_id}_step{step_idx:05d}_{vn}_gt_bbox_overlay.png"
            ov_rel = os.path.join("bbox_overlay", ov).replace("\\", "/")
            ov_path = os.path.join(bbox_dir, ov)
            Image.fromarray(_overlay(rgb_by_view[vn], tb, sb)).save(ov_path)
            print(f"[SAVE]: {ov_path}")
            vmeta[vn] = {"camera_prim_path": e["prim"], "image_size_hw": [h, w], "bbox_size_px": int(round(27 * (float(min(w, h)) / 224.0))), "target_projection": tp, "selected_projection": sp, "target_bbox_xyxy": tb, "selected_bbox_xyxy": sb, "bbox_files": {"bbox_overlay_image": ov_rel}}

        bbox = {
            "enabled": True,
            "type": "multi_view_bounding_box",
            "bbox_source": "projected_center_square",
            "bbox_min_size_px": 27,
            "bbox_base_size_px": 27,
            "bbox_auto_scaled_with_resolution": True,
            "save_bbox_images": True,
            "has_next_action": bool(has_next),
            "next_block_index_0based": next_idx,
            "next_block_id_1based": (int(next_idx) + 1) if next_idx is not None else None,
            "next_target_world_xyz": [float(x) for x in tw] if tw is not None else None,
            "selected_block_world_xyz": [float(x) for x in sw] if sw is not None else None,
            "views": vmeta,
        }

        steps.append(_collect_step_record(step_idx, target, scatter, images, views, bbox))
        print(f"[INFO]: step={step_idx:05d} placed={step_idx}/{len(target)} scattered={len(target)-step_idx}")

    struct_id = str(struct_meta["struct_id"])
    structure_target_images = {vn: os.path.join("struct", struct_id, f"{struct_id}_{vn}.png").replace("\\", "/") for vn in entries.keys()}
    _write_sample_json(out_dir, sample_id, target, cols, struct_meta, order_index, steps, structure_target_images)


def _render_struct_target_images(struct_id: str, target: list[tuple[int, int, int]], output_root: str, runtime: dict):
    struct_dir = os.path.join(output_root, "struct", struct_id)
    os.makedirs(struct_dir, exist_ok=True)
    sim = runtime["sim"]
    scene = runtime["scene"]
    sim_cfg = runtime["sim_cfg"]
    entries = runtime["entries"]
    total_blocks = int(runtime["total_blocks"])
    _reset_episode(runtime)

    _set_layout(scene, target, [{"world": _grid_to_world(*p), "quat": (1.0, 0.0, 0.0, 0.0)} for p in target], len(target))
    _hide_blocks(scene, len(target), total_blocks)
    _warmup(sim, scene, sim_cfg.dt, 12)

    for vn, e in entries.items():
        rgb = e["obj"].data.output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb_u8 = rgb.astype(np.uint8)
        target_path = os.path.join(struct_dir, f"{struct_id}_{vn}.png")
        Image.fromarray(rgb_u8).save(target_path)
        print(f"[SAVE]: {target_path}")


def main():
    if args_cli.device == "cpu":
        args_cli.device = "cuda:0"

    output_root = _root()
    struct_root = os.path.join(output_root, "struct")
    if not os.path.isdir(struct_root):
        raise FileNotFoundError(f"struct folder not found: {struct_root}. Please run step1 first.")
    structs = _discover_structs(struct_root)
    if not structs:
        raise RuntimeError(f"No struct entries found under {struct_root}")

    print(f"[INFO]: output_root={output_root}")
    print(f"[INFO]: struct_count={len(structs)}")
    print(f"[INFO]: render_count={int(args_cli.output_count)}")
    runtime = _build_runtime(args_cli.device)

    try:
        # Render target images for every structure first.
        for i, (struct_id, s_json, _) in enumerate(structs):
            canonical_target, _ = _load_struct_canonical(s_json)
            print(f"[INFO]: render_target struct={struct_id} ({i+1}/{len(structs)}) blocks={len(canonical_target)}")
            _render_struct_target_images(struct_id, canonical_target, output_root, runtime)

        rng = random.Random(42)
        for i in range(int(args_cli.output_count)):
            sample_id = f"{i + 1:05d}"
            struct_id, s_json, o_json = rng.choice(structs)
            target, struct_meta, order_index = _load_struct_and_order(s_json, o_json, rng)
            seed = 42 + i * 97
            print(f"[INFO]: sample={sample_id} struct={struct_id} order_index={order_index} blocks={len(target)}")
            _render_one(sample_id, target, struct_meta, order_index, output_root, seed, runtime)
    finally:
        sim = runtime["sim"]
        try:
            if not sim.has_gui():
                sim.stop()
        except Exception:
            pass
        try:
            sim.clear_all_callbacks()
        except Exception:
            pass
        try:
            sim.clear_instance()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass

    print("[INFO]: Finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()
