# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import os
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.observations import joint_pos as obs_joint_pos
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


DEFAULT_EE_BODY_NAME = os.getenv("VAGEN_IK_EE_BODY_NAME", "panda_link7") or "panda_link7"
DEFAULT_EE_FRAME_TARGET_NAME = os.getenv("VAGEN_IK_EE_FRAME_TARGET_NAME", "end_effector") or "end_effector"
USE_EE_FRAME_TCP = os.getenv("VAGEN_USE_EE_FRAME_TCP", "1").strip().lower() not in {"0", "false", "off", "no"}


class PinocchioPoseAction(ActionTerm):
    r"""Pose-target action term for arm joint targets (GPU DLS IK only).

    Input action (per env): ``[x, y, z, qw, qx, qy, qz]`` in world coordinates.
    Applied output (per env): joint position target for configured arm joints.
    """

    cfg: PinocchioPoseActionCfg
    _asset: Articulation

    def __init__(self, cfg: PinocchioPoseActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        if len(self._joint_ids) == 0:
            raise RuntimeError("PinocchioPoseAction resolved no joints; please check `joint_names`.")
        # For explicit Franka joint name lists, require full one-to-one mapping.
        regex_chars = set(r".*+?[](){}|\^$")
        has_regex = any(any(ch in str(name) for ch in regex_chars) for name in cfg.joint_names)
        if not has_regex and len(self._joint_ids) != len(cfg.joint_names):
            raise RuntimeError(
                "PinocchioPoseAction joint mapping mismatch for explicit joint list: "
                f"requested={list(cfg.joint_names)} resolved={list(self._joint_names)}"
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._ik_controller: DifferentialIKController
        self._body_idx: int
        self._body_name: str
        self._jacobi_body_idx: int
        self._jacobi_joint_ids: list[int]
        self._joint_obs_cfg: SceneEntityCfg
        self._ee_frame: FrameTransformer | None = None
        self._use_ee_frame_feedback = bool(cfg.use_ee_frame_feedback)
        self._ee_frame_target_idx = 0
        self._warned_missing_ee_frame = False
        self._warned_invalid_ee_frame = False

        self._nominal_joint_pos = None
        if cfg.use_default_nominal_joint_pos:
            self._nominal_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name, preserve_order=cfg.preserve_order)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one body match for '{cfg.ee_body_name}', got {len(body_ids)}: {body_names}."
            )
        self._body_idx = int(body_ids[0])
        self._body_name = str(body_names[0])
        self._joint_obs_cfg = SceneEntityCfg(cfg.asset_name, joint_ids=list(self._joint_ids))
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = list(self._joint_ids)
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [int(j_id + 6) for j_id in self._joint_ids]

        self._ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": float(cfg.damping)},
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._resolve_ee_frame()

        if self._use_ee_frame_feedback:
            curr_pose_source = "ee_frame" if self._ee_frame is not None else "ee_body(fallback)"
        else:
            curr_pose_source = "ee_body(disabled)"
        print(
            "[INFO]: PinocchioPoseAction initialized "
            f"backend=dls_gpu body={self._body_name} joints={self._joint_names} "
            f"curr_pose_source={curr_pose_source} "
            f"lambda={cfg.damping}"
        )

    def _resolve_ee_frame(self) -> None:
        """Resolves the configured ee_frame sensor and its target index."""
        if not self._use_ee_frame_feedback:
            self._ee_frame = None
            self._ee_frame_target_idx = 0
            return
        frame_name = str(self.cfg.ee_frame_name).strip()
        if not frame_name:
            self._ee_frame = None
            self._ee_frame_target_idx = 0
            return
        try:
            frame_obj = self._env.scene[frame_name]
        except Exception:
            self._ee_frame = None
            self._ee_frame_target_idx = 0
            if not self._warned_missing_ee_frame:
                print(
                    f"[WARN]: PinocchioPoseAction couldn't resolve ee_frame '{frame_name}'. "
                    "Falling back to ee_body feedback."
                )
                self._warned_missing_ee_frame = True
            return
        if not isinstance(frame_obj, FrameTransformer):
            self._ee_frame = None
            self._ee_frame_target_idx = 0
            if not self._warned_missing_ee_frame:
                print(
                    f"[WARN]: Scene object '{frame_name}' is not a FrameTransformer "
                    f"(got {type(frame_obj)}). Falling back to ee_body feedback."
                )
                self._warned_missing_ee_frame = True
            return

        self._ee_frame = frame_obj
        self._ee_frame_target_idx = 0
        target_name = str(self.cfg.ee_frame_target_name).strip()
        if not target_name:
            return
        try:
            target_ids, _ = self._ee_frame.find_bodies(target_name, preserve_order=True)
            if len(target_ids) > 0:
                self._ee_frame_target_idx = int(target_ids[0])
                return
        except Exception:
            pass
        if not self._warned_missing_ee_frame:
            print(
                f"[WARN]: ee_frame target '{target_name}' not found in '{frame_name}'. "
                "Using first target frame index 0."
            )
            self._warned_missing_ee_frame = True

    def _try_get_ee_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Returns ee_frame pose and validity mask, or None when unavailable."""
        if not self._use_ee_frame_feedback:
            return None
        if self._ee_frame is None:
            self._resolve_ee_frame()
        if self._ee_frame is None:
            return None
        try:
            ee_pos_w = self._ee_frame.data.target_pos_w[:, self._ee_frame_target_idx, :]
            ee_quat_w = self._ee_frame.data.target_quat_w[:, self._ee_frame_target_idx, :]
        except Exception:
            return None

        quat_norm = torch.linalg.vector_norm(ee_quat_w, dim=-1, keepdim=True)
        valid_mask = torch.isfinite(ee_pos_w).all(dim=-1)
        valid_mask = valid_mask & torch.isfinite(ee_quat_w).all(dim=-1)
        valid_mask = valid_mask & (quat_norm.squeeze(-1) > 1e-8)

        ee_quat_w = ee_quat_w / quat_norm.clamp_min(1e-8)
        if torch.any(~valid_mask) and not self._warned_invalid_ee_frame:
            print(
                "[WARN]: ee_frame returned invalid pose for some environments. "
                "Falling back to ee_body pose for invalid entries."
            )
            self._warned_invalid_ee_frame = True
        return ee_pos_w, ee_quat_w, valid_mask

    @property
    def action_dim(self) -> int:
        # [x, y, z, qw, qx, qy, qz]
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _process_actions_dls_gpu(self, ee_pos_target_w: torch.Tensor, ee_quat_target_w: torch.Tensor) -> None:
        # IMPORTANT: the ee-frame feedback may represent TCP (suction cup) position.
        # Here we need the *body* (link7) position
        # here so that tcp_offset_w = TCP_pos - link7_pos = actual suction-cup offset, which
        # is used to correct the Jacobian below.  Read directly from articulation data instead.
        ee_pos_curr_body_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_curr_body_w = self._asset.data.body_quat_w[:, self._body_idx]
        ee_quat_curr_body_w = ee_quat_curr_body_w / torch.linalg.vector_norm(
            ee_quat_curr_body_w, dim=-1, keepdim=True
        ).clamp_min(1e-8)

        ee_pos_curr_w = ee_pos_curr_body_w
        ee_quat_curr_w = ee_quat_curr_body_w
        jacobian_w = self.jacobian_w

        ee_frame_pose = self._try_get_ee_frame_pose_w()
        if ee_frame_pose is not None:
            ee_frame_pos_w, ee_frame_quat_w, ee_frame_valid = ee_frame_pose
            if bool(torch.any(ee_frame_valid)):
                valid_pos_quat = ee_frame_valid.unsqueeze(-1)
                ee_pos_curr_w = torch.where(valid_pos_quat, ee_frame_pos_w, ee_pos_curr_w)
                ee_quat_curr_w = torch.where(valid_pos_quat, ee_frame_quat_w, ee_quat_curr_w)
                jacobian_w = self.jacobian_w.clone()
                tcp_offset_w = ee_pos_curr_w - ee_pos_curr_body_w
                jacobian_trans = jacobian_w[:, :3, :] - torch.bmm(
                    math_utils.skew_symmetric_matrix(tcp_offset_w), jacobian_w[:, 3:, :]
                )
                jacobian_w[:, :3, :] = torch.where(
                    ee_frame_valid.unsqueeze(-1).unsqueeze(-1), jacobian_trans, jacobian_w[:, :3, :]
                )

        # Use shortest-arc quaternion to avoid orientation discontinuity from sign flips.
        quat_dot = torch.sum(ee_quat_target_w * ee_quat_curr_w, dim=-1, keepdim=True)
        ee_quat_target_w = torch.where(quat_dot < 0.0, -ee_quat_target_w, ee_quat_target_w)

        self._ik_controller.set_command(
            torch.cat((ee_pos_target_w, ee_quat_target_w), dim=-1),
            ee_pos_curr_w,
            ee_quat_curr_w,
        )

        joint_pos_curr = obs_joint_pos(env=self._env, asset_cfg=self._joint_obs_cfg)
        joint_pos_des = self._ik_controller.compute(ee_pos_curr_w, ee_quat_curr_w, jacobian_w, joint_pos_curr)
        q_next = joint_pos_des
        self._processed_actions[:] = q_next

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"PinocchioPoseAction expects last dim={self.action_dim}, got shape={tuple(actions.shape)}."
            )

        self._raw_actions[:] = actions

        ee_pos_target = self._raw_actions[:, :3]
        ee_quat_target = self._raw_actions[:, 3:7]
        quat_norm = torch.linalg.vector_norm(ee_quat_target, dim=-1, keepdim=True)
        ee_quat_target = ee_quat_target / quat_norm.clamp_min(1e-8)
        invalid_quat_mask = quat_norm.squeeze(-1) <= 1e-8
        if bool(torch.any(invalid_quat_mask)):
            ee_quat_curr = self._asset.data.body_quat_w[:, self._body_idx]
            ee_quat_curr = ee_quat_curr / torch.linalg.vector_norm(ee_quat_curr, dim=-1, keepdim=True).clamp_min(1e-8)
            ee_quat_target = torch.where(invalid_quat_mask.unsqueeze(-1), ee_quat_curr, ee_quat_target)

        self._process_actions_dls_gpu(ee_pos_target, ee_quat_target)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@configclass
class PinocchioPoseActionCfg(ActionTermCfg):
    """Configuration for :class:`PinocchioPoseAction`."""

    class_type: type[ActionTerm] = PinocchioPoseAction

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    preserve_order: bool = True
    ee_body_name: str = DEFAULT_EE_BODY_NAME
    use_ee_frame_feedback: bool = USE_EE_FRAME_TCP
    ee_frame_name: str = "ee_frame"
    ee_frame_target_name: str = DEFAULT_EE_FRAME_TARGET_NAME
    damping: float = 0.10
    step_gain: float = 0.70
    max_joint_delta: float = 0.08
    nullspace_gain: float = 0.02
    use_default_nominal_joint_pos: bool = True
