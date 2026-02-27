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

from .observations import ee_pos as obs_ee_pos
from .observations import ee_quat as obs_ee_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


DEFAULT_EE_BODY_NAME = os.getenv("VAGEN_IK_EE_BODY_NAME", "panda_link7") or "panda_link7"


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

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._ik_controller: DifferentialIKController
        self._body_idx: int
        self._body_name: str
        self._jacobi_body_idx: int
        self._jacobi_joint_ids: list[int]
        self._robot_cfg = SceneEntityCfg(cfg.asset_name)
        self._joint_obs_cfg: SceneEntityCfg
        self._ee_frame: FrameTransformer | None = None
        self._use_ee_frame_feedback = bool(cfg.use_ee_frame_feedback)

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
        print(
            "[INFO]: PinocchioPoseAction initialized "
            f"backend=dls_gpu body={self._body_name} joints={self._joint_names} "
            f"curr_pose_source={'ee_frame' if self._use_ee_frame_feedback else 'ee_pos'} "
            f"lambda={cfg.damping} step_gain={cfg.step_gain} "
            f"max_joint_delta={cfg.max_joint_delta} nullspace_gain={cfg.nullspace_gain}"
        )

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
        ee_pos_curr_body_w = obs_ee_pos(env=self._env, robot_cfg=self._robot_cfg, ee_body_name=self.cfg.ee_body_name)
        ee_quat_curr_body_w = obs_ee_quat(env=self._env, robot_cfg=self._robot_cfg, ee_body_name=self.cfg.ee_body_name)

        if self._use_ee_frame_feedback and self._ee_frame is not None:
            ee_pos_curr_w = self._ee_frame.data.target_pos_w[:, 0, :]
            ee_quat_curr_w = self._ee_frame.data.target_quat_w[:, 0, :]
            jacobian_w = self.jacobian_w.clone()
            tcp_offset_w = ee_pos_curr_w - ee_pos_curr_body_w
            jacobian_w[:, :3, :] = jacobian_w[:, :3, :] - torch.bmm(
                math_utils.skew_symmetric_matrix(tcp_offset_w), jacobian_w[:, 3:, :]
            )
        else:
            ee_pos_curr_w = ee_pos_curr_body_w
            ee_quat_curr_w = ee_quat_curr_body_w
            jacobian_w = self.jacobian_w

        self._ik_controller.set_command(
            torch.cat((ee_pos_target_w, ee_quat_target_w), dim=-1),
            ee_pos_curr_w,
            ee_quat_curr_w,
        )

        joint_pos_curr = obs_joint_pos(env=self._env, asset_cfg=self._joint_obs_cfg)
        joint_pos_des = self._ik_controller.compute(ee_pos_curr_w, ee_quat_curr_w, jacobian_w, joint_pos_curr)

        # Preserve legacy stability behavior from Pinocchio path.
        delta_joint_pos = joint_pos_des - joint_pos_curr
        delta_joint_pos = torch.clamp(delta_joint_pos, -float(self.cfg.max_joint_delta), float(self.cfg.max_joint_delta))
        delta_joint_pos = float(self.cfg.step_gain) * delta_joint_pos
        q_next = joint_pos_curr + delta_joint_pos

        if self._nominal_joint_pos is not None and float(self.cfg.nullspace_gain) > 0.0:
            q_next = q_next + float(self.cfg.nullspace_gain) * (self._nominal_joint_pos - joint_pos_curr)

        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        q_next = torch.clamp(q_next, min=joint_limits[..., 0], max=joint_limits[..., 1])
        self._processed_actions[:] = q_next

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"PinocchioPoseAction expects last dim={self.action_dim}, got shape={tuple(actions.shape)}."
            )

        self._raw_actions[:] = actions

        ee_pos_target = self._raw_actions[:, :3]
        ee_quat_target = self._raw_actions[:, 3:7]
        # Guard against non-normalized quaternions from upstream policies.
        quat_norm = torch.linalg.vector_norm(ee_quat_target, dim=-1, keepdim=True).clamp_min(1e-8)
        ee_quat_target = ee_quat_target / quat_norm

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
    use_ee_frame_feedback: bool = True
    ee_frame_name: str = "ee_frame"
    damping: float = 0.10
    step_gain: float = 0.70
    max_joint_delta: float = 0.08
    nullspace_gain: float = 0.02
    use_default_nominal_joint_pos: bool = True
