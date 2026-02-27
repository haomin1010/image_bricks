# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PinocchioPoseAction(ActionTerm):
    r"""Pose-target action term for UR10 joint targets (GPU DLS IK only).

    Input action (per env): ``[x, y, z, qw, qx, qy, qz]`` in world coordinates.
    Applied output (per env): joint position target for configured UR10 arm joints.
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

    def _compute_body_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def _target_pose_world_to_base(
        self,
        ee_pos_target_w: torch.Tensor,
        ee_quat_target_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        ee_pos_target_b, ee_quat_target_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_target_w, ee_quat_target_w
        )
        return ee_pos_target_b, ee_quat_target_b

    def _process_actions_dls_gpu(self, ee_pos_target_w: torch.Tensor, ee_quat_target_w: torch.Tensor) -> None:
        ee_pos_curr_b, ee_quat_curr_b = self._compute_body_pose_in_base()
        ee_pos_target_b, ee_quat_target_b = self._target_pose_world_to_base(ee_pos_target_w, ee_quat_target_w)
        self._ik_controller.set_command(
            torch.cat((ee_pos_target_b, ee_quat_target_b), dim=-1),
            ee_pos_curr_b,
            ee_quat_curr_b,
        )

        joint_pos_curr = self._asset.data.joint_pos[:, self._joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, self.jacobian_b, joint_pos_curr)

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
    ee_body_name: str = "ee_link"
    damping: float = 0.10
    step_gain: float = 0.70
    max_joint_delta: float = 0.08
    nullspace_gain: float = 0.02
    use_default_nominal_joint_pos: bool = True
