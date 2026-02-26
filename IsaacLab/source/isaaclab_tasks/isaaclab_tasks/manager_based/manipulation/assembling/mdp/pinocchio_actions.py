# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class _UR10PinocchioSolver:
    """Internal Pinocchio-based solver for UR10 pose-to-joint conversion."""

    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str = "ee_link",
        joint_names: list[str] | None = None,
        damping: float = 0.1,
        step_gain: float = 0.7,
        max_joint_delta: float = 0.08,
        nullspace_gain: float = 0.02,
    ):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"Pinocchio frame '{ee_frame_name}' not found in URDF: {urdf_path}")

        self.joint_names = joint_names or [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.joint_q_indices: list[int] = []
        for name in self.joint_names:
            j_id = self.model.getJointId(name)
            if j_id == 0:
                raise ValueError(f"Pinocchio joint '{name}' not found in URDF: {urdf_path}")
            try:
                idx_q_raw = self.model.joints[j_id].idx_q
                idx_q = int(idx_q_raw() if callable(idx_q_raw) else idx_q_raw)
            except Exception as exc:
                raise RuntimeError(f"Unable to resolve idx_q for joint '{name}' (id={j_id}).") from exc
            self.joint_q_indices.append(int(idx_q))

        self.damping = float(damping)
        self.step_gain = float(step_gain)
        self.max_joint_delta = float(max_joint_delta)
        self.nullspace_gain = float(nullspace_gain)
        self.lower_limits = np.asarray(self.model.lowerPositionLimit[self.joint_q_indices], dtype=np.float64)
        self.upper_limits = np.asarray(self.model.upperPositionLimit[self.joint_q_indices], dtype=np.float64)

    @staticmethod
    def default_ur10_urdf_candidates() -> list[str]:
        base = Path("/home/user/miniconda3/envs/bricks/lib/python3.11/site-packages/isaacsim/exts")
        return [
            str(base / "isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/ur10_robot_suction.urdf"),
            str(base / "isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/ur10_robot.urdf"),
            str(base / "isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf"),
        ]

    def solve_step(
        self,
        joint_pos: np.ndarray,
        ee_pos_target: np.ndarray,
        ee_quat_wxyz_target: np.ndarray,
        nominal_joint_pos: np.ndarray | None = None,
        nullspace_gain: float | None = None,
    ) -> np.ndarray:
        q = self.pin.neutral(self.model)
        for k, q_idx in enumerate(self.joint_q_indices):
            q[q_idx] = float(joint_pos[k])

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

        target_se3 = self.pin.SE3(
            self.pin.Quaternion(
                float(ee_quat_wxyz_target[0]),
                float(ee_quat_wxyz_target[1]),
                float(ee_quat_wxyz_target[2]),
                float(ee_quat_wxyz_target[3]),
            ).toRotationMatrix(),
            np.asarray(ee_pos_target, dtype=np.float64),
        )
        current = self.data.oMf[self.ee_frame_id]
        err_se3 = current.inverse() * target_se3
        err = self.pin.log6(err_se3).vector

        jac_full = self.pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, self.pin.LOCAL)
        jac = jac_full[:, self.joint_q_indices]

        jj_t = jac @ jac.T
        lam2 = (self.damping**2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(jj_t + lam2, err)

        ns_gain = self.nullspace_gain if nullspace_gain is None else float(nullspace_gain)
        if nominal_joint_pos is not None and ns_gain > 0.0:
            dq += ns_gain * (np.asarray(nominal_joint_pos) - np.asarray(joint_pos))

        dq = self.step_gain * dq
        dq = np.clip(dq, -self.max_joint_delta, self.max_joint_delta)
        q_next = np.asarray(joint_pos, dtype=np.float64) + dq

        for i in range(len(q_next)):
            cand = q_next[i]
            curr = float(joint_pos[i])
            q_next[i] = curr + ((cand - curr + np.pi) % (2.0 * np.pi) - np.pi)

        finite_lower = np.isfinite(self.lower_limits)
        finite_upper = np.isfinite(self.upper_limits)
        if np.any(finite_lower):
            q_next[finite_lower] = np.maximum(q_next[finite_lower], self.lower_limits[finite_lower])
        if np.any(finite_upper):
            q_next[finite_upper] = np.minimum(q_next[finite_upper], self.upper_limits[finite_upper])
        return q_next.astype(np.float32)


class PinocchioPoseAction(ActionTerm):
    r"""Pose-target action term that uses Pinocchio IK to generate UR10 joint targets.

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

        urdf_path = self._resolve_urdf_path(cfg.urdf_path)
        self._pin_solver = _UR10PinocchioSolver(
            urdf_path=urdf_path,
            ee_frame_name=cfg.ee_frame_name,
            joint_names=self._joint_names,
            damping=cfg.damping,
            step_gain=cfg.step_gain,
            max_joint_delta=cfg.max_joint_delta,
            nullspace_gain=cfg.nullspace_gain,
        )

        self._nominal_joint_pos = None
        if cfg.use_default_nominal_joint_pos:
            self._nominal_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        print(
            "[INFO]: PinocchioPoseAction initialized "
            f"urdf={urdf_path} joints={self._joint_names} "
            f"damping={cfg.damping} step_gain={cfg.step_gain} "
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

        joint_pos_curr = self._asset.data.joint_pos[:, self._joint_ids]

        for env_id in range(self.num_envs):
            nominal_joint_pos = None
            if self._nominal_joint_pos is not None:
                nominal_joint_pos = self._nominal_joint_pos[env_id].detach().cpu().numpy()

            q_next = self._pin_solver.solve_step(
                joint_pos=joint_pos_curr[env_id].detach().cpu().numpy(),
                ee_pos_target=ee_pos_target[env_id].detach().cpu().numpy(),
                ee_quat_wxyz_target=ee_quat_target[env_id].detach().cpu().numpy(),
                nominal_joint_pos=nominal_joint_pos,
            )
            self._processed_actions[env_id] = torch.from_numpy(q_next).to(device=self.device, dtype=torch.float32)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0

    @staticmethod
    def _resolve_urdf_path(urdf_path: str | None) -> str:
        if urdf_path is not None and Path(urdf_path).exists():
            return str(urdf_path)
        for candidate in _UR10PinocchioSolver.default_ur10_urdf_candidates():
            if Path(candidate).exists():
                return candidate
        if urdf_path is not None:
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        raise FileNotFoundError(
            "Could not resolve UR10 URDF for Pinocchio IK. "
            f"Checked candidates: {_UR10PinocchioSolver.default_ur10_urdf_candidates()}"
        )


@configclass
class PinocchioPoseActionCfg(ActionTermCfg):
    """Configuration for :class:`PinocchioPoseAction`."""

    class_type: type[ActionTerm] = PinocchioPoseAction

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    preserve_order: bool = True

    ee_frame_name: str = "ee_link"
    urdf_path: str | None = None
    damping: float = 0.10
    step_gain: float = 0.70
    max_joint_delta: float = 0.08
    nullspace_gain: float = 0.02
    use_default_nominal_joint_pos: bool = True
