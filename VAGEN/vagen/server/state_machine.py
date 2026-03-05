from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils


class StackingStateMachine:
    """Minimal freeze-mode state machine.

    Current behavior:
    - ignore all task-state transitions
    - hold end-effector at idle position (captured after reset)
    - command fixed vertical/downward quaternion
    """

    def __init__(
        self,
        num_envs,
        device,
        scene,
        cube_names,
        max_tasks=8,
        cube_z_size=0.045,
        grid_origin=[0.5, 0.0, 0.001],
        cell_size=0.056,
        grid_size=8,
    ):
        del cube_z_size, grid_size
        self.num_envs = int(num_envs)
        self.device = device
        self.scene = scene
        self.cube_names = list(cube_names)
        self.max_tasks = int(max_tasks)
        self.grid_origin = torch.tensor(grid_origin, device=device, dtype=torch.float32)
        self.cell_size = float(cell_size)

        # Fields consumed by server.py.
        self.IDLE = -2
        self.APPROACH_CUBE = 0
        self.DESCEND_CUBE = 1
        self.GRASP = 2
        self.LIFT = 3
        self.APPROACH_TARGET = 4
        self.DESCEND_TARGET = 5
        self.RELEASE = 6
        self.ASCEND_TARGET = 8
        self.ASCEND_HOME = 9
        self.DESCEND_FINAL_Z_OFFSET = 0.06
        self.GRASP_X_BIAS = 0.025
        self.GRASP_SETTLE_STEPS = 8
        self.LIFT_STEPS = 40
        self.APPROACH_TARGET_STEPS = 40
        self.state = torch.full((self.num_envs,), self.IDLE, dtype=torch.long, device=device)
        self.task_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.target_positions = torch.zeros((self.num_envs, self.max_tasks, 3), device=device)
        self.num_tasks_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.new_task_available = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.new_task_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)

        # Freeze pose state.
        self.idle_pos = torch.zeros((self.num_envs, 3), device=device)
        self.idle_quat = torch.zeros((self.num_envs, 4), device=device)
        self.idle_pos_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        # Per-env frozen world target during GRASP, so gripper does not move while closing.
        self.grasp_hold_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.grasp_hold_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.release_hold_pos_w = torch.zeros((self.num_envs, 3), device=device)

    def set_env_targets(self, env_ids, targets):
        """No-op in freeze mode; kept only for API compatibility."""
        del env_ids, targets
        return None

    def reset_envs(self, env_ids):
        self.state[env_ids] = self.IDLE
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
        self.new_task_available[env_ids] = False
        self.new_task_index[env_ids] = -1
        self.idle_pos_initialized[env_ids] = False
        self.grasp_hold_valid[env_ids] = False
        self.release_hold_pos_w[env_ids] = 0.0

    def reset_all(self):
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def reset_state(self, env_idx):
        self.state_timer[env_idx] = 0

    def _extract_policy_obs(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            raise TypeError(f"Invalid observation container: {type(obs)}")
        policy_obs = obs.get("policy", obs)
        if not isinstance(policy_obs, dict):
            raise TypeError(f"Invalid policy observation container: {type(policy_obs)}")
        return policy_obs

    def _capture_idle_pose_once(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> None:
        new_idle_envs = ~self.idle_pos_initialized
        if not torch.any(new_idle_envs):
            return
        num_new = int(new_idle_envs.sum().item())
        # Idle hold pose defined in WORLD frame.
        target_pos_w = torch.zeros((num_new, 3), device=self.device, dtype=root_pos_w.dtype)
        target_pos_w[:, 0] = 0.5
        target_pos_w[:, 1] = 0.0
        target_pos_w[:, 2] = 0.45
        target_quat_w = torch.zeros((num_new, 4), device=self.device, dtype=root_quat_w.dtype)
        target_quat_w[:, 1] = 1.0
        target_pos_root, target_quat_root = math_utils.subtract_frame_transforms(
            root_pos_w[new_idle_envs],
            root_quat_w[new_idle_envs],
            target_pos_w,
            target_quat_w,
        )
        self.idle_pos[new_idle_envs] = target_pos_root
        self.idle_quat[new_idle_envs] = target_quat_root
        self.idle_pos_initialized[new_idle_envs] = True
        for env_idx in torch.where(new_idle_envs)[0]:
            idx = int(env_idx.item())
            print(
                f"[INIT] env={idx} "
                f"idle_quat={self.idle_quat[idx].tolist()} "
                f"idle_pos={self.idle_pos[idx].tolist()}"
            )

    def compute_ee_pose_targets(self, obs):
        policy_obs = self._extract_policy_obs(obs)
        ee_pos_env = policy_obs.get("ee_pos", None)
        ee_quat_w = policy_obs.get("ee_quat", None)
        cube_pos_w = policy_obs.get("cube_pos", None)
        gripper_pos_obs = policy_obs.get("gripper_pos", None)
        gripper_closed_obs = policy_obs.get("gripper_closed", None)
        cube_grasped_obs = policy_obs.get("cubegrasped", None)
        if cube_grasped_obs is None:
            cube_grasped_obs = policy_obs.get("grasped", None)
        root_pos_w = policy_obs.get("root_pos", None)
        root_quat_w = policy_obs.get("root_quat", None)
        env_origin = policy_obs.get("env_origin", None)
        if (
            ee_pos_env is None
            or ee_quat_w is None
            or cube_pos_w is None
            or root_pos_w is None
            or root_quat_w is None
            or env_origin is None
        ):
            raise KeyError(
                "Missing required policy obs keys: "
                "ee_pos/ee_quat/cube_pos/root_pos/root_quat/env_origin"
            )

        # abs-IK action expects pose in robot root frame. Convert observed ee pose first.
        ee_pos_w = ee_pos_env + env_origin
        ee_pos_root, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        self._capture_idle_pose_once(root_pos_w, root_quat_w)

        # Default hold target in WORLD frame.
        target_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_pos_w.dtype)
        target_pos_w[:, 0] = 0.5
        target_pos_w[:, 1] = 0.0
        target_pos_w[:, 2] = 0.45
        target_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=root_quat_w.dtype)
        target_quat_w[:, 1] = 1.0

        # Active cube-driven states.
        if cube_pos_w.ndim != 2 or cube_pos_w.shape[1] < 3 or (cube_pos_w.shape[1] % 3 != 0):
            raise ValueError(f"Invalid cube_pos shape: {tuple(cube_pos_w.shape)}")
        cube_pos_w_reshaped = cube_pos_w.view(self.num_envs, -1, 3)
        num_obs_cubes = int(cube_pos_w_reshaped.shape[1])

        approach_mask = self.state == self.APPROACH_CUBE
        descend_mask = self.state == self.DESCEND_CUBE
        grasp_mask = self.state == self.GRASP
        active_cube_mask = torch.logical_or(torch.logical_or(approach_mask, descend_mask), grasp_mask)

        if torch.any(active_cube_mask):
            cube_slot_all = self.task_index.to(dtype=torch.long)
            invalid_slot_mask = torch.logical_or(cube_slot_all < 0, cube_slot_all >= num_obs_cubes)
            bad_env_mask = torch.logical_and(active_cube_mask, invalid_slot_mask)
            if torch.any(bad_env_mask):
                bad_env_ids = torch.where(bad_env_mask)[0].tolist()
                bad_slots = cube_slot_all[bad_env_mask].tolist()
                raise IndexError(
                    f"CUBE state requested invalid cube slot(s): env_ids={bad_env_ids} "
                    f"slots={bad_slots} num_obs_cubes={num_obs_cubes}"
                )

            active_env_ids = torch.where(active_cube_mask)[0]
            active_cube_slots = cube_slot_all[active_env_ids]
            active_cube_pos_w = cube_pos_w_reshaped[active_env_ids, active_cube_slots]

            # Both APPROACH and DESCEND use cube XY.
            target_pos_w[active_env_ids, 0:2] = active_cube_pos_w[:, 0:2]

            # DESCEND_CUBE: keep XY and descend near cube with safety clearance.
            descend_env_ids = torch.where(descend_mask)[0]
            if descend_env_ids.numel() > 0:
                descend_cube_slots = cube_slot_all[descend_env_ids]
                descend_cube_pos_w = cube_pos_w_reshaped[descend_env_ids, descend_cube_slots]
                # Apply the same X bias in DESCEND as GRASP to keep trajectory consistent.
                target_pos_w[descend_env_ids, 0] = descend_cube_pos_w[:, 0] + self.GRASP_X_BIAS
                # Hard-coded staged descend for stability:
                # phase-0: +0.12, phase-1: +0.09, phase-2: +0.075, phase-3: +0.06
                # each phase holds for 12 control steps.
                descend_timer = self.state_timer[descend_env_ids]
                descend_offset = torch.full_like(descend_cube_pos_w[:, 2], self.DESCEND_FINAL_Z_OFFSET)
                descend_offset = torch.where(descend_timer < 12, 0.12, descend_offset)
                descend_offset = torch.where(
                    torch.logical_and(descend_timer >= 12, descend_timer < 24), 0.09, descend_offset
                )
                descend_offset = torch.where(
                    torch.logical_and(descend_timer >= 24, descend_timer < 36), 0.075, descend_offset
                )
                descend_target_z = descend_cube_pos_w[:, 2] + descend_offset
                target_pos_w[descend_env_ids, 2] = descend_target_z

                # DESCEND_CUBE -> GRASP only after final stage and Z aligned.
                final_stage_mask = descend_timer >= 36
                to_grasp_mask = final_stage_mask
                if torch.any(to_grasp_mask):
                    to_grasp_env_ids = descend_env_ids[to_grasp_mask]
                    to_grasp_cube_pos_w = descend_cube_pos_w[to_grasp_mask]
                    self.state[to_grasp_env_ids] = self.GRASP
                    self.state_timer[to_grasp_env_ids] = 0
                    # GRASP hold pose uses fixed encoded target at DESCEND terminal target.
                    self.grasp_hold_pos_w[to_grasp_env_ids, 0] = to_grasp_cube_pos_w[:, 0]
                    self.grasp_hold_pos_w[to_grasp_env_ids, 1] = to_grasp_cube_pos_w[:, 1]
                    self.grasp_hold_pos_w[to_grasp_env_ids, 2] = (
                        to_grasp_cube_pos_w[:, 2] + self.DESCEND_FINAL_Z_OFFSET + 0.045
                    )
                    self.grasp_hold_valid[to_grasp_env_ids] = True

            # GRASP: hold current world pose and only close gripper (no movement).
            grasp_env_ids = torch.where(grasp_mask)[0]
            if grasp_env_ids.numel() > 0:
                target_pos_w[grasp_env_ids] = self.grasp_hold_pos_w[grasp_env_ids]

            # APPROACH_CUBE -> DESCEND_CUBE transition when XY is aligned.
            approach_env_ids = torch.where(approach_mask)[0]
            if approach_env_ids.numel() > 0:
                approach_cube_slots = cube_slot_all[approach_env_ids]
                approach_cube_pos_w = cube_pos_w_reshaped[approach_env_ids, approach_cube_slots]
                dist_xy = torch.linalg.vector_norm(ee_pos_w[approach_env_ids, 0:2] - approach_cube_pos_w[:, 0:2], dim=1)
                switch_mask = dist_xy < 0.01
                if torch.any(switch_mask):
                    switch_env_ids = approach_env_ids[switch_mask]
                    self.state[switch_env_ids] = self.DESCEND_CUBE
                    self.state_timer[switch_env_ids] = 0

        # LIFT: smooth arithmetic interpolation to cube XY with carry height Z=0.35.
        lift_mask = self.state == self.LIFT
        if torch.any(lift_mask):
            lift_env_ids = torch.where(lift_mask)[0]
            lift_cube_slots = torch.clamp(self.task_index[lift_env_ids].to(torch.long), 0, num_obs_cubes - 1)
            lift_cube_pos_w = cube_pos_w_reshaped[lift_env_ids, lift_cube_slots]
            lift_start_pos_w = self.grasp_hold_pos_w[lift_env_ids]
            lift_timer = self.state_timer[lift_env_ids].to(dtype=lift_start_pos_w.dtype)
            lift_alpha = torch.clamp((lift_timer + 1.0) / float(self.LIFT_STEPS), min=0.0, max=1.0)
            target_pos_w[lift_env_ids, 0:2] = lift_start_pos_w[:, 0:2] + (
                lift_cube_pos_w[:, 0:2] - lift_start_pos_w[:, 0:2]
            ) * lift_alpha.unsqueeze(-1)
            target_pos_w[lift_env_ids, 2] = lift_start_pos_w[:, 2] + (0.45 - lift_start_pos_w[:, 2]) * lift_alpha
            to_approach_target_mask = self.state_timer[lift_env_ids] >= self.LIFT_STEPS
            if torch.any(to_approach_target_mask):
                to_approach_target_env_ids = lift_env_ids[to_approach_target_mask]
                self.state[to_approach_target_env_ids] = self.APPROACH_TARGET
                self.state_timer[to_approach_target_env_ids] = 0

        # APPROACH_TARGET: smooth arithmetic interpolation to final goal XY at carry height Z=0.45.
        approach_target_mask = self.state == self.APPROACH_TARGET
        if torch.any(approach_target_mask):
            approach_target_env_ids = torch.where(approach_target_mask)[0]
            target_slot = torch.clamp(self.task_index[approach_target_env_ids].to(torch.long), 0, self.max_tasks - 1)
            goal_pos_w = self.target_positions[approach_target_env_ids, target_slot]
            approach_cube_slots = torch.clamp(self.task_index[approach_target_env_ids].to(torch.long), 0, num_obs_cubes - 1)
            approach_start_cube_pos_w = cube_pos_w_reshaped[approach_target_env_ids, approach_cube_slots]
            approach_timer = self.state_timer[approach_target_env_ids].to(dtype=goal_pos_w.dtype)
            approach_alpha = torch.clamp((approach_timer + 1.0) / float(self.APPROACH_TARGET_STEPS), min=0.0, max=1.0)
            target_pos_w[approach_target_env_ids, 0:2] = approach_start_cube_pos_w[:, 0:2] + (
                goal_pos_w[:, 0:2] - approach_start_cube_pos_w[:, 0:2]
            ) * approach_alpha.unsqueeze(-1)
            target_pos_w[approach_target_env_ids, 2] = 0.45
            to_descend_target_mask = self.state_timer[approach_target_env_ids] >= self.APPROACH_TARGET_STEPS
            if torch.any(to_descend_target_mask):
                to_descend_target_env_ids = approach_target_env_ids[to_descend_target_mask]
                self.state[to_descend_target_env_ids] = self.DESCEND_TARGET
                self.state_timer[to_descend_target_env_ids] = 0

        # DESCEND_TARGET: per-step smooth descend, 1 cm each control step.
        descend_target_mask = self.state == self.DESCEND_TARGET
        if torch.any(descend_target_mask):
            descend_target_env_ids = torch.where(descend_target_mask)[0]
            target_slot = torch.clamp(self.task_index[descend_target_env_ids].to(torch.long), 0, self.max_tasks - 1)
            goal_pos_w = self.target_positions[descend_target_env_ids, target_slot]
            target_pos_w[descend_target_env_ids, 0:2] = goal_pos_w[:, 0:2]
            target_pos_w[descend_target_env_ids, 0] = goal_pos_w[:, 0]
            descend_timer = self.state_timer[descend_target_env_ids]
            descend_target_z = 0.45 - 0.01 * descend_timer.to(dtype=goal_pos_w.dtype)
            descend_target_z = torch.clamp(descend_target_z, min=0.07)
            target_pos_w[descend_target_env_ids, 2] = descend_target_z

            final_stage_mask = descend_timer >= 38
            # No reach check: enter release right after staged descend finishes.
            to_release_mask = final_stage_mask
            if torch.any(to_release_mask):
                to_release_env_ids = descend_target_env_ids[to_release_mask]
                self.state[to_release_env_ids] = self.RELEASE
                self.state_timer[to_release_env_ids] = 0
                self.release_hold_pos_w[to_release_env_ids] = ee_pos_w[to_release_env_ids]

        # RELEASE: hold terminal pose and open gripper.
        release_mask = self.state == self.RELEASE
        if torch.any(release_mask):
            release_env_ids = torch.where(release_mask)[0]
            target_pos_w[release_env_ids] = self.release_hold_pos_w[release_env_ids]
            target_pos_w[release_env_ids, 2] = 0.07
            # After opening for a short while, start reverse-lift from target.
            to_ascend_target_mask = self.state_timer[release_env_ids] >= 20
            if torch.any(to_ascend_target_mask):
                to_ascend_target_env_ids = release_env_ids[to_ascend_target_mask]
                self.state[to_ascend_target_env_ids] = self.ASCEND_TARGET
                self.state_timer[to_ascend_target_env_ids] = 0

        # ASCEND_TARGET: reverse staged motion of DESCEND_TARGET at the same XY.
        ascend_target_mask = self.state == self.ASCEND_TARGET
        if torch.any(ascend_target_mask):
            ascend_target_env_ids = torch.where(ascend_target_mask)[0]
            target_slot = torch.clamp(self.task_index[ascend_target_env_ids].to(torch.long), 0, self.max_tasks - 1)
            goal_pos_w = self.target_positions[ascend_target_env_ids, target_slot]
            target_pos_w[ascend_target_env_ids, 0:2] = goal_pos_w[:, 0:2]
            ascend_timer = self.state_timer[ascend_target_env_ids]
            ascend_target_z = 0.07 + 0.01 * ascend_timer.to(dtype=goal_pos_w.dtype)
            ascend_target_z = torch.clamp(ascend_target_z, max=0.45)
            target_pos_w[ascend_target_env_ids, 2] = ascend_target_z

            to_ascend_home_mask = ascend_timer >= 38
            if torch.any(to_ascend_home_mask):
                to_ascend_home_env_ids = ascend_target_env_ids[to_ascend_home_mask]
                self.state[to_ascend_home_env_ids] = self.ASCEND_HOME
                self.state_timer[to_ascend_home_env_ids] = 0

        # ASCEND_HOME: keep ascent height and move horizontally back to initial XY.
        ascend_home_mask = self.state == self.ASCEND_HOME
        if torch.any(ascend_home_mask):
            ascend_home_env_ids = torch.where(ascend_home_mask)[0]
            target_pos_w[ascend_home_env_ids, 0] = 0.5
            target_pos_w[ascend_home_env_ids, 1] = 0.0
            target_pos_w[ascend_home_env_ids, 2] = 0.45
            home_xy_dist = torch.linalg.vector_norm(
                ee_pos_w[ascend_home_env_ids, 0:2] - target_pos_w[ascend_home_env_ids, 0:2], dim=1
            )
            done_home_mask = home_xy_dist <= 0.03
            if torch.any(done_home_mask):
                done_home_env_ids = ascend_home_env_ids[done_home_mask]
                self.state[done_home_env_ids] = self.IDLE
                self.state_timer[done_home_env_ids] = 0
                self.task_index[done_home_env_ids] += 1
                self.new_task_available[done_home_env_ids] = False
                self.new_task_index[done_home_env_ids] = -1

        # Hard-code vertical-down quaternion for all active stages.
        active_stage_mask = self.state != self.IDLE
        if torch.any(active_stage_mask):
            target_quat_w[active_stage_mask] = 0.0
            target_quat_w[active_stage_mask, 1] = 1.0

        target_pos_all, target_quat_all = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        gripper_cmd_all = torch.ones((self.num_envs,), device=self.device)

        # Build a task-index-aware grasp signal from observations.
        env_ids_all = torch.arange(self.num_envs, device=self.device)
        cube_slot_safe = torch.clamp(self.task_index.to(dtype=torch.long), min=0, max=max(0, num_obs_cubes - 1))
        selected_cube_pos_w = cube_pos_w_reshaped[env_ids_all, cube_slot_safe]
        cube_near_mask = torch.linalg.vector_norm(selected_cube_pos_w - ee_pos_w, dim=1) < 0.07

        if gripper_pos_obs is not None:
            if gripper_pos_obs.ndim == 1:
                gripper_pos_obs = gripper_pos_obs.unsqueeze(1)
            mean_opening = torch.mean(torch.abs(gripper_pos_obs), dim=1)
            gripper_engaged_mask = mean_opening < 0.035
        elif gripper_closed_obs is not None:
            if gripper_closed_obs.ndim > 1:
                gripper_closed_obs = gripper_closed_obs.reshape(self.num_envs, -1)[:, 0]
            gripper_engaged_mask = (
                gripper_closed_obs if gripper_closed_obs.dtype == torch.bool else (gripper_closed_obs > 0.5)
            )
        else:
            gripper_engaged_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        cube_grasped_mask = torch.logical_and(cube_near_mask, gripper_engaged_mask)

        # Optional debug view of raw task-provided signal (can be stale if fixed to cube_1).
        raw_grasped_mask = None
        if cube_grasped_obs is not None:
            if cube_grasped_obs.ndim > 1:
                cube_grasped_obs = cube_grasped_obs.reshape(self.num_envs, -1)[:, 0]
            raw_grasped_mask = cube_grasped_obs if cube_grasped_obs.dtype == torch.bool else (cube_grasped_obs > 0.5)

        # GRASP/LIFT/APPROACH_TARGET/DESCEND_TARGET state keeps gripper closed.
        grasp_mask = self.state == self.GRASP
        lift_mask = self.state == self.LIFT
        approach_target_mask = self.state == self.APPROACH_TARGET
        descend_target_mask = self.state == self.DESCEND_TARGET
        close_mask = torch.logical_or(
            torch.logical_or(torch.logical_or(grasp_mask, lift_mask), approach_target_mask),
            descend_target_mask,
        )
        if torch.any(close_mask):
            gripper_cmd_all[close_mask] = -1.0

        # GRASP -> LIFT on grasp confirmation from observation.
        if torch.any(grasp_mask):
            grasp_wait_done_mask = self.state_timer >= self.GRASP_SETTLE_STEPS
            grasp_ready_mask = torch.logical_and(grasp_mask, grasp_wait_done_mask)
            grasp_confirmed_mask = torch.logical_and(grasp_ready_mask, cube_grasped_mask)
            if torch.any(grasp_confirmed_mask):
                self.state[grasp_confirmed_mask] = self.LIFT
                self.state_timer[grasp_confirmed_mask] = 0
                self.grasp_hold_valid[grasp_confirmed_mask] = False

        self.state_timer += 1

        for env_idx in range(self.num_envs):
            print(
                f"[DBG HOLD env={env_idx}] "
                f"state={int(self.state[env_idx].item())} "
                f"cube_slot={int(self.task_index[env_idx].item())} "
                f"cube_grasped={bool(cube_grasped_mask[env_idx].item())} "
                f"raw_grasped={(bool(raw_grasped_mask[env_idx].item()) if raw_grasped_mask is not None else None)} "
                f"ee_pos_env={ee_pos_env[env_idx].tolist()} "
                f"ee_pos_root={ee_pos_root[env_idx].tolist()} "
                f"target_w={target_pos_w[env_idx].tolist()} "
                f"target_root={target_pos_all[env_idx].tolist()}"
            )

        return target_pos_all, target_quat_all, gripper_cmd_all
