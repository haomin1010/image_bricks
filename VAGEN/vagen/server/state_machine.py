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
        cube_z_size=0.0203 * 2.0,
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
        self.state = torch.full((self.num_envs,), self.IDLE, dtype=torch.long, device=device)
        self.task_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.target_positions = torch.zeros((self.num_envs, self.max_tasks, 3), device=device)
        self.num_tasks_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.new_task_available = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.new_task_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)

        # Per-env frozen world target during GRASP, so gripper does not move while closing.
        self.grasp_hold_pos_w = torch.zeros((self.num_envs, 3), device=device)
        self.grasp_hold_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.release_hold_pos_w = torch.zeros((self.num_envs, 3), device=device)

    def set_env_targets(self, env_ids, targets):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        targets = torch.as_tensor(targets, device=self.device, dtype=self.target_positions.dtype).reshape(-1, 3)
        n = min(env_ids.numel(), targets.shape[0])
        env_ids = env_ids[:n]
        targets = targets[:n]
        task_slots = torch.clamp(self.task_index[env_ids].to(torch.long), 0, self.max_tasks - 1)
        self.target_positions[env_ids, task_slots] = targets
        self.num_tasks_per_env[env_ids] = torch.maximum(self.num_tasks_per_env[env_ids], task_slots + 1)
        self.state[env_ids] = self.APPROACH_CUBE
        self.state_timer[env_ids] = 0
        self.new_task_available[env_ids] = True
        self.new_task_index[env_ids] = task_slots
        return None

    def reset_envs(self, env_ids):
        self.state[env_ids] = self.IDLE
        self.task_index[env_ids] = 0
        self.state_timer[env_ids] = 0
        self.new_task_available[env_ids] = False
        self.new_task_index[env_ids] = -1
        self.grasp_hold_valid[env_ids] = False
        self.release_hold_pos_w[env_ids] = 0.0

    def reset_all(self):
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def reset_state(self, env_idx):
        self.state_timer[env_idx] = 0

    def compute_ee_pose_targets(self, obs):
        policy_obs = obs.get("policy", obs)
        
        ee_pos_env = policy_obs.get("ee_pos")
        ee_quat_w = policy_obs.get("ee_quat")
        cube_pos_w = policy_obs.get("cube_pos")
        gripper_pos_obs = policy_obs.get("gripper_pos")
        gripper_closed_obs = policy_obs.get("gripper_closed")
        cube_grasped_obs = policy_obs.get("cubegrasped")
        root_pos_w = policy_obs.get("root_pos")
        root_quat_w = policy_obs.get("root_quat")
        env_origin = policy_obs.get("env_origin")

        # abs-IK action expects pose in robot root frame. Convert observed ee pose first.
        ee_pos_w = ee_pos_env + env_origin

        # Default hold target in WORLD frame.
        target_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_pos_w.dtype)
        target_pos_w[:, 0] = 0.5
        target_pos_w[:, 1] = 0.0
        target_pos_w[:, 2] = 0.30
        target_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=root_quat_w.dtype)
        target_quat_w[:, 1] = 1.0
        gripper_cmd_all = torch.ones((self.num_envs,), device=self.device)

        # Active cube-driven states.
        if cube_pos_w.ndim != 2 or cube_pos_w.shape[1] < 3 or (cube_pos_w.shape[1] % 3 != 0):
            raise ValueError(f"Invalid cube_pos shape: {tuple(cube_pos_w.shape)}")
        cube_pos_w_reshaped = cube_pos_w.view(self.num_envs, -1, 3)
        num_obs_cubes = int(cube_pos_w_reshaped.shape[1])

        cube_slot_all = self.task_index.to(dtype=torch.long)

        # APPROACH_CUBE branch.
        approach_mask = self.state == self.APPROACH_CUBE
        approach_env_ids = torch.where(approach_mask)[0]
        if approach_env_ids.numel() > 0:
            approach_cube_slots = cube_slot_all[approach_env_ids]
            invalid_approach_slots = torch.logical_or(approach_cube_slots < 0, approach_cube_slots >= num_obs_cubes)
            if torch.any(invalid_approach_slots):
                bad_env_ids = approach_env_ids[invalid_approach_slots].tolist()
                bad_slots = approach_cube_slots[invalid_approach_slots].tolist()
                raise IndexError(
                    f"APPROACH_CUBE invalid cube slot(s): env_ids={bad_env_ids} "
                    f"slots={bad_slots} num_obs_cubes={num_obs_cubes}"
                )
            approach_cube_pos_w = cube_pos_w_reshaped[approach_env_ids, approach_cube_slots]
            approach_timer = self.state_timer[approach_env_ids].to(dtype=approach_cube_pos_w.dtype)
            approach_alpha = torch.clamp((approach_timer + 1.0) / float(40), min=0.0, max=1.0)
            target_pos_w[approach_env_ids, 0] = 0.5 + (approach_cube_pos_w[:, 0] - 0.5) * approach_alpha
            target_pos_w[approach_env_ids, 1] = 0.0 + approach_cube_pos_w[:, 1] * approach_alpha
            target_pos_w[approach_env_ids, 2] = 0.30 + (0.15 - 0.30) * approach_alpha
            to_descend_cube_mask = self.state_timer[approach_env_ids] >= 40
            if torch.any(to_descend_cube_mask):
                to_descend_cube_env_ids = approach_env_ids[to_descend_cube_mask]
                self.state[to_descend_cube_env_ids] = self.DESCEND_CUBE
                self.state_timer[to_descend_cube_env_ids] = 0

        # DESCEND_CUBE branch.
        descend_mask = self.state == self.DESCEND_CUBE
        descend_env_ids = torch.where(descend_mask)[0]
        if descend_env_ids.numel() > 0:
            descend_cube_slots = cube_slot_all[descend_env_ids]
            invalid_descend_slots = torch.logical_or(descend_cube_slots < 0, descend_cube_slots >= num_obs_cubes)
            if torch.any(invalid_descend_slots):
                bad_env_ids = descend_env_ids[invalid_descend_slots].tolist()
                bad_slots = descend_cube_slots[invalid_descend_slots].tolist()
                raise IndexError(
                    f"DESCEND_CUBE invalid cube slot(s): env_ids={bad_env_ids} "
                    f"slots={bad_slots} num_obs_cubes={num_obs_cubes}"
                )
            descend_cube_pos_w = cube_pos_w_reshaped[descend_env_ids, descend_cube_slots]
            target_pos_w[descend_env_ids, 0:2] = descend_cube_pos_w[:, 0:2]
            target_pos_w[descend_env_ids, 0] = descend_cube_pos_w[:, 0] 

            descend_timer = self.state_timer[descend_env_ids].to(dtype=descend_cube_pos_w.dtype)
            descend_alpha = torch.clamp((descend_timer + 1.0) / float(36), min=0.0, max=1.0)
            descend_target_z = descend_cube_pos_w[:, 2]
            target_pos_w[descend_env_ids, 2] = 0.30 + (descend_target_z - 0.30) * descend_alpha

            to_grasp_mask = self.state_timer[descend_env_ids] >= 36
            if torch.any(to_grasp_mask):
                to_grasp_env_ids = descend_env_ids[to_grasp_mask]
                to_grasp_cube_pos_w = descend_cube_pos_w[to_grasp_mask]
                self.state[to_grasp_env_ids] = self.GRASP
                self.state_timer[to_grasp_env_ids] = 0
                self.grasp_hold_pos_w[to_grasp_env_ids, 0] = to_grasp_cube_pos_w[:, 0]
                self.grasp_hold_pos_w[to_grasp_env_ids, 1] = to_grasp_cube_pos_w[:, 1]
                self.grasp_hold_pos_w[to_grasp_env_ids, 2] = (
                    to_grasp_cube_pos_w[:, 2]
                )
                self.grasp_hold_valid[to_grasp_env_ids] = True

        # GRASP branch.
        grasp_mask = self.state == self.GRASP
        grasp_env_ids = torch.where(grasp_mask)[0]
        if grasp_env_ids.numel() > 0:
            grasp_timer = self.state_timer[grasp_env_ids].to(dtype=target_pos_w.dtype)
            target_pos_w[grasp_env_ids] = self.grasp_hold_pos_w[grasp_env_ids]
            # Keep a tiny lift during closing to avoid pressing hard on the cube.
            target_pos_w[grasp_env_ids, 2] = self.grasp_hold_pos_w[grasp_env_ids, 2] + 0.003
            # Gentle close ramp: avoid instant full closing impulse.
            close_alpha = torch.clamp(grasp_timer / float(12), min=0.0, max=1.0)
            gripper_cmd_all[grasp_env_ids] = -1.0 * close_alpha
            grasp_cube_slots = torch.clamp(self.task_index[grasp_env_ids].to(torch.long), min=0, max=max(0, num_obs_cubes - 1))
            grasp_cube_pos_w = cube_pos_w_reshaped[grasp_env_ids, grasp_cube_slots]
            grasp_cube_near_mask = (
                torch.linalg.vector_norm(grasp_cube_pos_w[:, 0:2] - ee_pos_w[grasp_env_ids, 0:2], dim=1) < 0.01
            )

            if gripper_pos_obs is not None:
                grasp_gripper_pos = gripper_pos_obs
                if grasp_gripper_pos.ndim == 1:
                    grasp_gripper_pos = grasp_gripper_pos.unsqueeze(1)
                grasp_gripper_engaged_mask = torch.mean(torch.abs(grasp_gripper_pos[grasp_env_ids]), dim=1) < 0.035
            elif gripper_closed_obs is not None:
                grasp_gripper_closed = gripper_closed_obs
                if grasp_gripper_closed.ndim > 1:
                    grasp_gripper_closed = grasp_gripper_closed.reshape(self.num_envs, -1)[:, 0]
                grasp_gripper_engaged_all = (
                    grasp_gripper_closed
                    if grasp_gripper_closed.dtype == torch.bool
                    else (grasp_gripper_closed > 0.5)
                )
                grasp_gripper_engaged_mask = grasp_gripper_engaged_all[grasp_env_ids]
            else:
                grasp_gripper_engaged_mask = torch.zeros((grasp_env_ids.numel(),), dtype=torch.bool, device=self.device)

            # GRASP -> LIFT on grasp confirmation from observation.
            grasp_wait_done_mask = self.state_timer[grasp_env_ids] >= 12
            grasp_confirmed_mask = torch.logical_and(
                torch.logical_and(grasp_wait_done_mask, grasp_cube_near_mask), grasp_gripper_engaged_mask
            )
            if torch.any(grasp_confirmed_mask):
                grasp_confirmed_env_ids = grasp_env_ids[grasp_confirmed_mask]
                self.state[grasp_confirmed_env_ids] = self.LIFT
                self.state_timer[grasp_confirmed_env_ids] = 0
                self.grasp_hold_valid[grasp_confirmed_env_ids] = False

        # LIFT: smooth arithmetic interpolation to cube XY with carry height Z=0.35.
        lift_mask = self.state == self.LIFT
        if torch.any(lift_mask):
            lift_env_ids = torch.where(lift_mask)[0]
            gripper_cmd_all[lift_env_ids] = -1.0
            lift_cube_slots = torch.clamp(self.task_index[lift_env_ids].to(torch.long), 0, num_obs_cubes - 1)
            lift_cube_pos_w = cube_pos_w_reshaped[lift_env_ids, lift_cube_slots]
            lift_start_pos_w = self.grasp_hold_pos_w[lift_env_ids]
            lift_timer = self.state_timer[lift_env_ids].to(dtype=lift_start_pos_w.dtype)
            lift_alpha = torch.clamp((lift_timer + 1.0) / float(40), min=0.0, max=1.0)
            target_pos_w[lift_env_ids, 0:2] = lift_start_pos_w[:, 0:2] + (
                lift_cube_pos_w[:, 0:2] - lift_start_pos_w[:, 0:2]
            ) * lift_alpha.unsqueeze(-1)
            target_pos_w[lift_env_ids, 2] = lift_start_pos_w[:, 2] + (0.30 - lift_start_pos_w[:, 2]) * lift_alpha
            to_approach_target_mask = self.state_timer[lift_env_ids] >= 40
            if torch.any(to_approach_target_mask):
                to_approach_target_env_ids = lift_env_ids[to_approach_target_mask]
                self.state[to_approach_target_env_ids] = self.APPROACH_TARGET
                self.state_timer[to_approach_target_env_ids] = 0

        # APPROACH_TARGET: smooth arithmetic interpolation to final goal XY at carry height Z=0.30.
        approach_target_mask = self.state == self.APPROACH_TARGET
        if torch.any(approach_target_mask):
            approach_target_env_ids = torch.where(approach_target_mask)[0]
            gripper_cmd_all[approach_target_env_ids] = -1.0
            target_slot = torch.clamp(self.task_index[approach_target_env_ids].to(torch.long), 0, self.max_tasks - 1)
            goal_pos_w = self.target_positions[approach_target_env_ids, target_slot]
            approach_cube_slots = torch.clamp(self.task_index[approach_target_env_ids].to(torch.long), 0, num_obs_cubes - 1)
            approach_start_cube_pos_w = cube_pos_w_reshaped[approach_target_env_ids, approach_cube_slots]
            approach_timer = self.state_timer[approach_target_env_ids].to(dtype=goal_pos_w.dtype)
            approach_alpha = torch.clamp((approach_timer + 1.0) / float(40), min=0.0, max=1.0)
            target_pos_w[approach_target_env_ids, 0:2] = approach_start_cube_pos_w[:, 0:2] + (
                goal_pos_w[:, 0:2] - approach_start_cube_pos_w[:, 0:2]
            ) * approach_alpha.unsqueeze(-1)
            target_pos_w[approach_target_env_ids, 2] = 0.30
            to_descend_target_mask = self.state_timer[approach_target_env_ids] >= 40
            if torch.any(to_descend_target_mask):
                to_descend_target_env_ids = approach_target_env_ids[to_descend_target_mask]
                self.state[to_descend_target_env_ids] = self.DESCEND_TARGET
                self.state_timer[to_descend_target_env_ids] = 0
                # Keep release reference pose anchored to the target pose.
                self.release_hold_pos_w[to_descend_target_env_ids] = goal_pos_w[to_descend_target_mask]

        # DESCEND_TARGET: per-step smooth descend, 1 cm each control step.
        descend_target_mask = self.state == self.DESCEND_TARGET
        if torch.any(descend_target_mask):
            descend_target_env_ids = torch.where(descend_target_mask)[0]
            gripper_cmd_all[descend_target_env_ids] = -1.0
            target_slot = torch.clamp(self.task_index[descend_target_env_ids].to(torch.long), 0, self.max_tasks - 1)
            goal_pos_w = self.target_positions[descend_target_env_ids, target_slot]
            target_pos_w[descend_target_env_ids, 0:2] = goal_pos_w[:, 0:2]
            target_pos_w[descend_target_env_ids, 0] = goal_pos_w[:, 0]
            descend_timer = self.state_timer[descend_target_env_ids]
            descend_target_z = 0.30 - 0.01 * descend_timer.to(dtype=goal_pos_w.dtype)
            release_target_z = self.release_hold_pos_w[descend_target_env_ids, 2] + 0.10
            descend_target_z = torch.maximum(descend_target_z, release_target_z)
            target_pos_w[descend_target_env_ids, 2] = descend_target_z

            final_stage_mask = descend_timer >= 38
            # No reach check: enter release right after staged descend finishes.
            to_release_mask = final_stage_mask
            if torch.any(to_release_mask):
                to_release_env_ids = descend_target_env_ids[to_release_mask]
                self.state[to_release_env_ids] = self.RELEASE
                self.state_timer[to_release_env_ids] = 0

        # RELEASE: hold terminal pose and open gripper.
        release_mask = self.state == self.RELEASE
        if torch.any(release_mask):
            release_env_ids = torch.where(release_mask)[0]
            target_pos_w[release_env_ids] = self.release_hold_pos_w[release_env_ids]
            target_pos_w[release_env_ids, 2] = self.release_hold_pos_w[release_env_ids, 2] + 0.10
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
            release_target_z = self.release_hold_pos_w[ascend_target_env_ids, 2] + 0.10
            ascend_target_z = release_target_z + 0.01 * ascend_timer.to(dtype=goal_pos_w.dtype)
            ascend_target_z = torch.clamp(ascend_target_z, max=0.30)
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
            target_pos_w[ascend_home_env_ids, 2] = 0.30
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

        # Orientation policy:
        # - GRASP / RELEASE / DESCEND_CUBE / DESCEND_TARGET: fully lock to vertical-down quaternion.
        # - Other active stages: lock roll/pitch to vertical-down, keep current yaw.
        active_stage_mask = self.state != self.IDLE
        full_lock_mask = (
            (self.state == self.GRASP)
            | (self.state == self.RELEASE)
            | (self.state == self.DESCEND_CUBE)
            | (self.state == self.DESCEND_TARGET)
        )
        rp_lock_mask = torch.logical_and(active_stage_mask, torch.logical_not(full_lock_mask))

        if torch.any(full_lock_mask):
            target_quat_w[full_lock_mask] = 0.0
            target_quat_w[full_lock_mask, 1] = 1.0

        if torch.any(rp_lock_mask):
            _, _, current_yaw = math_utils.euler_xyz_from_quat(ee_quat_w[rp_lock_mask], wrap_to_2pi=False)
            lock_roll = torch.full_like(current_yaw, torch.pi)
            lock_pitch = torch.zeros_like(current_yaw)
            target_quat_w[rp_lock_mask] = math_utils.quat_from_euler_xyz(lock_roll, lock_pitch, current_yaw)

        target_pos_all, target_quat_all = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )

        self.state_timer += 1

        return target_pos_all, target_quat_all, gripper_cmd_all
