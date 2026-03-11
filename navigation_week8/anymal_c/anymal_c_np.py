
# last edit:2026年3月11日 09点16分


# import threading
from dataclasses import dataclass
import json
import pprint

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg

_ENV_NAME = "anymal_c_navigation_flat-v0"


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@registry.env(_ENV_NAME, "np")
class AnymalCEnv(NpEnv):
    _cfg: AnymalCEnvCfg

    _OBS_DIM = 54
    _ACT_DIM = 12

    _GOAL_POS_THRESHOLD = 0.3
    _GOAL_YAW_THRESHOLD = np.deg2rad(15.0)
    _STOP_ANGULAR_THRESHOLD = 0.05
    _TILT_TERMINATION_ANGLE = np.deg2rad(75.0)

    def __init__(self, cfg: AnymalCEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)
        self._target_marker_body = self._safe_get_body("target_marker")
        self._robot_heading_arrow_body = self._safe_get_body("robot_heading_arrow")
        self._desired_heading_arrow_body = self._safe_get_body("desired_heading_arrow")

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._ACT_DIM,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._OBS_DIM,), dtype=np.float32)

        self._init_buffer()
        self._init_contact_geometry()

    def _safe_get_body(self, body_name: str):
        try:
            return self._model.get_body(body_name)
        except Exception:
            return None

    def _init_buffer(self):
        cfg = self._cfg

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        for i in range(self._num_action):
            actuator_name = self._model.actuator_names[i]
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in actuator_name:
                    self.default_angles[i] = angle

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_pos[-self._num_action :] = self.default_angles
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)

        self._gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._action_scale = float(cfg.control_config.action_scale)
        self._commands_scale = np.array(
            [cfg.normalization.lin_vel, 
             cfg.normalization.lin_vel, 
             cfg.normalization.ang_vel], dtype=np.float32
        )

        ctrl_limits = self._model.actuator_ctrl_limits
        self._actuator_ctrl_low = np.asarray(ctrl_limits[0], dtype=np.float32)
        self._actuator_ctrl_high = np.asarray(ctrl_limits[1], dtype=np.float32)

        self._obs_noise_scales = np.zeros((self._OBS_DIM,), dtype=np.float32)
        self._obs_noise_scales[0:3] = cfg.noise_config.scale_linvel
        self._obs_noise_scales[3:6] = cfg.noise_config.scale_gyro
        self._obs_noise_scales[6:9] = cfg.noise_config.scale_gravity
        self._obs_noise_scales[9:21] = cfg.noise_config.scale_joint_angle
        self._obs_noise_scales[21:33] = cfg.noise_config.scale_joint_vel

    def _init_contact_geometry(self):
        ground_idx = None
        for name in [self._cfg.asset.ground_name, "floor", "ground"]:
            try:
                ground_idx = self._model.get_geom_index(name)
                break
            except Exception:
                print("Warning:Can not find ground idx")
                continue

        if ground_idx is None:
            self._termination_contact = None
            print("Warning: Termination contact is None")
            return

        candidate_tokens = [token.lower() for token in self._cfg.asset.terminate_after_contacts_on]
        candidate_tokens.extend(["base", "shell", "battery", "hatch"])

        base_indices = []
        seen = set()
        for geom_name in self._model.geom_names:
            if geom_name is None:
                continue
            lname = geom_name.lower()
            if not any(token in lname for token in candidate_tokens):
                continue
            try:
                geom_idx = self._model.get_geom_index(geom_name)
            except Exception:
                print("Warning:can not find idx,name=",geom_name)
                continue
            if geom_idx in seen:
                continue
            seen.add(geom_idx)
            base_indices.append(geom_idx)

        if not base_indices:
            self._termination_contact = None
            return

        self._termination_contact = np.array([[idx, ground_idx] for idx in base_indices], dtype=np.uint32)

    def _check_base_contact(self, data: mtx.SceneData) -> np.ndarray:
        if self._termination_contact is None:
            return np.zeros((self._num_envs,), dtype=bool)
        cquery = self._model.get_contact_query(data)
        contacts = cquery.is_colliding(self._termination_contact)
        contacts = contacts.reshape((self._num_envs, -1))
        return contacts.any(axis=1)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        # return quaternion.rotate_inverse(quat, self._gravity)
        return quaternion.rotate_vector(quat, self._gravity)

    def _compute_navigation_state(self, root_pos: np.ndarray, root_quat: np.ndarray, info: dict) -> dict[str, np.ndarray]:
        cmd_cfg = self._cfg.commands

        target_pos = info["target_pos"]
        target_yaw = info["target_yaw"]

        robot_pos = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)

        position_error = target_pos - robot_pos
        distance = np.linalg.norm(position_error, axis=1)

        reached_position = distance < self._GOAL_POS_THRESHOLD

        heading_error = _wrap_to_pi(target_yaw - robot_heading)
        reached_heading = np.abs(heading_error) < self._GOAL_YAW_THRESHOLD

        reached_pose = np.logical_and(reached_position, reached_heading)

        desired_vel_xy = np.clip(position_error * float(cmd_cfg.position_gain), -float(cmd_cfg.max_command), float(cmd_cfg.max_command))
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达目标位置的期望速度置为0

        desired_yaw_rate = np.clip(heading_error * float(cmd_cfg.yaw_gain), -float(cmd_cfg.max_command), float(cmd_cfg.max_command))
        deadband = np.deg2rad(float(cmd_cfg.yaw_deadband_deg))
        desired_yaw_rate = np.where(np.abs(heading_error) < deadband, 0.0, desired_yaw_rate)

        desired_yaw_rate = np.where(reached_pose, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_pose[:, np.newaxis], 0.0, desired_vel_xy)

        commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1).astype(np.float32)

        return {
            "position_error": position_error.astype(np.float32),
            "distance": distance.astype(np.float32),
            "heading_error": heading_error.astype(np.float32),
            "reached_position": reached_position,
            "reached_heading": reached_heading,
            "reached_pose": reached_pose,
            "commands": commands,
            "desired_vel_xy": desired_vel_xy.astype(np.float32),
        }

    def _build_observation(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        info: dict,
        nav_state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_pos_rel = joint_pos - self.default_angles
        projected_gravity = self._compute_projected_gravity(root_quat)

        command_obs = nav_state["commands"] * self._commands_scale
        position_error_obs = nav_state["position_error"] / 5.0
        heading_error_obs = (nav_state["heading_error"] / np.pi).reshape(-1, 1)
        distance_obs = np.clip(nav_state["distance"] / 5.0, 0.0, 1.0).reshape(-1, 1)

        stop_ready = np.logical_and(nav_state["reached_pose"], np.abs(base_ang_vel[:, 2]) < self._STOP_ANGULAR_THRESHOLD)

        obs = np.concatenate(
            [
                base_lin_vel * self._cfg.normalization.lin_vel,
                base_ang_vel * self._cfg.normalization.ang_vel,
                projected_gravity,
                joint_pos_rel * self._cfg.normalization.dof_pos,
                joint_vel * self._cfg.normalization.dof_vel,
                info["current_actions"],
                command_obs,
                position_error_obs,
                heading_error_obs,
                distance_obs,
                nav_state["reached_pose"].astype(np.float32).reshape(-1, 1),
                stop_ready.astype(np.float32).reshape(-1, 1),
            ],
            axis=-1,
        ).astype(np.float32)

        noise_level = float(self._cfg.noise_config.level)
        if noise_level > 0.0:
            obs_noise = np.random.uniform(low=-1.0, high=1.0, size=obs.shape).astype(np.float32)
            obs = obs + obs_noise * self._obs_noise_scales * noise_level

        assert obs.shape[1] == self._OBS_DIM, obs.shape
        return obs, stop_ready

    def _set_body_mocap_pose(self, body, data: mtx.SceneData, pose: np.ndarray):
        if body is None:
            return
        mocap = getattr(body, "mocap", None)
        if mocap is None:
            return
        mocap.set_pose(data, pose)

    def _update_target_marker(self, data: mtx.SceneData, target_pos: np.ndarray, target_yaw: np.ndarray):
        num_envs = data.shape[0]
        marker_pos = np.column_stack(
            [
                target_pos[:, 0],
                target_pos[:, 1],
                np.full((num_envs,), 0.5, dtype=np.float32),
            ]
        )
        marker_quat = quaternion.from_euler(0, 0, target_yaw)
        self._set_body_mocap_pose(self._target_marker_body, data, np.concatenate([marker_pos, marker_quat], axis=1))

    def _update_heading_arrows(
        self,
        data: mtx.SceneData,
        root_pos: np.ndarray,
        desired_vel_xy: np.ndarray,
        base_lin_vel_xy: np.ndarray,
    ):
        arrow_pos = root_pos.copy()
        arrow_pos[:, 2] = root_pos[:, 2] + 0.2

        current_yaw = np.where(
            np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0,
        )
        robot_arrow_quat = quaternion.from_euler(0, 0, current_yaw)
        self._set_body_mocap_pose(
            self._robot_heading_arrow_body,
            data,
            np.concatenate([arrow_pos, robot_arrow_quat], axis=1),
        )

        desired_yaw = np.where(
            np.linalg.norm(desired_vel_xy, axis=1) > 1e-6,
            np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]),
            0.0,
        )
        desired_arrow_quat = quaternion.from_euler(0, 0, desired_yaw)
        self._set_body_mocap_pose(
            self._desired_heading_arrow_body,
            data,
            np.concatenate([arrow_pos, desired_arrow_quat], axis=1),
        )

    def _compute_reward(
        self,
        data: mtx.SceneData,
        info: dict,
        nav_state: dict[str, np.ndarray],
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        root_quat: np.ndarray,
        joint_vel: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg_reward = self._cfg.reward_config

        termination_penalty = np.zeros((self._num_envs,), dtype=np.float32)

        # 关节速度异常判定
        vel_max = np.abs(joint_vel).max(axis=1)
        vel_overflow = vel_max > float(self._cfg.max_dof_vel)
        vel_extreme = (np.isnan(joint_vel).any(axis=1)) | (np.isinf(joint_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, cfg_reward.termination_penalty, termination_penalty)

        # base接触地面
        base_contact = self._check_base_contact(data)
        termination_penalty = np.where(base_contact, cfg_reward.termination_penalty, termination_penalty)


        # 侧翻
        projected_gravity = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        gz = projected_gravity[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > self._TILT_TERMINATION_ANGLE
        termination_penalty = np.where(side_flip_mask, cfg_reward.termination_penalty, termination_penalty)

        # 线速度跟踪奖励（未加权）
        lin_vel_error = np.sum(np.square(nav_state["commands"][:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin = np.exp(-lin_vel_error / 0.25)

        # 加速度跟踪奖励（未加权）
        ang_vel_error = np.square(nav_state["commands"][:, 2] - base_ang_vel[:, 2])
        tracking_ang = np.exp(-ang_vel_error / 0.25)

        # 接近奖励（无需加权）
        distance_to_target = nav_state["distance"]
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        approach_reward = np.clip(
            distance_improvement * float(cfg_reward.approach_scale),
            -float(cfg_reward.approach_clip),
            float(cfg_reward.approach_clip),
        )
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        
        # 首次到达奖励
        reached_pose = nav_state["reached_pose"]
        info["ever_reached"] = info.get("ever_reached", np.zeros((self._num_envs,), dtype=bool))
        first_time_reach = np.logical_and(reached_pose, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_pose)
        arrival_bonus = np.where(first_time_reach, cfg_reward.arrival_bonus, 0.0)

        # 到位位置后，如果角速度小于阈值，则给一个零角速度奖励
        yaw_rate_abs = np.abs(base_ang_vel[:, 2])
        zero_ang_mask = yaw_rate_abs < self._STOP_ANGULAR_THRESHOLD
        zero_ang_bonus = np.where(np.logical_and(reached_pose, zero_ang_mask), cfg_reward.zero_ang_bonus, 0.0)

        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        stop_base = float(cfg_reward.stop_bonus_scale) * (
            0.8 * np.exp(-((speed_xy / 0.2) ** 2)) + 1.2 * np.exp(-((yaw_rate_abs / 0.1) ** 4))
        )
        stop_bonus = np.where(reached_pose, stop_base + zero_ang_bonus, 0.0)
        
        orientation_penalty = (
            np.square(projected_gravity[:, 0])
            + np.square(projected_gravity[:, 1])
            + np.square(projected_gravity[:, 2] + 1.0)
        )

        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])
        ang_vel_xy_penalty = np.sum(np.square(base_ang_vel[:, :2]), axis=1)
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        action_delta = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_delta), axis=1)

        regularization = (
            cfg_reward.lin_vel_z_penalty * lin_vel_z_penalty
            + cfg_reward.ang_vel_xy_penalty * ang_vel_xy_penalty
            + cfg_reward.orientation_penalty * orientation_penalty
            + cfg_reward.torque_penalty * torque_penalty
            + cfg_reward.dof_vel_penalty * dof_vel_penalty
            + cfg_reward.action_rate_penalty * action_rate_penalty
        )

        move_reward = (
            cfg_reward.tracking_lin_weight * tracking_lin
            + cfg_reward.tracking_ang_weight * tracking_ang
            + approach_reward
            + regularization
            + termination_penalty
        )

        stop_reward = stop_bonus + arrival_bonus + regularization + termination_penalty

        reward = np.where(reached_pose, stop_reward, move_reward).astype(np.float32)

        reward_terms = {
            "tracking_lin": tracking_lin.astype(np.float32),
            "tracking_ang": tracking_ang.astype(np.float32),
            "approach_reward": approach_reward.astype(np.float32),
            "stop_bonus": stop_bonus.astype(np.float32),
            "arrival_bonus": arrival_bonus.astype(np.float32),
            "lin_vel_z": (-lin_vel_z_penalty).astype(np.float32),
            "ang_vel_xy": (-ang_vel_xy_penalty).astype(np.float32),
            "orientation": (-orientation_penalty).astype(np.float32),
            "torque": (-torque_penalty).astype(np.float32),
            "dof_vel": (-dof_vel_penalty).astype(np.float32),
            "action_rate": (-action_rate_penalty).astype(np.float32),
            "termination": termination_penalty.astype(np.float32),
            "first_reach": first_time_reach.astype(np.float32),
            "move_total": move_reward.astype(np.float32),
            "stop_total": stop_reward.astype(np.float32),
        }

        return reward, reward_terms

    def _compute_terminated(
        self,
        obs: np.ndarray,
        joint_vel: np.ndarray,
        root_quat: np.ndarray,
        data: mtx.SceneData,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        finite_ok = np.isfinite(obs).all(axis=1)
        finite_ok = np.logical_and(finite_ok, np.isfinite(joint_vel).all(axis=1))
        dof_vel_max = np.max(np.abs(joint_vel), axis=1)

        dof_vel_exceeded = dof_vel_max > float(self._cfg.max_dof_vel)
        dof_vel_extreme = dof_vel_max > 1e6

        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        base_tilted = tilt_angle > self._TILT_TERMINATION_ANGLE

        base_contact = self._check_base_contact(data)

        terminated = np.logical_not(finite_ok)
        terminated = np.logical_or(terminated, dof_vel_exceeded)
        terminated = np.logical_or(terminated, dof_vel_extreme)
        terminated = np.logical_or(terminated, base_tilted)
        terminated = np.logical_or(terminated, base_contact)

        term_terms = {
            "nan_or_inf": np.logical_not(finite_ok).astype(np.float32),
            "dof_vel": dof_vel_exceeded.astype(np.float32),
            "tilt": base_tilted.astype(np.float32),
            "base_contact": base_contact.astype(np.float32),
        }

        return terminated, term_terms

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # actions = np.clip(actions.astype(np.float32), -1.0, 1.0)


        if "current_actions" not in state.info:
            state.info["current_actions"] = np.zeros((self._num_envs, self._num_action), dtype=np.float32)
        if "last_actions" not in state.info:
            state.info["last_actions"] = np.zeros((self._num_envs, self._num_action), dtype=np.float32)

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        actions_scaled = actions * self._action_scale
        target_pos = self.default_angles + actions_scaled
        # 指导里似乎没有裁剪操作
        target_pos = np.clip(target_pos, self._actuator_ctrl_low, self._actuator_ctrl_high)

        state.data.actuator_ctrls = target_pos
        return state

    def update_state(self, state: NpEnvState):
        data = state.data

        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]

        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data).astype(np.float32)
        base_ang_vel = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data).astype(np.float32)

        joint_pos = self.get_dof_pos(data).astype(np.float32)
        joint_vel = self.get_dof_vel(data).astype(np.float32)

        nav_state = self._compute_navigation_state(root_pos, root_quat, state.info)

        state.info["commands"] = nav_state["commands"]
        state.info["distance_to_target"] = nav_state["distance"]
        state.info["has_reached_target"] = nav_state["reached_pose"]
        # new add
        state.info["desired_vel_xy"]=nav_state["desired_vel_xy"]

        # self._update_target_marker(data, state.info["target_pos"], state.info["target_yaw"])
        self._update_heading_arrows(data, root_pos, nav_state["desired_vel_xy"], base_lin_vel[:, :2])

        obs, stop_ready = self._build_observation(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            root_quat=root_quat,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            info=state.info,
            nav_state=nav_state,
        )

        reward, reward_terms = self._compute_reward(
            data=data,
            info=state.info,
            nav_state=nav_state,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            root_quat=root_quat,
            joint_vel=joint_vel,
        )

        terminated, term_terms = self._compute_terminated(
            obs=obs,
            joint_vel=joint_vel,
            root_quat=root_quat,
            data=data,
        )

        if "success" not in state.info:
            state.info["success"] = np.zeros((self._num_envs,), dtype=bool)
        success_now = np.logical_and(nav_state["reached_pose"], stop_ready)
        state.info["success"] = np.logical_or(state.info["success"], success_now)

        # state.info["Reward"] = reward_terms
        # state.info["Termination"] = term_terms
        state.info["metrics"] = {
            "distance_to_target": nav_state["distance"].astype(np.float32),
            # "heading_error_abs": np.abs(nav_state["heading_error"]).astype(np.float32),
            "reach_position_rate": nav_state["reached_position"].astype(np.float32),
            "reach_heading_rate": nav_state["reached_heading"].astype(np.float32),
            "reach_rate": nav_state["reached_pose"].astype(np.float32),
            # "stop_ready_rate": stop_ready.astype(np.float32),
            "success_rate": state.info["success"].astype(np.float32),
            # "speed_world_xy": np.linalg.norm(base_lin_vel[:, :2], axis=1).astype(np.float32),
            # "tracking_lin": reward_terms["tracking_lin"],
            # "tracking_ang": reward_terms["tracking_ang"],
            # "approach_reward": reward_terms["approach_reward"],
        }

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        # 对齐locomotion的输出字典
        keys_to_keep=['current_actions','desired_vel_xy','ever_reached','lase_actions',
                      'min_distance','pose_commands','steps']
        outdic={key:state.info[key] for key in keys_to_keep if key in state.info}
        # pprint.pprint(outdic)
        return state

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]

        pos_range = cfg.init_state.pos_randomization_range
        # 初始位置随机生成
        robot_init_x = np.random.uniform(pos_range[0], pos_range[2], num_envs).astype(np.float32)
        robot_init_y = np.random.uniform(pos_range[1], pos_range[3], num_envs).astype(np.float32)
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)

        cmd_range = cfg.commands.pose_command_range
        target_offset = np.random.uniform(
            low=np.array(cmd_range[:2], dtype=np.float32),
            high=np.array(cmd_range[3:5], dtype=np.float32),
            size=(num_envs, 2),
        ).astype(np.float32)
        target_pos = robot_init_pos + target_offset

        target_yaw = np.random.uniform(low=cmd_range[2], high=cmd_range[5], size=(num_envs,)).astype(np.float32)
        pose_commands=np.concatenate([target_pos,target_yaw[:,np.newaxis]],axis=1)

        # dof_pos指的是三个轴的位置和速度（6）+12个可活动关节的位置/速度
        init_dof_pos = np.tile(self._init_dof_pos, (num_envs, 1)).astype(np.float32) # tile:矩阵重复
        init_dof_vel = np.tile(self._init_dof_vel, (num_envs, 1)).astype(np.float32)
        # print("Reset:init dof pos=",init_dof_pos)
        # print("init dof vel=",init_dof_vel)

        # init_state.pos 设定的中心位置
        noise_pos = np.zeros((num_envs, self._num_dof_pos), dtype=np.float32)
        noise_pos[:, 0] = robot_init_x - float(cfg.init_state.pos[0])
        noise_pos[:, 1] = robot_init_y - float(cfg.init_state.pos[1])

        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]

        info = {
            "target_pos": target_pos,
            "target_yaw": target_yaw,
            "pose_commands": pose_commands,
            "commands": np.zeros((num_envs, 3), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "has_reached_target": np.zeros((num_envs,), dtype=bool),
            "distance_to_target": np.zeros((num_envs,), dtype=np.float32),
            "ever_reached": np.zeros((num_envs,), dtype=bool),
            "min_distance": np.full((num_envs,), np.inf, dtype=np.float32),
            "success": np.zeros((num_envs,), dtype=bool),
        }
        

        nav_state = self._compute_navigation_state(root_pos, root_quat, info)
        info["commands"] = nav_state["commands"]
        info["has_reached_target"] = nav_state["reached_pose"]
        info["distance_to_target"] = nav_state["distance"]
        info["min_distance"] = nav_state["distance"].copy()

        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data).astype(np.float32)
        base_ang_vel = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data).astype(np.float32)
        joint_pos = self.get_dof_pos(data).astype(np.float32)
        joint_vel = self.get_dof_vel(data).astype(np.float32)

        self._update_target_marker(data, info["target_pos"], info["target_yaw"])
        self._update_heading_arrows(data, root_pos, nav_state["desired_vel_xy"], base_lin_vel[:, :2])

        obs, stop_ready = self._build_observation(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            root_quat=root_quat,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            info=info,
            nav_state=nav_state,
        )

        zeros = np.zeros((num_envs,), dtype=np.float32)
        # info["Reward"] = {
        #     "tracking_lin": zeros.copy(),
        #     "tracking_ang": zeros.copy(),
        #     "approach_reward": zeros.copy(),
        #     "stop_bonus": zeros.copy(),
        #     "arrival_bonus": zeros.copy(),
        #     "lin_vel_z": zeros.copy(),
        #     "ang_vel_xy": zeros.copy(),
        #     "orientation": zeros.copy(),
        #     "torque": zeros.copy(),
        #     "dof_vel": zeros.copy(),
        #     "action_rate": zeros.copy(),
        #     "termination": zeros.copy(),
        #     "first_reach": zeros.copy(),
        #     "move_total": zeros.copy(),
        #     "stop_total": zeros.copy(),
        # }
        # info["Termination"] = {
        #     "nan_or_inf": zeros.copy(),
        #     "dof_vel": zeros.copy(),
        #     "tilt": zeros.copy(),
        #     "base_contact": zeros.copy(),
        # }
        info["metrics"] = {
            "distance_to_target": nav_state["distance"].astype(np.float32),
            # "heading_error_abs": np.abs(nav_state["heading_error"]).astype(np.float32),
            "reach_position_rate": nav_state["reached_position"].astype(np.float32),
            "reach_heading_rate": nav_state["reached_heading"].astype(np.float32),
            "reach_rate": nav_state["reached_pose"].astype(np.float32),
            # "stop_ready_rate": stop_ready.astype(np.float32),
            "success_rate": zeros.copy(),
            # "speed_world_xy": np.linalg.norm(base_lin_vel[:, :2], axis=1).astype(np.float32),
            # "tracking_lin": zeros.copy(),
            # "tracking_ang": zeros.copy(),
            # "approach_reward": zeros.copy(),
        }
        # 对齐locomotion的输出字典
        keys_to_keep=['current_actions','ever_reached','lase_actions',
                      'min_distance','pose_commands']
        outdic={key:info[key] for key in keys_to_keep if key in info}
        # pprint.pprint(outdic)
        # print("my np,reset obs=",obs)
        return obs, info


