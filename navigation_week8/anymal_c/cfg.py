
import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg


model_file = os.path.join(os.path.dirname(__file__), "xmls", "scene.xml")


@dataclass
class NoiseConfig:
    level: float = 0.2
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    # Action in [-1, 1] is scaled to joint target offset around default pose.
    action_scale: float = 0.06


@dataclass
class InitState:
    # World-frame spawn anchor for base free joint.
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.56])

    # Spawn range [x_min, y_min, x_max, y_max].
    pos_randomization_range: list[float] = field(default_factory=lambda: [-5.0, -5.0, 5.0, 5.0])

    # Standing nominal joint pose.
    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "LF_HAA": 0.0,
            "RF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RH_HAA": 0.0,
            "LF_HFE": 0.4,
            "RF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RH_HFE": -0.4,
            "LF_KFE": -0.8,
            "RF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RH_KFE": 0.8,
        }
    )


@dataclass
class Commands:
    # [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max], target pose relative to spawn.
    pose_command_range: list[float] = field(default_factory=lambda: [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14])

    # P-controller gains for pose -> velocity command conversion.
    position_gain: float = 1.0
    yaw_gain: float = 1.0

    # Deadband on heading error before commanding yaw-rate.
    yaw_deadband_deg: float = 8.0

    # Saturation for desired body velocity and yaw-rate command.
    max_command: float = 1.0


@dataclass
class Normalization:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05


@dataclass
class Asset:
    body_name: str = "base"
    foot_names: list[str] = field(default_factory=lambda: ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    terminate_after_contacts_on: list[str] = field(default_factory=lambda: ["base"])
    # Ground geom name in navigation scene.xml.
    ground_name: str = "floor"


@dataclass
class Sensor:
    base_linvel: str = "base_linvel"
    base_gyro: str = "base_gyro"


@dataclass
class RewardConfig:
    # Termination penalty if fallen / exploded.
    termination_penalty: float = -20.0

    # Not reached target: movement tracking + approach shaping.
    tracking_lin_weight: float = 1.5
    tracking_ang_weight: float = 0.3
    approach_scale: float = 4.0
    approach_clip: float = 1.0

    # Reached target: stop-and-stabilize reward.
    stop_bonus_scale: float = 2.0
    zero_ang_bonus: float = 6.0
    arrival_bonus: float = 10.0

    # Regularization / penalties.
    lin_vel_z_penalty: float = -2.0
    ang_vel_xy_penalty: float = -0.05
    orientation_penalty: float = 0.0
    torque_penalty: float = -1e-5
    dof_vel_penalty: float = 0.0
    action_rate_penalty: float = -1e-3


@registry.envcfg("anymal_c_navigation_flat-v0")
@dataclass
class AnymalCEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0
    max_episode_seconds: float = 7.0
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01

    # Joint velocity safety threshold.
    max_dof_vel: float = 100.0

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


