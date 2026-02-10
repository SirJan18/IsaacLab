#!/usr/bin/env python3
"""
Crazyflie L2F Hover Training Script - Fixed Version

This script trains a Crazyflie 2.1 hover policy that's fully compatible with
the Learning to Fly (L2F) framework and deployable to STM32 firmware.

KEY DESIGN DECISIONS (matching L2F exactly):
1. Observation: 146 dims (position, rotation matrix, velocity, angular velocity, action history)
2. Actions: 4 normalized motor RPM commands in [-1, 1]
3. Reward: Squared cost formulation from L2F
4. Network: 64->64 hidden layers with tanh activation
5. Observation normalization baked into first layer for export

Usage:
    # Training mode (headless, 4096 envs, 500 iterations)
    python train_hover.py --num_envs 4096 --max_iterations 500 --headless
    
    # Play mode with trained checkpoint
    python train_hover.py --play --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Isaac Sim setup - must happen before other imports
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie L2F Hover Training")
    
    # Mode selection
    parser.add_argument("--play", action="store_true", help="Run in play mode with trained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode")
    
    # Training parameters  
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=500, help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N iterations")
    
    # Hyperparameters (tuned for quadrotor)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    
    # AppLauncher adds its own args (including --headless)
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()
    return args


# Parse args and launch Isaac Sim
args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG

# Import our custom Crazyflie configuration
from crazyflie_21_cfg import CRAZYFLIE_21_CFG, CrazyflieL2FParams

# Import flight evaluation utilities
from flight_eval_utils import FlightDataLogger, quat_to_euler


# ==============================================================================
# L2F Physics Constants
# ==============================================================================

class L2FConstants:
    """Physical parameters matching learning-to-fly exactly."""
    
    # Mass and geometry
    MASS = 0.027  # kg (27g)
    ARM_LENGTH = 0.028  # m (28mm)
    GRAVITY = 9.81  # m/s²
    
    # Inertia (diagonal)
    IXX = 3.85e-6    # kg·m²
    IYY = 3.85e-6    # kg·m²
    IZZ = 5.9675e-6  # kg·m²
    
    # Motor model
    THRUST_COEFFICIENT = 3.16e-10  # N/RPM²
    TORQUE_COEFFICIENT = 0.005964552  # Nm/N
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    MOTOR_TIME_CONSTANT = 0.15  # seconds
    
    # Rotor positions (X-config)
    # M1: front-right (+x, -y), M2: back-right (-x, -y)
    # M3: back-left (-x, +y), M4: front-left (+x, +y)
    ROTOR_POSITIONS = [
        (0.028, -0.028, 0.0),   # M1
        (-0.028, -0.028, 0.0),  # M2
        (-0.028, 0.028, 0.0),   # M3
        (0.028, 0.028, 0.0),    # M4
    ]
    
    # Rotor yaw directions: -1=CW, +1=CCW
    ROTOR_YAW_DIRS = [-1.0, 1.0, -1.0, 1.0]
    
    # Computed hover RPM
    @classmethod
    def hover_rpm(cls) -> float:
        thrust_per_motor = cls.MASS * cls.GRAVITY / 4.0
        return math.sqrt(thrust_per_motor / cls.THRUST_COEFFICIENT)
    
    # Hover action in normalized space [-1, 1]
    @classmethod
    def hover_action(cls) -> float:
        return 2.0 * cls.hover_rpm() / cls.RPM_MAX - 1.0


# ==============================================================================
# Environment Configuration
# ==============================================================================

@configclass
class CrazyflieL2FEnvCfg(DirectRLEnvCfg):
    """Configuration for L2F-compatible Crazyflie hover environment."""
    
    # Episode settings
    episode_length_s = 4.0  # L2F uses 4 seconds
    decimation = 1  # Control at physics rate (100 Hz)
    
    # Spaces - CRITICAL: Must match L2F exactly
    # Observation: pos(3) + rot_matrix(9) + lin_vel(3) + ang_vel(3) + action_history(32*4=128) = 146
    observation_space = 146
    action_space = 4  # 4 motor RPM commands
    state_space = 0
    debug_vis = True
    
    # Simulation - 100 Hz physics (L2F uses 100Hz)
    sim: SimulationCfg = SimulationCfg(
        dt=1/100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )
    
    # Robot - use custom Crazyflie 2.1 with L2F parameters
    robot: ArticulationCfg = CRAZYFLIE_21_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # L2F Reward parameters - tuned for stable hover learning
    reward_scale = 1.0  # Increased for sharper gradients
    reward_constant = 2.0
    reward_position_weight = 50.0  # Very high - strongly penalize XY drift
    reward_height_weight = 20.0  # Extra penalty for height error (upward drift)
    reward_orientation_weight = 30.0  # High - tilt causes lateral acceleration
    reward_xy_velocity_weight = 10.0  # High - penalize horizontal motion strongly
    reward_z_velocity_weight = 2.0  # Lower - allow small vertical adjustments
    reward_angular_velocity_weight = 2.0  # Penalize spinning/wobbling
    reward_action_weight = 0.01
    reward_action_baseline = 0.334  # L2F default
    
    # Initialization - STAGE 1: PERFECT START (no perturbations)
    # Start exactly at target height with zero velocity
    # This teaches the drone to maintain hover first before handling perturbations
    init_target_height = 1.0  # m - target hover height
    init_height_offset_min = 0.0  # No height offset - spawn exactly at target
    init_height_offset_max = 0.0  # No height offset
    init_max_xy_offset = 0.0  # No XY offset - spawn exactly at target XY
    init_max_angle = 0.0  # No angle perturbation - spawn perfectly level
    init_max_linear_velocity = 0.0  # No initial velocity
    init_max_angular_velocity = 0.0  # No initial angular velocity
    init_guidance_probability = 1.0  # Always spawn perfectly at target
    
    # Termination thresholds - tight for Stage 1 (perfect hover)
    term_xy_threshold = 0.5  # m - must stay within 50cm of target XY
    term_z_min = 0.5  # m - must not drop below 50cm
    term_z_max = 1.5  # m - must not go above 1.5m
    term_tilt_threshold = 0.5  # rad (~29 deg) - tight tilt limit
    term_linear_velocity_threshold = 2.0  # m/s - must not move too fast
    term_angular_velocity_threshold = 5.0  # rad/s - must not spin too fast
    
    # Domain randomization
    enable_disturbance = True
    disturbance_force_std = 0.0132  # N (mass * g / 20)
    disturbance_torque_std = 2.65e-5  # Nm
    
    # Action history
    action_history_length = 32


# ==============================================================================
# Environment Implementation
# ==============================================================================

class CrazyflieL2FEnv(DirectRLEnv):
    """Crazyflie environment implementing L2F physics and control contract."""
    
    cfg: CrazyflieL2FEnvCfg
    
    def __init__(self, cfg: CrazyflieL2FEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Cache physics parameters
        self._mass = L2FConstants.MASS
        self._arm_length = L2FConstants.ARM_LENGTH
        self._thrust_coef = L2FConstants.THRUST_COEFFICIENT
        self._torque_coef = L2FConstants.TORQUE_COEFFICIENT
        self._motor_tau = L2FConstants.MOTOR_TIME_CONSTANT
        self._min_rpm = L2FConstants.RPM_MIN
        self._max_rpm = L2FConstants.RPM_MAX
        self._gravity = L2FConstants.GRAVITY
        self._hover_rpm = L2FConstants.hover_rpm()
        self._hover_action = L2FConstants.hover_action()
        self._dt = cfg.sim.dt
        
        # Motor dynamics alpha
        self._motor_alpha = min(self._dt / self._motor_tau, 1.0)
        
        # Rotor geometry tensors
        self._rotor_positions = torch.tensor(
            L2FConstants.ROTOR_POSITIONS, device=self.device, dtype=torch.float32
        )
        self._rotor_yaw_dirs = torch.tensor(
            L2FConstants.ROTOR_YAW_DIRS, device=self.device, dtype=torch.float32
        )
        
        # Pre-compute mixer matrix (4 motors -> [Fx,Fy,Fz, Tx,Ty,Tz])
        # This replaces per-element torque computation with a single matmul
        rp = self._rotor_positions
        yd = self._rotor_yaw_dirs
        # mixer: (4, 3) for torques: [roll, pitch, yaw] = mixer^T @ thrust_per_motor
        self._torque_mixer = torch.stack([
            rp[:, 1],                    # roll  = sum(F_i * y_i)
            -rp[:, 0],                   # pitch = -sum(F_i * x_i)
            self._torque_coef * yd,      # yaw   = k_torque * sum(dir_i * F_i)
        ], dim=-1)  # shape: (4, 3)
        
        # State tensors
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._rpm_state = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Force/torque buffers
        self._thrust_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torque_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Action history buffer (32 timesteps * 4 actions)
        self._action_history = torch.zeros(
            self.num_envs, cfg.action_history_length, 4, device=self.device
        )
        
        # Disturbance forces
        self._disturbance_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_torque = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Episode statistics
        self._episode_sums = {
            "xy_cost": torch.zeros(self.num_envs, device=self.device),
            "height_cost": torch.zeros(self.num_envs, device=self.device),
            "orientation_cost": torch.zeros(self.num_envs, device=self.device),
            "xy_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "z_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "angular_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "action_cost": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Get body ID for force application
        self._body_id = self._robot.find_bodies("body")[0]
        
        # Cache target positions (env_origins + target height) to avoid recomputing every step
        self._target_pos = self._terrain.env_origins.clone()
        self._target_pos[:, 2] += self.cfg.init_target_height
        
        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
        # Print info
        self._print_env_info()
    
    def _print_env_info(self):
        """Print environment configuration."""
        print("\n" + "="*60)
        print("Crazyflie L2F Hover Environment (Stability-Focused)")
        print("="*60)
        print(f"  Physics dt:        {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Episode length:    {self.cfg.episode_length_s:.1f} s")
        print(f"  Num envs:          {self.num_envs}")
        print(f"  Observation dim:   {self.cfg.observation_space}")
        print(f"  Action dim:        {self.cfg.action_space}")
        print(f"  Mass:              {self._mass*1000:.1f} g")
        print(f"  Hover RPM:         {self._hover_rpm:.0f}")
        print(f"  Hover action:      {self._hover_action:.4f}")
        print(f"  Motor alpha:       {self._motor_alpha:.4f}")
        print(f"  Target height:     {self.cfg.init_target_height:.2f} m")
        print("="*60 + "\n")
    
    def _setup_scene(self):
        """Set up the simulation scene."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w,x,y,z] to flattened rotation matrix (9 elements).
        
        MUST match L2F observe_rotation_matrix() exactly.
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Row-major order as in L2F
        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*w*z
        r02 = 2*x*z + 2*w*y
        r10 = 2*x*y + 2*w*z
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*w*x
        r20 = 2*x*z - 2*w*y
        r21 = 2*y*z + 2*w*x
        r22 = 1 - 2*x*x - 2*y*y
        
        return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions using L2F motor model.
        
        FIRMWARE-COMPATIBLE Motor Model (no hover bias):
        1. Actions are normalized RPM commands in [-1, 1]
        2. Map DIRECTLY to target RPM: target_rpm = (action + 1) / 2 * max_rpm
        3. First-order dynamics: rpm += alpha * (target_rpm - rpm)
        4. Thrust per motor: F = k_f * rpm²
        5. Compute body forces/torques via mixer
        
        CRITICAL: The firmware does NOT add any hover bias. Actions in [-1, 1]
        map directly to [0, MAX_RPM]. For hover, policy must output ~0.334.
        This matches firmware's rl_tools_controller.c:
            float a_pp = (action_output[i] + 1)/2;
            float des_rpm = (MAX_RPM - MIN_RPM) * a_pp + MIN_RPM;
        """
        # Store and clamp actions
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # Map DIRECTLY to target RPM - NO hover bias!
        # This matches the firmware's interpretation exactly.
        # At hover, policy should learn to output ~0.334 (hover_action)
        target_rpm = (self._actions + 1.0) / 2.0 * self._max_rpm
        
        # Apply first-order motor dynamics
        self._rpm_state = self._rpm_state + self._motor_alpha * (target_rpm - self._rpm_state)
        self._rpm_state = self._rpm_state.clamp(self._min_rpm, self._max_rpm)
        
        # Compute thrust per motor: F = k_f * rpm²
        thrust_per_motor = self._thrust_coef * self._rpm_state ** 2
        
        # Total thrust (body z-axis)
        self._thrust_body[:, 0, :2] = 0.0
        self._thrust_body[:, 0, 2] = thrust_per_motor.sum(dim=-1)
        
        # Compute torques via pre-computed mixer matrix: (N, 4) @ (4, 3) -> (N, 3)
        self._torque_body[:, 0, :] = thrust_per_motor @ self._torque_mixer
        
        # Add disturbances
        if self.cfg.enable_disturbance:
            self._thrust_body[:, 0, :] += self._disturbance_force
            self._torque_body[:, 0, :] += self._disturbance_torque
        
        # Update action history (roll without clone)
        self._action_history = torch.roll(self._action_history, -1, dims=1)
        self._action_history[:, -1] = self._actions
    
    def _apply_action(self):
        """Apply forces and torques to the robot."""
        self._robot.set_external_force_and_torque(
            forces=self._thrust_body,
            torques=self._torque_body,
            body_ids=self._body_id,
        )
        self._robot.write_data_to_sim()
    
    def _get_observations(self) -> dict:
        """Construct observations matching L2F firmware format (146 dims).
        
        CRITICAL: Observations MUST match firmware's update_state() exactly!
        
        Firmware clips observations:
        - Position: ±0.5m (pos_distance_limit_position)
        - Velocity: ±2.0 m/s (vel_distance_limit_position)
        
        Layout:
        - [0:3]   Position error (clipped to ±0.5m)
        - [3:12]  Rotation matrix (9 elements, row-major)
        - [12:15] Linear velocity (clipped to ±2.0 m/s)
        - [15:18] Angular velocity in body frame (radians/s)
        - [18:146] Action history (32 * 4 = 128)
        """
        # Get state
        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # Position relative to target (target is at init_target_height above ground)
        # Firmware computes: state->position - target_pos
        pos_error = pos_w - self._target_pos
        
        # CRITICAL: Clip position error to match firmware
        # Firmware: clip(pos_error, -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT)
        pos_error_clipped = pos_error.clamp(-0.5, 0.5)
        
        # CRITICAL: Clip velocity to match firmware
        # Firmware: clip(velocity_error, -VEL_DISTANCE_LIMIT, VEL_DISTANCE_LIMIT)
        lin_vel_clipped = lin_vel_w.clamp(-2.0, 2.0)
        
        # Rotation matrix
        rot_matrix = self._quat_to_rotation_matrix(quat_w)
        
        # Action history (flatten)
        action_history_flat = self._action_history.view(self.num_envs, -1)
        
        # Concatenate (146 dims total)
        obs = torch.cat([
            pos_error_clipped,   # 3 (clipped to ±0.5m)
            rot_matrix,          # 9
            lin_vel_clipped,     # 3 (clipped to ±2.0 m/s)
            ang_vel_b,           # 3
            action_history_flat, # 128
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward matching L2F squared cost formulation.
        
        reward = -scale * weighted_cost + constant
        
        where weighted_cost = pos_w * ||pos||² + ori_w * (1-qw²) + vel_w * ||vel||² + act_w * ||act-baseline||²
        """
        cfg = self.cfg
        
        # Get state
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # Position relative to target (target is at init_target_height)
        pos_error = pos_w - self._target_pos
        
        # XY position cost: ||xy_error||²
        xy_cost = (pos_error[:, :2] ** 2).sum(dim=-1)
        
        # Height cost: height_error² (separate to penalize upward drift more)
        height_cost = pos_error[:, 2] ** 2
        
        # Orientation cost: 1 - qw² (deviation from upright)
        orientation_cost = 1.0 - quat[:, 0] ** 2
        
        # Split velocity into XY (horizontal) and Z (vertical) components
        # Penalize horizontal drift velocity more than vertical adjustments
        xy_velocity_cost = (lin_vel[:, :2] ** 2).sum(dim=-1)
        z_velocity_cost = lin_vel[:, 2] ** 2
        
        # Angular velocity cost: ||ang_vel||²
        angular_velocity_cost = (ang_vel ** 2).sum(dim=-1)
        
        # Action cost: penalize deviation from hover action
        # Since we removed the hover bias, optimal action is now ~0.334 (hover_action)
        # This encourages the policy to output actions near the hover point
        action_deviation = self._actions - self._hover_action
        action_cost = (action_deviation ** 2).sum(dim=-1)
        
        # Weighted sum
        weighted_cost = (
            cfg.reward_position_weight * xy_cost +
            cfg.reward_height_weight * height_cost +
            cfg.reward_orientation_weight * orientation_cost +
            cfg.reward_xy_velocity_weight * xy_velocity_cost +
            cfg.reward_z_velocity_weight * z_velocity_cost +
            cfg.reward_angular_velocity_weight * angular_velocity_cost +
            cfg.reward_action_weight * action_cost
        )
        
        # Compute reward
        reward = -cfg.reward_scale * weighted_cost + cfg.reward_constant
        
        # Clamp reward to [0, 2] for stable learning
        reward = reward.clamp(0.0, cfg.reward_constant)
        
        # Track stats
        self._episode_sums["xy_cost"] += xy_cost
        self._episode_sums["height_cost"] += height_cost
        self._episode_sums["orientation_cost"] += orientation_cost
        self._episode_sums["xy_velocity_cost"] += xy_velocity_cost
        self._episode_sums["z_velocity_cost"] += z_velocity_cost
        self._episode_sums["angular_velocity_cost"] += angular_velocity_cost
        self._episode_sums["action_cost"] += action_cost
        self._episode_sums["total_reward"] += reward
        
        return reward
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        cfg = self.cfg
        
        # Get state
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # XY position relative to env origin
        xy_offset = pos_w[:, :2] - self._terrain.env_origins[:, :2]
        xy_exceeded = torch.norm(xy_offset, dim=-1) > cfg.term_xy_threshold
        
        # Height check
        height = pos_w[:, 2] - self._terrain.env_origins[:, 2]
        too_low = height < cfg.term_z_min
        too_high = height > cfg.term_z_max
        
        # Tilt check: compute tilt angle from quaternion
        # For small tilts: tilt_angle ≈ acos(qw) * 2, but qw² relates to uprightness
        # Use: tilt_angle² ≈ 2*(1 - qw²) for small angles
        # For larger angles: use actual acos
        qw = quat[:, 0]
        tilt_angle = 2.0 * torch.acos(torch.clamp(torch.abs(qw), 0.0, 1.0))
        too_tilted = tilt_angle > cfg.term_tilt_threshold
        
        # Velocity checks
        lin_vel_exceeded = torch.norm(lin_vel, dim=-1) > cfg.term_linear_velocity_threshold
        ang_vel_exceeded = torch.norm(ang_vel, dim=-1) > cfg.term_angular_velocity_threshold
        
        terminated = xy_exceeded | too_low | too_high | too_tilted | lin_vel_exceeded | ang_vel_exceeded
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return
        
        # Log stats before reset
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            extras = {}
            for key, values in self._episode_sums.items():
                avg = torch.mean(values[env_ids]).item()
                steps = self.episode_length_buf[env_ids].float().mean().item()
                if steps > 0:
                    extras[f"Episode/{key}"] = avg / steps
            self.extras["log"] = extras
        
        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        n = len(env_ids)
        cfg = self.cfg
        
        # Guidance: spawn perfectly at target with some probability
        guidance_mask = torch.rand(n, device=self.device) < cfg.init_guidance_probability
        
        # Initialize position near target height with perturbations
        pos = torch.zeros(n, 3, device=self.device)
        
        # XY offset for non-guided envs
        pos[~guidance_mask, 0] = torch.empty((~guidance_mask).sum(), device=self.device).uniform_(
            -cfg.init_max_xy_offset, cfg.init_max_xy_offset
        )
        pos[~guidance_mask, 1] = torch.empty((~guidance_mask).sum(), device=self.device).uniform_(
            -cfg.init_max_xy_offset, cfg.init_max_xy_offset
        )
        
        # Height: target height + random offset (allows learning to ascend/descend)
        height_offset = torch.empty(n, device=self.device).uniform_(
            cfg.init_height_offset_min, cfg.init_height_offset_max
        )
        height_offset[guidance_mask] = 0  # Guided envs start at exact target
        pos[:, 2] = cfg.init_target_height + height_offset
        pos = pos + self._terrain.env_origins[env_ids]
        
        # Sample orientation (small random quaternion)
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0  # Identity
        if cfg.init_max_angle > 0:
            # Small random rotation for non-guided envs
            axis = torch.randn(n, 3, device=self.device)
            axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
            angle = torch.empty(n, device=self.device).uniform_(0, cfg.init_max_angle)
            angle[guidance_mask] = 0
            
            half_angle = angle / 2
            quat[:, 0] = torch.cos(half_angle)
            quat[:, 1:] = axis * torch.sin(half_angle).unsqueeze(-1)
            quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
        
        # Sample velocities
        lin_vel = torch.empty(n, 3, device=self.device).uniform_(
            -cfg.init_max_linear_velocity, cfg.init_max_linear_velocity
        )
        lin_vel[guidance_mask] = 0
        
        ang_vel = torch.empty(n, 3, device=self.device).uniform_(
            -cfg.init_max_angular_velocity, cfg.init_max_angular_velocity
        )
        ang_vel[guidance_mask] = 0
        
        # Write to sim
        root_pose = torch.cat([pos, quat], dim=-1)
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)
        
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        
        # Initialize motor state to hover RPM (so drone is already flying)
        self._rpm_state[env_ids] = self._hover_rpm
        
        # Initialize action history to hover action (not 0!)
        # Since we removed hover bias, the policy must output hover_action for hover
        # Action history should reflect this initial hover state
        self._action_history[env_ids] = self._hover_action
        self._actions[env_ids] = self._hover_action
        
        # Sample disturbances
        if cfg.enable_disturbance:
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_torque_std
        
        # Reset stats
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization."""
        if debug_vis:
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self._goal_markers = VisualizationMarkers(marker_cfg)
        else:
            if hasattr(self, "_goal_markers"):
                self._goal_markers.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "_goal_markers"):
            goal_pos = self._terrain.env_origins.clone()
            goal_pos[:, 2] += self.cfg.init_target_height  # Show target height
            self._goal_markers.visualize(goal_pos)


# ==============================================================================
# L2F-Compatible Actor Network
# ==============================================================================

class L2FActorNetwork(nn.Module):
    """Actor network matching L2F architecture exactly.
    
    Architecture: 146 -> 64 (tanh) -> 64 (tanh) -> 4 (tanh)
    
    This MUST match the firmware expectation for successful deployment.
    
    IMPORTANT: Output is biased toward hover action (~0.334) at initialization.
    """
    
    # Hover action value (computed from physics)
    HOVER_ACTION = 2.0 * math.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) / 21702.0 - 1.0
    
    def __init__(self, obs_dim: int = 146, hidden_dim: int = 64, action_dim: int = 4, init_std: float = 0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Learnable log std - start with small std for stable learning
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(init_std))
        
        # Initialize weights with hover action bias
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
        # Output layer: small weights, bias toward hover action
        # We want tanh(fc3(x)) ≈ HOVER_ACTION when x ≈ 0
        # Since atanh(0.334) ≈ 0.347, set bias to achieve this
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        # Bias the output toward hover action
        hover_bias = math.atanh(max(-0.99, min(0.99, self.HOVER_ACTION)))
        nn.init.constant_(self.fc3.bias, hover_bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning mean actions bounded to [-1, 1]."""
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        return mean
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Get action from policy."""
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-1.0, 1.0)
    
    def get_action_and_log_prob(self, obs: torch.Tensor):
        """Get action and log probability."""
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1.0, 1.0), log_prob


class L2FCriticNetwork(nn.Module):
    """Critic network matching L2F architecture.
    
    Architecture: 146 -> 64 (tanh) -> 64 (tanh) -> 1 (linear)
    """
    
    def __init__(self, obs_dim: int = 146, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)


# ==============================================================================
# PPO Agent with Observation Normalization
# ==============================================================================

class RunningMeanStd:
    """Running mean and std for observation normalization."""
    
    def __init__(self, shape: tuple, epsilon: float = 1e-8, device: torch.device = None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon
        self.device = device
    
    def update(self, x: torch.Tensor):
        """Update statistics with batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observation."""
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.device = device
        return self


class L2FPPOAgent:
    """PPO Agent with L2F-compatible architecture and observation normalization."""
    
    def __init__(
        self,
        obs_dim: int = 146,
        action_dim: int = 4,
        device: torch.device = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks (L2F architecture)
        self.actor = L2FActorNetwork(obs_dim, 64, action_dim).to(device)
        self.critic = L2FCriticNetwork(obs_dim, 64).to(device)
        
        # Observation normalization
        self.obs_normalizer = RunningMeanStd((obs_dim,), device=device)
        self.normalize_observations = True
        
        # Optimizer - actor.parameters() includes log_std, don't add it twice
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def normalize_obs(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Normalize observations."""
        if not self.normalize_observations:
            return obs
        if update:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Get action from policy."""
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=False)
            return self.actor.get_action(obs_norm, deterministic)
    
    def get_action_and_value(self, obs: torch.Tensor):
        """Get action, log prob, and value."""
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=True)
            action, log_prob = self.actor.get_action_and_log_prob(obs_norm)
            value = self.critic(obs_norm)
        return action, log_prob, value
    
    def get_value(self, obs: torch.Tensor):
        """Get value estimate."""
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=False)
            return self.critic(obs_norm)
    
    def update(self, obs: torch.Tensor, actions: torch.Tensor,
               log_probs: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor,
               minibatch_size: int = 4096):
        """PPO update step with minibatch training."""
        obs = obs.detach()
        actions = actions.detach()
        log_probs = log_probs.detach()
        returns = returns.detach()
        advantages = advantages.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Normalize observations (already updated during collection)
        obs_norm = self.normalize_obs(obs, update=False)
        
        total_loss = 0.0
        num_samples = obs.shape[0]
        num_updates = 0
        
        for _ in range(self.epochs):
            # Shuffle indices for minibatch sampling
            indices = torch.randperm(num_samples, device=obs.device)
            
            for start in range(0, num_samples, minibatch_size):
                end = min(start + minibatch_size, num_samples)
                mb_idx = indices[start:end]
                
                mb_obs = obs_norm[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs = log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                
                # Get current policy
                mean = self.actor(mb_obs)
                std = torch.exp(self.actor.log_std)
                dist = torch.distributions.Normal(mean, std)
                
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss (clipped)
                ratio = (new_log_probs - mb_log_probs).exp()
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()
                
                # Value loss
                values = self.critic(mb_obs)
                value_loss = ((values - mb_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return total_loss / max(num_updates, 1)
    
    def save(self, path: str, iteration: int, best_reward: float):
        """Save checkpoint with observation normalization stats."""
        torch.save({
            "iteration": iteration,
            "best_reward": best_reward,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.actor.log_std.data,
            "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.obs_normalizer.mean,
            "obs_var": self.obs_normalizer.var,
            "obs_count": self.obs_normalizer.count,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.log_std.data = checkpoint["log_std"]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "obs_mean" in checkpoint:
            self.obs_normalizer.mean = checkpoint["obs_mean"].to(self.device)
            self.obs_normalizer.var = checkpoint["obs_var"].to(self.device)
            self.obs_normalizer.count = checkpoint["obs_count"]
        return checkpoint.get("iteration", 0), checkpoint.get("best_reward", 0.0)


# ==============================================================================
# Training Loop
# ==============================================================================

def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    
    # Pre-compute masks for all timesteps
    not_dones = 1.0 - dones.float()
    
    # Shift values to get next_values: values[1:] + [next_value]
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
    
    # Compute all TD residuals at once
    deltas = rewards + gamma * next_values * not_dones - values
    
    # Sequential scan for GAE (unavoidable due to recurrence)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        last_gae = deltas[t] + gamma * gae_lambda * not_dones[t] * last_gae
        advantages[t] = last_gae
    
    returns = advantages + values
    return returns, advantages


def train(env: CrazyflieL2FEnv, agent: L2FPPOAgent, args):
    """Main training loop."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # L2F uses longer rollouts
    steps_per_rollout = 128
    num_envs = env.num_envs
    
    best_reward = float("-inf")
    
    print(f"\n{'='*60}")
    print("Starting L2F-Compatible PPO Training")
    print(f"{'='*60}")
    print(f"  Environments:       {num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {steps_per_rollout}")
    print(f"  Total batch size:   {steps_per_rollout * num_envs}")
    print(f"  Observation dim:    {env.cfg.observation_space}")
    print(f"  Action dim:         {env.cfg.action_space}")
    print(f"{'='*60}\n")
    
    for iteration in range(args.max_iterations):
        # Collect rollout
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []
        
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        
        episode_rewards = torch.zeros(num_envs, device=env.device)
        
        for step in range(steps_per_rollout):
            action, log_prob, value = agent.get_action_and_value(obs)
            
            obs_buffer.append(obs)
            action_buffer.append(action)
            log_prob_buffer.append(log_prob)
            value_buffer.append(value)
            
            obs_dict, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_dict["policy"]
            done = terminated | truncated
            
            reward_buffer.append(reward)
            done_buffer.append(done)
            episode_rewards += reward
            
            obs = next_obs
        
        # Stack buffers
        obs_t = torch.stack(obs_buffer)
        actions_t = torch.stack(action_buffer)
        log_probs_t = torch.stack(log_prob_buffer)
        values_t = torch.stack(value_buffer)
        rewards_t = torch.stack(reward_buffer)
        dones_t = torch.stack(done_buffer)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = agent.get_value(obs)
        
        returns_t, advantages_t = compute_gae(
            rewards_t, values_t, dones_t, next_value,
            gamma=agent.gamma, gae_lambda=agent.gae_lambda
        )
        
        # Flatten for update
        obs_flat = obs_t.reshape(-1, obs_t.shape[-1])
        actions_flat = actions_t.reshape(-1, actions_t.shape[-1])
        log_probs_flat = log_probs_t.reshape(-1)
        returns_flat = returns_t.reshape(-1)
        advantages_flat = advantages_t.reshape(-1)
        
        # Update policy
        loss = agent.update(obs_flat, actions_flat, log_probs_flat, returns_flat, advantages_flat)
        
        # Compute stats
        mean_reward = episode_rewards.mean().item() / steps_per_rollout
        mean_return = returns_flat.mean().item()
        
        # Check for best model
        is_best = mean_reward > best_reward
        if is_best:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward)
        
        # Log progress
        if iteration % 10 == 0 or is_best:
            star = " *BEST*" if is_best else ""
            print(f"[Iter {iteration:4d}] Reward: {mean_reward:8.3f} | Return: {mean_return:8.2f} | Loss: {loss:.4f}{star}")
        
        # Save checkpoint periodically
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward)
    
    # Save final model
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), args.max_iterations, best_reward)
    print(f"\nTraining complete! Best reward: {best_reward:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Print export instructions
    print("\n" + "="*60)
    print("Next Steps for Firmware Export:")
    print("="*60)
    print("1. Run export_to_firmware.py to generate actor.h")
    print("2. Build firmware with firmware/build_firmware.py")
    print("3. Flash cf2.bin to Crazyflie")
    plt.show()


def play(env: CrazyflieL2FEnv, agent: L2FPPOAgent, checkpoint_path: str):
    """Run trained policy with visualization and data logging."""
    iteration, best_reward = agent.load(checkpoint_path)
    print(f"\n[Play Mode] Loaded checkpoint from iteration {iteration}")
    print(f"[Play Mode] Best training reward: {best_reward:.3f}")
    print("[Play Mode] Press Ctrl+C to stop\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    step_count = 0
    episode_reward = 0.0
    
    # Initialize logger
    logger = FlightDataLogger()
    
    # Create eval directory structure with timestamp
    run_tag = int(time.time())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "eval", "hover", f"hover_{run_tag}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # File paths for periodic saving (overwrite each time)
    csv_filename = os.path.join(eval_dir, "hover_eval_latest.csv")
    title_prefix = "Hover Evaluation"
    
    try:
        while simulation_app.is_running():
            action = agent.get_action(obs, deterministic=True)
            obs_dict, reward, _, _, _ = env.step(action)
            obs = obs_dict["policy"]
            
            episode_reward += reward.mean().item()
            step_count += 1
            
            # Log flight data using FlightDataLogger
            logger.log_step(env, env_idx=0)
            
            # Save every 500 steps
            if step_count % 500 == 0:
                print(f"[Step {step_count:5d}] Episode reward: {episode_reward:.2f} | Saving...")
                logger.save_and_plot(csv_filename, title_prefix=title_prefix, output_dir=eval_dir)
            elif step_count % 100 == 0:
                print(f"[Step {step_count:5d}] Episode reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\n[Play Mode] Stopped by user")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    # Create config
    cfg = CrazyflieL2FEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    # Create environment
    env = CrazyflieL2FEnv(cfg)
    
    # Create agent
    agent = L2FPPOAgent(
        obs_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
    )
    
    if args.play:
        # Play mode
        if args.checkpoint is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            args.checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        play(env, agent, args.checkpoint)
    else:
        # Training mode
        train(env, agent, args)
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
