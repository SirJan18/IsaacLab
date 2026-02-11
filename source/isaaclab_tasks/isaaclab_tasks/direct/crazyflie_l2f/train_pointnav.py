#!/usr/bin/env python3
"""
Crazyflie L2F Point Navigation Training Script

This script trains a Crazyflie 2.1 point navigation policy that layers
goal-directed behavior on top of stable hover. Fully compatible with
the Learning to Fly (L2F) framework and deployable to STM32 firmware.

KEY DESIGN DECISIONS:
1. Base observation: 146 dims (same as hover) + 3 dims goal-relative position = 149 dims
2. Actions: 4 normalized motor RPM commands in [-1, 1]
3. Reward: Hover stability costs + navigation progress reward
4. Network: 64->64 hidden layers with tanh activation
5. Preserves all hover invariants from train_hover.py

INVARIANTS PRESERVED FROM train_hover.py:
- Same motor model (first-order dynamics, α=dt/τ)
- Same physics (100 Hz, 27g mass, L2F thrust coefficients)
- Same action space interpretation ([-1,1] -> [0, MAX_RPM])
- Same base observation layout (pos_error, rot_matrix, vel, ang_vel, action_history)
- Hover stability rewards are ADDED to navigation rewards

Usage (from IsaacLab directory):
    # Sanity test first (verify environment runs correctly)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav.py --sanity_test --num_envs 16
    
    # Training mode (headless, 4096 envs, 1000 iterations)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav.py --num_envs 4096 --max_iterations 1000 --headless
    
    # Play mode with trained checkpoint
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav.py --play --checkpoint source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\checkpoints_pointnav\\best_model.pt --num_envs 64
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
import time

# Isaac Sim setup - must happen before other imports
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie L2F Point Navigation Training")
    
    # Mode selection
    parser.add_argument("--play", action="store_true", help="Run in play mode with trained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode or resume training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint (uses --checkpoint or latest)")
    parser.add_argument("--sanity_test", action="store_true", help="Run sanity test (few steps, verify no crashes)")
    
    # Training parameters  
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")
    
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
from flight_eval_utils import FlightDataLogger


# ==============================================================================
# L2F Physics Constants (IDENTICAL to train_hover.py)
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
class CrazyfliePointNavEnvCfg(DirectRLEnvCfg):
    """Configuration for L2F-compatible Crazyflie point navigation environment.
    
    DIFFERENCES FROM CrazyflieL2FEnvCfg (train_hover.py):
    1. Observation space: 149 = 146 (hover) + 3 (goal relative position)
    2. Longer episodes (10s) to allow time to reach goals
    3. Goal sampling with minimum distance enforcement
    4. Navigation-specific reward terms added to hover stability costs
    """
    
    # Episode settings - longer for navigation
    episode_length_s = 10.0  # 10 seconds to reach goal
    decimation = 1  # Control at physics rate (100 Hz)
    
    # Spaces - Extended from hover
    # Base observation: pos(3) + rot_matrix(9) + lin_vel(3) + ang_vel(3) + action_history(32*4=128) = 146
    # + goal_relative_pos(3) = 149
    observation_space = 149
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
    
    # =========================================================================
    # L2F HOVER STABILITY REWARDS (PRESERVED FROM train_hover.py)
    # These ensure the drone maintains stable flight while navigating
    # =========================================================================
    # ITERATION 5: Weakened hover dominance to allow navigation learning
    # Problem: hover_cost (10-25) was overpowering progress_reward (0.05-0.15)
    # Result: Policy learned "hover and micro-correct" instead of "go to goal"
    # Solution: Reduce hover weights so navigation signal can compete
    # =========================================================================
    hover_reward_scale = 0.2  # Reduced from 0.5 to weaken hover dominance
    hover_reward_constant = 0.5  # Reduced from 1.0 to lower per-step "free reward"
    hover_position_weight = 0.0  # Disabled - we use goal-relative position instead
    hover_height_weight = 6.0  # Reduced from 10.0
    hover_orientation_weight = 15.0  # Reduced from 30.0
    hover_xy_velocity_weight = 0.5  # Reduced from 2.0 - we want motion now
    hover_z_velocity_weight = 1.0  # Reduced from 2.0
    hover_angular_velocity_weight = 1.0  # Reduced from 2.0
    hover_action_weight = 0.01
    
    # Hover gating: reduce hover reward when far from goal
    # Prevents "survive stably" from dominating "reach goal"
    hover_gate_radius = 0.5  # Full hover reward within this XY distance
    hover_gate_min = 0.2  # Minimum hover reward multiplier when far from goal
    
    # =========================================================================
    # NAVIGATION-SPECIFIC REWARDS
    # =========================================================================
    nav_progress_weight = 5.0  # Dense reward for moving toward goal
    nav_reach_bonus = 100.0  # Sparse bonus for reaching goal (was 10.0 - too weak)
    nav_timeout_penalty = 0.0  # No penalty for timeout (just lack of reach bonus)
    nav_braking_weight = 1.0  # Reward for slowing down near goal (within braking_radius)
    nav_braking_radius = 0.3  # Start braking reward when within 30cm of goal
    
    # Height control rewards
    # FIXED: height_recovery is now delta-based (reward for climbing, not for being low)
    nav_height_recovery_weight = 1.0  # Reward for reducing deficit (climbing back up)
    nav_height_track_weight = 0.5  # Gaussian reward for staying near target height
    
    # Speed penalty: penalize going too fast (above soft linvel threshold)
    # Uses XY speed only - don't punish vertical corrections
    nav_speed_penalty_weight = 0.1  # Penalty coefficient for speed above soft threshold
    nav_speed_penalty_threshold = 4.0  # XY speed above which penalty applies
    
    # Low-height penalty: asymmetric penalty for being below safe altitude
    # Makes low-altitude "farming" unprofitable without changing termination
    nav_low_height_penalty_floor = 0.7  # Below this height, penalty applies
    nav_low_height_penalty_weight = 3.0  # Quadratic penalty coefficient
    
    # =========================================================================
    # GOAL SAMPLING CONFIGURATION
    # =========================================================================
    goal_min_distance = 0.2  # Minimum distance from spawn (prevents overlap)
    goal_max_distance = 0.5  # Maximum distance (start small, can curriculum)
    goal_height = 1.0  # Goals at same height as spawn for now
    goal_reach_threshold = 0.1  # Within 10cm = success
    
    # Position error clipping for observations
    # CRITICAL: Must be >= goal_max_distance so policy can "see" goals
    obs_position_clip = 1.0  # Clip position error to ±1.0m
    obs_velocity_clip = 2.0  # Clip velocity to ±2.0 m/s
    
    # =========================================================================
    # INITIALIZATION (same as hover with small perturbations)
    # =========================================================================
    init_target_height = 1.0  # m - spawn height
    init_height_offset_min = -0.05  # Small height perturbation
    init_height_offset_max = 0.05
    init_max_xy_offset = 0.0  # Spawn at origin, goal is offset
    init_max_angle = 0.1  # Small angle perturbation (~5.7 deg)
    init_max_linear_velocity = 0.2  # Small initial velocity
    init_max_angular_velocity = 0.2
    init_guidance_probability = 0.2  # 20% spawn perfectly
    
    # =========================================================================
    # TERMINATION THRESHOLDS
    # =========================================================================
    term_xy_threshold = 2.0  # m - wider boundary for navigation
    
    # -------------------------------------------------------------------------
    # HEIGHT TERMINATION: Persistence-based for navigation feasibility
    # -------------------------------------------------------------------------
    # Problem: Instant termination at 0.3m kills exactly the behavior we need.
    # When drone tilts to move laterally, it momentarily loses vertical lift
    # and dips. Instant termination teaches "tilt = death" even though tilt
    # is required for motion.
    # Solution: Three zones (safe, soft, hard) with persistence for soft zone.
    # -------------------------------------------------------------------------
    term_z_soft_min: float = 0.25  # m - penalty + persistence
    term_z_hard_min: float = 0.10  # m - immediate (true crash)
    term_z_soft_max: float = 2.50  # m - penalty + persistence  
    term_z_hard_max: float = 3.00  # m - immediate (runaway climb)
    term_z_persistence_steps: int = 50  # Must be in soft zone for 50 steps (500ms)
    
    # -------------------------------------------------------------------------
    # TILT TERMINATION: Persistence-based for navigation feasibility
    # -------------------------------------------------------------------------
    # Hover-era tilt limits (0.5-0.8 rad) are too restrictive for point navigation.
    # Navigation requires sustained roll/pitch to translate horizontally, which
    # causes single-step tilt spikes that prematurely terminate otherwise valid
    # maneuvers. With instant termination at 0.8 rad, too_tilted caused 85-96%
    # of all terminations, bottlenecking learning.
    #
    # Solution: Persistence-based termination
    # - Soft threshold: Tilt must exceed this for N consecutive steps to terminate
    # - Hard threshold: Immediate termination (true safety cutoff for extreme tilt)
    # This allows transient spikes during aggressive maneuvers while catching
    # sustained dangerous tilts.
    # -------------------------------------------------------------------------
    # ITERATION 4: Even more aggressive relaxation needed.
    # At iter 50: tilt still 93%, p50=21 (need >51), p90=31 (need >154)
    # Random policy causes rapid tilt - need to allow recovery from extreme angles.
    # -------------------------------------------------------------------------
    # ITERATION 5: Rebalanced for navigation feasibility
    # Problem: 25 steps (250ms) persistence was too short for aggressive maneuvers.
    # Agent accelerates, tilts aggressively, needs 300-500ms to recover, gets killed.
    # Solution: Lower soft threshold (earlier penalty pressure), much longer persistence
    # (allows recovery), hard threshold still prevents flips.
    # -------------------------------------------------------------------------
    term_tilt_soft_threshold = 1.22  # rad (~70 deg) - persistence-based
    term_tilt_hard_threshold = 2.62  # rad (~150 deg) - immediate termination (near flip)
    term_tilt_persistence_steps = 50  # Must exceed soft threshold for 50 consecutive steps (500ms at 100Hz)
    
    # -------------------------------------------------------------------------
    # LINEAR VELOCITY TERMINATION: Persistence-based for navigation feasibility
    # -------------------------------------------------------------------------
    # Problem: Hard cutoff at 3 m/s causes 42% of terminations.
    # Agent can't learn to brake if it's killed immediately for going fast.
    # Solution: Persistence-based termination allows brief speed spikes.
    # -------------------------------------------------------------------------
    # ITERATION 6: Further relaxation for braking learning
    # Problem: linvel_exceeded is now 48% of terminations.
    # Agent overshoots, can't learn recovery/braking before termination.
    # Solution: Raise thresholds and increase persistence to 500ms.
    # -------------------------------------------------------------------------
    term_linear_velocity_soft_threshold: float = 4.0  # m/s - persistence-based
    term_linear_velocity_hard_threshold: float = 6.0  # m/s - immediate (true runaway)
    term_linear_velocity_persistence_steps: int = 50  # Must exceed soft for 50 steps (500ms)
    # -------------------------------------------------------------------------
    # ANGULAR VELOCITY TERMINATION: Further relaxed for navigation feasibility
    # -------------------------------------------------------------------------
    # History of adjustments:
    # - Original hover limit: 8.0 rad/s → caused 67-95% terminations
    # - First relaxation: 15.0 rad/s → still caused 77% terminations at iter 50
    # - Second: 25.0 rad/s with instant termination → still 70% at iter 50
    #
    # ITERATION 4: Persistence-based angular velocity termination
    # Same pattern as tilt: soft threshold with persistence, hard threshold immediate.
    # -------------------------------------------------------------------------
    term_angular_velocity_soft_threshold = 30.0  # rad/s - persistence-based
    term_angular_velocity_hard_threshold = 50.0  # rad/s - immediate (true runaway spin)
    term_angular_velocity_persistence_steps = 10  # Must exceed soft for 10 consecutive steps (100ms)
    
    # Domain randomization
    enable_disturbance = True
    disturbance_force_std = 0.0132  # N (mass * g / 20)
    disturbance_torque_std = 2.65e-5  # Nm
    
    # Action history
    action_history_length = 32
    
    # -------------------------------------------------------------------------
    # HOVER-CENTERED ACTION PARAMETERIZATION (Option A fix for feasibility)
    # -------------------------------------------------------------------------
    # Problem: Original mapping (actions+1)/2 * RPM_MAX means:
    #   - action=0 → 50% thrust (unstable, drone falls)
    #   - Random exploration creates massive thrust asymmetry
    #   - Drone tips over immediately and never recovers
    #
    # Solution: Center actions around hover thrust
    #   - action=0 → hover RPM (~33.4% of max)
    #   - action=±1 → hover ± action_scale * (max_rpm - hover_rpm)
    #   - Symmetric exploration around stable flight
    #
    # This is the standard L2F approach for training on unstable platforms.
    # -------------------------------------------------------------------------
    # ITERATION 1: action_scale=1.0 still caused 86% tilt terminations
    # Problem: PPO's initial log_std ≈ log(0.5) means σ ≈ 0.5
    # With action_scale=1.0, a 2σ deviation on one motor = ±1.0 action
    # That's still ±100% of hover-to-limit range → massive torque imbalance
    #
    # ITERATION 2: action_scale=0.3 (aggressive constraint)
    # Now a 2σ deviation = ±0.3 * 100% = ±30% of hover-to-limit range
    # This keeps early exploration much closer to hover
    # The policy can still learn to use the full range as std decreases
    # -------------------------------------------------------------------------
    use_hover_centered_actions: bool = True
    action_scale: float = 0.3  # Constrain early exploration to ±30% of available range


# ==============================================================================
# Environment Implementation
# ==============================================================================

class CrazyfliePointNavEnv(DirectRLEnv):
    """Crazyflie environment for point navigation with L2F physics."""
    
    cfg: CrazyfliePointNavEnvCfg
    
    def __init__(self, cfg: CrazyfliePointNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Cache physics parameters (IDENTICAL to train_hover.py)
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
        
        # Pre-compute mixer matrix (4 motors -> [roll, pitch, yaw] torques)
        rp = self._rotor_positions
        yd = self._rotor_yaw_dirs
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
        
        # =====================================================================
        # NAVIGATION-SPECIFIC STATE
        # =====================================================================
        # Goal positions (world frame)
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Previous distance to goal (for progress reward) - XY only
        self._prev_dist_xy = torch.zeros(self.num_envs, device=self.device)
        
        # Previous speed (for braking reward near goal)
        self._prev_speed = torch.zeros(self.num_envs, device=self.device)
        
        # Previous height deficit (for delta-based height recovery reward)
        self._prev_height_below_target = torch.zeros(self.num_envs, device=self.device)
        
        # Track if goal was reached this episode
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Episode statistics
        self._episode_sums = {
            "height_cost": torch.zeros(self.num_envs, device=self.device),
            "orientation_cost": torch.zeros(self.num_envs, device=self.device),
            "xy_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "z_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "angular_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "action_cost": torch.zeros(self.num_envs, device=self.device),
            "hover_reward": torch.zeros(self.num_envs, device=self.device),
            "progress_reward": torch.zeros(self.num_envs, device=self.device),
            "braking_reward": torch.zeros(self.num_envs, device=self.device),
            "speed_penalty": torch.zeros(self.num_envs, device=self.device),
            "reach_bonus": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "goal_reached": torch.zeros(self.num_envs, device=self.device),
            "final_distance": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Termination reason counters (for debugging)
        self._term_counters = {
            "xy_exceeded": 0,
            "too_low": 0,
            "too_high": 0,
            "too_tilted": 0,
            "lin_vel_exceeded": 0,
            "ang_vel_exceeded": 0,
            "goal_reached": 0,
            "timeout": 0,
            "total": 0,
        }
        
        # Episode length tracking for diagnostics
        self._episode_lengths = []  # Store completed episode lengths
        self._max_episode_buffer = 10000  # Keep last N episodes for stats
        
        # Tilt persistence counter for navigation-friendly termination
        # Only terminate when tilt exceeds soft threshold for N consecutive steps
        self._tilt_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Angular velocity persistence counter (same pattern as tilt)
        self._angvel_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Linear velocity persistence counter (same pattern as tilt)
        self._linvel_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Height violation counters for persistence-based termination
        self._height_low_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._height_high_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Get body ID for force application
        self._body_id = self._robot.find_bodies("body")[0]
        
        # Cache spawn position (env_origins + target height) to avoid recomputing every step
        self._spawn_pos = self._terrain.env_origins.clone()
        self._spawn_pos[:, 2] += self.cfg.init_target_height
        
        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
        # Print info
        self._print_env_info()
        
        # Verify invariants
        self._verify_invariants()
    
    def _verify_invariants(self):
        """Verify critical invariants are satisfied."""
        cfg = self.cfg
        
        # Invariant 1: Position clip must cover goal range
        assert cfg.obs_position_clip >= cfg.goal_max_distance, \
            f"obs_position_clip ({cfg.obs_position_clip}) must be >= goal_max_distance ({cfg.goal_max_distance})"
        
        # Invariant 2: Minimum goal distance must be > reach threshold (no trivial success)
        assert cfg.goal_min_distance > cfg.goal_reach_threshold, \
            f"goal_min_distance ({cfg.goal_min_distance}) must be > goal_reach_threshold ({cfg.goal_reach_threshold})"
        
        # Invariant 3: Spawn height must be within termination bounds
        assert cfg.term_z_hard_min < cfg.init_target_height < cfg.term_z_hard_max, \
            f"init_target_height ({cfg.init_target_height}) must be within [{cfg.term_z_hard_min}, {cfg.term_z_hard_max}]"
        
        # Invariant 3b: Height thresholds must be properly ordered
        assert cfg.term_z_hard_min < cfg.term_z_soft_min < cfg.init_target_height, \
            f"Height thresholds must be ordered: hard_min < soft_min < init_height"
        assert cfg.init_target_height < cfg.term_z_soft_max < cfg.term_z_hard_max, \
            f"Height thresholds must be ordered: init_height < soft_max < hard_max"
        
        # Invariant 4: Hover action is physically correct
        expected_hover = L2FConstants.hover_action()
        assert abs(expected_hover - 0.334) < 0.01, \
            f"hover_action ({expected_hover}) should be ~0.334"
        
        # Invariant 5: Motor alpha is in valid range
        assert 0 < self._motor_alpha < 1, \
            f"motor_alpha ({self._motor_alpha}) must be in (0, 1)"
        
        # Invariant 6: Tilt thresholds are properly ordered (soft < hard)
        assert cfg.term_tilt_soft_threshold < cfg.term_tilt_hard_threshold, \
            f"term_tilt_soft_threshold ({cfg.term_tilt_soft_threshold}) must be < term_tilt_hard_threshold ({cfg.term_tilt_hard_threshold})"
        
        # Invariant 7: Persistence steps must be > 0
        assert cfg.term_tilt_persistence_steps > 0, \
            f"term_tilt_persistence_steps ({cfg.term_tilt_persistence_steps}) must be > 0"
        
        # Invariant 8: Angular velocity thresholds are properly ordered (soft < hard)
        assert cfg.term_angular_velocity_soft_threshold < cfg.term_angular_velocity_hard_threshold, \
            f"term_angular_velocity_soft_threshold ({cfg.term_angular_velocity_soft_threshold}) must be < term_angular_velocity_hard_threshold ({cfg.term_angular_velocity_hard_threshold})"
        
        # Invariant 9: Angular velocity persistence steps must be > 0
        assert cfg.term_angular_velocity_persistence_steps > 0, \
            f"term_angular_velocity_persistence_steps ({cfg.term_angular_velocity_persistence_steps}) must be > 0"
        
        # Invariant 10: Linear velocity thresholds are properly ordered (soft < hard)
        assert cfg.term_linear_velocity_soft_threshold < cfg.term_linear_velocity_hard_threshold, \
            f"term_linear_velocity_soft_threshold ({cfg.term_linear_velocity_soft_threshold}) must be < term_linear_velocity_hard_threshold ({cfg.term_linear_velocity_hard_threshold})"
        
        # Invariant 11: Linear velocity persistence steps must be > 0
        assert cfg.term_linear_velocity_persistence_steps > 0, \
            f"term_linear_velocity_persistence_steps ({cfg.term_linear_velocity_persistence_steps}) must be > 0"
        
        # Invariant 12: Height persistence steps must be > 0
        assert cfg.term_z_persistence_steps > 0, \
            f"term_z_persistence_steps ({cfg.term_z_persistence_steps}) must be > 0"
        
        print("[INVARIANTS] All invariants verified ✓")
    
    def _print_env_info(self):
        """Print environment configuration."""
        print("\n" + "="*60)
        print("Crazyflie L2F Point Navigation Environment")
        print("="*60)
        print(f"  Physics dt:        {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Episode length:    {self.cfg.episode_length_s:.1f} s")
        print(f"  Num envs:          {self.num_envs}")
        print(f"  Observation dim:   {self.cfg.observation_space} (146 hover + 3 goal)")
        print(f"  Action dim:        {self.cfg.action_space}")
        print(f"  Mass:              {self._mass*1000:.1f} g")
        print(f"  Hover RPM:         {self._hover_rpm:.0f}")
        print(f"  Hover action:      {self._hover_action:.4f}")
        print(f"  Motor alpha:       {self._motor_alpha:.4f}")
        print("--- Navigation ---")
        print(f"  Goal distance:     [{self.cfg.goal_min_distance:.2f}, {self.cfg.goal_max_distance:.2f}] m")
        print(f"  Reach threshold:   {self.cfg.goal_reach_threshold:.2f} m")
        print(f"  Position clip:     ±{self.cfg.obs_position_clip:.1f} m")
        print("--- Tilt Termination (persistence-based) ---")
        print(f"  Soft threshold:    {self.cfg.term_tilt_soft_threshold:.2f} rad (~{math.degrees(self.cfg.term_tilt_soft_threshold):.0f}°) @ {self.cfg.term_tilt_persistence_steps} consecutive steps")
        print(f"  Hard threshold:    {self.cfg.term_tilt_hard_threshold:.2f} rad (~{math.degrees(self.cfg.term_tilt_hard_threshold):.0f}°) immediate")
        print("--- Angular Velocity Termination (persistence-based) ---")
        print(f"  Soft threshold:    {self.cfg.term_angular_velocity_soft_threshold:.1f} rad/s @ {self.cfg.term_angular_velocity_persistence_steps} consecutive steps")
        print(f"  Hard threshold:    {self.cfg.term_angular_velocity_hard_threshold:.1f} rad/s immediate")
        print("--- Linear Velocity Termination (persistence-based) ---")
        print(f"  Soft threshold:    {self.cfg.term_linear_velocity_soft_threshold:.1f} m/s @ {self.cfg.term_linear_velocity_persistence_steps} consecutive steps")
        print(f"  Hard threshold:    {self.cfg.term_linear_velocity_hard_threshold:.1f} m/s immediate")
        print("--- Height Termination (persistence-based) ---")
        print(f"  Low soft:          {self.cfg.term_z_soft_min:.2f} m @ {self.cfg.term_z_persistence_steps} consecutive steps")
        print(f"  Low hard:          {self.cfg.term_z_hard_min:.2f} m immediate")
        print(f"  High soft:         {self.cfg.term_z_soft_max:.2f} m @ {self.cfg.term_z_persistence_steps} consecutive steps")
        print(f"  High hard:         {self.cfg.term_z_hard_max:.2f} m immediate")
        print("--- Action Parameterization ---")
        if self.cfg.use_hover_centered_actions:
            print(f"  Mode:              HOVER-CENTERED (action=0 → hover thrust)")
            print(f"  Action scale:      {self.cfg.action_scale:.2f}")
            print(f"  Hover RPM:         {self._hover_rpm:.0f} ({self._hover_rpm/self._max_rpm*100:.1f}% of max)")
        else:
            print(f"  Mode:              RAW (action=0 → 50% thrust)")
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
        """Convert quaternion to flattened rotation matrix (9 elements).
        
        Isaac Lab uses [w,x,y,z] ordering (verified by checking quat[:,0] ~ 1 at reset).
        MUST match L2F observe_rotation_matrix() exactly.
        """
        # Isaac Lab quaternion ordering: [w, x, y, z]
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
    
    def _get_tilt_angle(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute tilt angle using rotation matrix (quaternion-order agnostic).
        
        This method is robust to quaternion ordering by computing the body Z axis
        in world frame and comparing to world up [0,0,1].
        
        tilt_angle = acos(body_z · world_up) = acos(R[2,2])
        """
        # Get rotation matrix (flattened: [r00,r01,r02,r10,r11,r12,r20,r21,r22])
        rot_matrix = self._quat_to_rotation_matrix(quat)
        
        # R[2,2] is the z-component of body z-axis in world frame
        # This is element index 8 in the flattened matrix
        r22 = rot_matrix[:, 8]
        
        # Tilt angle = acos(R[2,2])
        # Clamp to avoid numerical issues with acos
        cos_tilt = torch.clamp(r22, -1.0, 1.0)
        tilt_angle = torch.acos(cos_tilt)
        
        return tilt_angle
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions using L2F motor model.
        
        Action mapping depends on use_hover_centered_actions config:
        - If False: (actions+1)/2 * max_rpm (original, action=0 → 50% thrust)
        - If True:  hover_rpm + actions * scale * range (action=0 → hover thrust)
        """
        # Store and clamp actions
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        if self.cfg.use_hover_centered_actions:
            # HOVER-CENTERED: action=0 → hover, action=±1 → limits
            # This ensures exploration is symmetric around stable hover
            hover_rpm = self._hover_rpm
            # Available range: can go from 0 to max_rpm
            # For positive actions: range is (max_rpm - hover_rpm)
            # For negative actions: range is (hover_rpm - min_rpm)
            pos_range = (self._max_rpm - hover_rpm) * self.cfg.action_scale
            neg_range = (hover_rpm - self._min_rpm) * self.cfg.action_scale
            
            # Asymmetric mapping to allow full range
            target_rpm = torch.where(
                self._actions >= 0,
                hover_rpm + self._actions * pos_range,
                hover_rpm + self._actions * neg_range
            )
        else:
            # Original mapping: (actions+1)/2 * max_rpm
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
        """Construct observations: 146 hover dims + 3 goal-relative dims = 149.
        
        Layout:
        - [0:3]     Position error from spawn (clipped to ±obs_position_clip)
        - [3:12]    Rotation matrix (9 elements, row-major)
        - [12:15]   Linear velocity (clipped to ±obs_velocity_clip)
        - [15:18]   Angular velocity in body frame (radians/s)
        - [18:146]  Action history (32 * 4 = 128)
        - [146:149] Goal position relative to drone (clipped to ±obs_position_clip)
        """
        cfg = self.cfg
        
        # Get state
        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # Position relative to spawn origin (for compatibility with hover obs)
        pos_error = pos_w - self._spawn_pos
        pos_error_clipped = pos_error.clamp(-cfg.obs_position_clip, cfg.obs_position_clip)
        
        # Velocity (clipped)
        lin_vel_clipped = lin_vel_w.clamp(-cfg.obs_velocity_clip, cfg.obs_velocity_clip)
        
        # Rotation matrix
        rot_matrix = self._quat_to_rotation_matrix(quat_w)
        
        # Action history (flatten)
        action_history_flat = self._action_history.view(self.num_envs, -1)
        
        # Goal relative position (world frame, clipped)
        goal_relative = self._goal_pos - pos_w
        goal_relative_clipped = goal_relative.clamp(-cfg.obs_position_clip, cfg.obs_position_clip)
        
        # Concatenate (149 dims total)
        obs = torch.cat([
            pos_error_clipped,       # 3 (clipped)
            rot_matrix,              # 9
            lin_vel_clipped,         # 3 (clipped)
            ang_vel_b,               # 3
            action_history_flat,     # 128
            goal_relative_clipped,   # 3 (goal-relative position, clipped)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward: hover stability costs + navigation rewards.
        
        REWARD STRUCTURE:
        1. Hover stability (negative costs) - penalize instability
        2. Progress reward (dense) - reward moving toward goal
        3. Reach bonus (sparse) - reward reaching goal
        
        This layers navigation on top of hover stability.
        """
        cfg = self.cfg
        
        # Get state
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # =====================================================================
        # HOVER STABILITY COSTS (preserved from train_hover.py)
        # =====================================================================
        
        # Height cost: deviation from target height
        height_error = pos_w[:, 2] - self._spawn_pos[:, 2]
        height_cost = height_error ** 2
        
        # Orientation cost: 1 - qw² (deviation from upright)
        orientation_cost = 1.0 - quat[:, 0] ** 2
        
        # Velocity costs
        xy_velocity_cost = (lin_vel[:, :2] ** 2).sum(dim=-1)
        z_velocity_cost = lin_vel[:, 2] ** 2
        angular_velocity_cost = (ang_vel ** 2).sum(dim=-1)
        
        # Action cost: penalize deviation from hover action
        action_deviation = self._actions - self._hover_action
        action_cost = (action_deviation ** 2).sum(dim=-1)
        
        # Weighted hover cost
        hover_cost = (
            cfg.hover_height_weight * height_cost +
            cfg.hover_orientation_weight * orientation_cost +
            cfg.hover_xy_velocity_weight * xy_velocity_cost +
            cfg.hover_z_velocity_weight * z_velocity_cost +
            cfg.hover_angular_velocity_weight * angular_velocity_cost +
            cfg.hover_action_weight * action_cost
        )
        
        # Hover reward (positive when costs are low)
        hover_reward = -cfg.hover_reward_scale * hover_cost + cfg.hover_reward_constant
        hover_reward = hover_reward.clamp(0.0, cfg.hover_reward_constant)
        
        # =====================================================================
        # NAVIGATION REWARDS (new for pointnav)
        # =====================================================================
        
        # Progress reward: decrease in XY distance (dense)
        # FIXED: Use XY distance only - decouple navigation from altitude
        delta_xy = pos_w[:, :2] - self._goal_pos[:, :2]
        dist_xy = torch.norm(delta_xy, dim=-1)
        
        # Gate hover reward by distance to goal
        # Far from goal: hover_reward * 0.2 (can't farm stability reward)
        # Near goal: hover_reward * 1.0 (full reward for precision)
        gate = torch.clamp(1.0 - (dist_xy / cfg.hover_gate_radius), 0.0, 1.0)
        hover_gate = cfg.hover_gate_min + (1.0 - cfg.hover_gate_min) * gate
        hover_reward = hover_reward * hover_gate
        progress = self._prev_dist_xy - dist_xy  # positive when getting closer
        progress_reward = cfg.nav_progress_weight * progress
        
        # Update previous XY distance
        self._prev_dist_xy = dist_xy.clone()
        
        # Reach bonus: sparse reward for reaching goal (XY distance for planar nav)
        # Using XY distance so height deviation doesn't block reach
        just_reached = (dist_xy < cfg.goal_reach_threshold) & (~self._goal_reached)
        reach_bonus = just_reached.float() * cfg.nav_reach_bonus
        
        # Mark goals as reached
        self._goal_reached = self._goal_reached | (dist_xy < cfg.goal_reach_threshold)
        
        # Braking reward: reward for slowing down when near goal (XY speed)
        # This teaches "go fast → slow down → stabilize" behavior
        v_xy = torch.norm(lin_vel[:, :2], dim=-1)  # XY speed only
        near_goal = dist_xy < cfg.nav_braking_radius
        speed_reduction = self._prev_speed - v_xy  # positive when slowing down
        braking_reward = torch.where(
            near_goal,
            cfg.nav_braking_weight * speed_reduction,
            torch.zeros_like(speed_reduction)
        )
        
        # Update previous speed for next step
        self._prev_speed = v_xy.clone()
        
        # Height tracking reward: Gaussian centered at target height
        # Rewards staying near target, decays smoothly with distance
        height_error = pos_w[:, 2] - self._spawn_pos[:, 2]
        height_tracking_reward = cfg.nav_height_track_weight * torch.exp(-5.0 * height_error ** 2)
        
        # Height recovery reward: DELTA-based (reward for climbing, not for being low)
        # FIXED: Only reward when reducing height deficit, not for being low
        height_below = torch.relu(target_height - pos_w[:, 2])  # >= 0 when below target
        height_recovery = self._prev_height_below_target - height_below  # positive when climbing
        height_recovery_reward = cfg.nav_height_recovery_weight * torch.clamp(height_recovery, -0.5, 0.5)
        
        # Update previous height deficit
        self._prev_height_below_target = height_below.clone()
        
        # Speed penalty: penalize going too fast horizontally (XY speed only)
        # Don't punish vertical corrections
        speed_excess = torch.relu(v_xy - cfg.nav_speed_penalty_threshold)
        speed_penalty = -cfg.nav_speed_penalty_weight * speed_excess ** 2
        
        # Low-height penalty: asymmetric quadratic penalty for being below floor
        # Makes low-altitude farming unprofitable without changing termination
        height_above_ground = pos_w[:, 2] - self._terrain.env_origins[:, 2]
        low_margin = torch.relu(cfg.nav_low_height_penalty_floor - height_above_ground)
        low_height_penalty = -cfg.nav_low_height_penalty_weight * (low_margin ** 2)
        
        # =====================================================================
        # TOTAL REWARD
        # =====================================================================
        
        reward = hover_reward + progress_reward + braking_reward + height_tracking_reward + height_recovery_reward + speed_penalty + low_height_penalty + reach_bonus
        
        # Track stats
        self._episode_sums["height_cost"] += height_cost
        self._episode_sums["orientation_cost"] += orientation_cost
        self._episode_sums["xy_velocity_cost"] += xy_velocity_cost
        self._episode_sums["z_velocity_cost"] += z_velocity_cost
        self._episode_sums["angular_velocity_cost"] += angular_velocity_cost
        self._episode_sums["action_cost"] += action_cost
        self._episode_sums["hover_reward"] += hover_reward
        self._episode_sums["progress_reward"] += progress_reward
        self._episode_sums["braking_reward"] += braking_reward
        self._episode_sums["speed_penalty"] += speed_penalty
        self._episode_sums["reach_bonus"] += reach_bonus
        self._episode_sums["total_reward"] += reward
        self._episode_sums["goal_reached"] += just_reached.float()
        self._episode_sums["final_distance"] = dist_xy  # Overwrite with current (XY distance)
        
        return reward
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.
        
        Terminated: Safety violations (crash, tilt, etc.) OR goal reached
        Truncated: Episode timeout
        """
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
        
        # =====================================================================
        # HEIGHT CHECK: Persistence-based for navigation feasibility
        # =====================================================================
        # Problem: Instant termination kills learning. When drone tilts to move,
        # it momentarily loses lift and dips. Instant termination teaches
        # "tilt = death" even though tilt is required for motion.
        # Solution: Three zones (safe, soft, hard) with persistence for soft.
        # =====================================================================
        height = pos_w[:, 2] - self._terrain.env_origins[:, 2]
        
        # Hard thresholds - immediate termination
        too_low_hard = height < cfg.term_z_hard_min
        too_high_hard = height > cfg.term_z_hard_max
        
        # Soft thresholds - persistence-based
        too_low_soft = height < cfg.term_z_soft_min
        too_high_soft = height > cfg.term_z_soft_max
        
        # Update low-height persistence counter
        self._height_low_violation_counter = torch.where(
            too_low_soft,
            self._height_low_violation_counter + 1,
            torch.zeros_like(self._height_low_violation_counter)
        )
        
        # Update high-height persistence counter
        self._height_high_violation_counter = torch.where(
            too_high_soft,
            self._height_high_violation_counter + 1,
            torch.zeros_like(self._height_high_violation_counter)
        )
        
        # Terminate if in soft zone for too long
        too_low_persistence = self._height_low_violation_counter >= cfg.term_z_persistence_steps
        too_high_persistence = self._height_high_violation_counter >= cfg.term_z_persistence_steps
        
        # Combined height termination
        too_low = too_low_hard | too_low_persistence
        too_high = too_high_hard | too_high_persistence
        
        # =====================================================================
        # TILT CHECK: Persistence-based for navigation feasibility
        # =====================================================================
        # Navigation requires sustained roll/pitch that would trigger instant
        # termination under hover-era limits. We use a two-tier approach:
        # 1. Hard threshold: Immediate termination (extreme unsafe tilt)
        # 2. Soft threshold: Only terminate after N consecutive violations
        # This allows transient spikes during maneuvers while catching sustained tilts.
        # =====================================================================
        tilt_angle = self._get_tilt_angle(quat)
        
        # Hard threshold - immediate termination for extreme tilt (safety cutoff)
        hard_tilted = tilt_angle > cfg.term_tilt_hard_threshold
        
        # Soft threshold - persistence-based termination
        soft_tilted = tilt_angle > cfg.term_tilt_soft_threshold
        
        # Update persistence counter: increment if over soft threshold, reset if not
        self._tilt_violation_counter = torch.where(
            soft_tilted,
            self._tilt_violation_counter + 1,
            torch.zeros_like(self._tilt_violation_counter)
        )
        
        # Terminate if exceeded soft threshold for N consecutive steps
        persistence_tilted = self._tilt_violation_counter >= cfg.term_tilt_persistence_steps
        
        # Combined tilt termination: hard OR persisted soft
        too_tilted = hard_tilted | persistence_tilted
        
        # =====================================================================
        # LINEAR VELOCITY CHECK: Persistence-based, XY speed only
        # =====================================================================
        # FIXED: Use XY speed only - don't terminate on vertical corrections
        v_xy = torch.norm(lin_vel[:, :2], dim=-1)
        
        # Hard threshold - immediate termination for extreme horizontal speed
        hard_fast = v_xy > cfg.term_linear_velocity_hard_threshold
        
        # Soft threshold - persistence-based
        soft_fast = v_xy > cfg.term_linear_velocity_soft_threshold
        
        # Update persistence counter
        self._linvel_violation_counter = torch.where(
            soft_fast,
            self._linvel_violation_counter + 1,
            torch.zeros_like(self._linvel_violation_counter)
        )
        
        # Terminate if exceeded soft threshold for N consecutive steps
        persistence_fast = self._linvel_violation_counter >= cfg.term_linear_velocity_persistence_steps
        
        # Combined linear velocity termination
        lin_vel_exceeded = hard_fast | persistence_fast
        
        # =====================================================================
        # ANGULAR VELOCITY CHECK: Persistence-based (same pattern as tilt)
        # =====================================================================
        ang_vel_mag = torch.norm(ang_vel, dim=-1)
        
        # Hard threshold - immediate termination for extreme spin
        hard_spin = ang_vel_mag > cfg.term_angular_velocity_hard_threshold
        
        # Soft threshold - persistence-based
        soft_spin = ang_vel_mag > cfg.term_angular_velocity_soft_threshold
        
        # Update persistence counter
        self._angvel_violation_counter = torch.where(
            soft_spin,
            self._angvel_violation_counter + 1,
            torch.zeros_like(self._angvel_violation_counter)
        )
        
        # Terminate if exceeded soft threshold for N consecutive steps
        persistence_spin = self._angvel_violation_counter >= cfg.term_angular_velocity_persistence_steps
        
        # Combined angular velocity termination
        ang_vel_exceeded = hard_spin | persistence_spin
        
        # Safety terminations
        safety_terminated = xy_exceeded | too_low | too_high | too_tilted | lin_vel_exceeded | ang_vel_exceeded
        
        # Goal reached termination (success!)
        goal_terminated = self._goal_reached
        
        terminated = safety_terminated | goal_terminated
        
        # Update termination counters
        self._term_counters["xy_exceeded"] += xy_exceeded.sum().item()
        self._term_counters["too_low"] += too_low.sum().item()
        self._term_counters["too_high"] += too_high.sum().item()
        self._term_counters["too_tilted"] += too_tilted.sum().item()
        self._term_counters["lin_vel_exceeded"] += lin_vel_exceeded.sum().item()
        self._term_counters["ang_vel_exceeded"] += ang_vel_exceeded.sum().item()
        self._term_counters["goal_reached"] += goal_terminated.sum().item()
        self._term_counters["timeout"] += (time_out & ~terminated).sum().item()
        self._term_counters["total"] += (terminated | time_out).sum().item()
        
        return terminated, time_out
    
    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample goal positions with minimum distance enforcement.
        
        Uses polar coordinates to ensure goals are:
        1. At least goal_min_distance away from spawn
        2. At most goal_max_distance away from spawn
        3. At the specified goal height
        """
        n = len(env_ids)
        cfg = self.cfg
        
        # Sample distance uniformly in [min, max]
        distance = torch.empty(n, device=self.device).uniform_(
            cfg.goal_min_distance, cfg.goal_max_distance
        )
        
        # Sample angle uniformly in [0, 2π]
        angle = torch.empty(n, device=self.device).uniform_(0, 2 * math.pi)
        
        # Convert to XY offset
        x_offset = distance * torch.cos(angle)
        y_offset = distance * torch.sin(angle)
        
        # Goal position = spawn origin + offset
        goal = self._terrain.env_origins[env_ids].clone()
        goal[:, 0] += x_offset
        goal[:, 1] += y_offset
        goal[:, 2] = goal[:, 2] + cfg.goal_height  # Set goal height
        
        self._goal_pos[env_ids] = goal
        
        # Sanity check: verify minimum distance
        spawn_pos = self._terrain.env_origins[env_ids].clone()
        spawn_pos[:, 2] += cfg.init_target_height
        actual_dist = torch.norm(goal - spawn_pos, dim=-1)
        
        if (actual_dist < cfg.goal_min_distance - 0.01).any():
            print(f"[WARNING] Some goals too close: min_dist={actual_dist.min():.3f}")
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return
        
        # Record episode lengths before reset
        if len(env_ids) > 0:
            ep_lengths = self.episode_length_buf[env_ids].cpu().tolist()
            self._episode_lengths.extend(ep_lengths)
            # Keep buffer bounded
            if len(self._episode_lengths) > self._max_episode_buffer:
                self._episode_lengths = self._episode_lengths[-self._max_episode_buffer:]
        
        # Log stats before reset
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            extras = {}
            for key, values in self._episode_sums.items():
                if key in ["goal_reached", "final_distance"]:
                    # These are totals/final values, not averages
                    extras[f"Episode/{key}"] = torch.mean(values[env_ids]).item()
                else:
                    avg = torch.mean(values[env_ids]).item()
                    steps = self.episode_length_buf[env_ids].float().mean().item()
                    if steps > 0:
                        extras[f"Episode/{key}"] = avg / steps
            
            # Compute reach rate
            reach_count = self._goal_reached[env_ids].float().sum().item()
            extras["Episode/reach_rate"] = reach_count / len(env_ids)
            
            self.extras["log"] = extras
        
        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        n = len(env_ids)
        cfg = self.cfg
        
        # Sample new goals FIRST (before spawning drone)
        self._sample_goals(env_ids)
        
        # Reset goal-reached flag
        self._goal_reached[env_ids] = False
        
        # Guidance: spawn perfectly at target with some probability
        guidance_mask = torch.rand(n, device=self.device) < cfg.init_guidance_probability
        
        # Initialize position near target height with perturbations
        pos = torch.zeros(n, 3, device=self.device)
        
        # XY offset for non-guided envs (spawn at origin, not goal)
        pos[~guidance_mask, 0] = torch.empty((~guidance_mask).sum(), device=self.device).uniform_(
            -cfg.init_max_xy_offset, cfg.init_max_xy_offset
        )
        pos[~guidance_mask, 1] = torch.empty((~guidance_mask).sum(), device=self.device).uniform_(
            -cfg.init_max_xy_offset, cfg.init_max_xy_offset
        )
        
        # Height: target height + random offset
        height_offset = torch.empty(n, device=self.device).uniform_(
            cfg.init_height_offset_min, cfg.init_height_offset_max
        )
        height_offset[guidance_mask] = 0
        pos[:, 2] = cfg.init_target_height + height_offset
        pos = pos + self._terrain.env_origins[env_ids]
        
        # Sample orientation (small random quaternion)
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0  # Identity
        if cfg.init_max_angle > 0:
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
        
        # Initialize motor state to hover RPM
        self._rpm_state[env_ids] = self._hover_rpm
        
        # Initialize action history to hover action
        self._action_history[env_ids] = self._hover_action
        self._actions[env_ids] = self._hover_action
        
        # Initialize previous XY distance to goal
        delta_xy = self._goal_pos[env_ids, :2] - pos[:, :2]
        self._prev_dist_xy[env_ids] = torch.norm(delta_xy, dim=-1)
        
        # Initialize previous speed to zero (drone starts at rest)
        self._prev_speed[env_ids] = 0.0
        
        # Initialize previous height deficit to zero (starts at target height)
        self._prev_height_below_target[env_ids] = 0.0
        
        # Sample disturbances
        if cfg.enable_disturbance:
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_torque_std
        
        # Reset tilt violation counter for persistence-based termination
        self._tilt_violation_counter[env_ids] = 0
        
        # Reset angular velocity violation counter
        self._angvel_violation_counter[env_ids] = 0
        
        # Reset linear velocity violation counter
        self._linvel_violation_counter[env_ids] = 0
        
        # Reset height violation counters
        self._height_low_violation_counter[env_ids] = 0
        self._height_high_violation_counter[env_ids] = 0
        
        # Reset stats
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
    
    def get_episode_length_stats(self) -> dict:
        """Get episode length statistics for diagnostics."""
        if len(self._episode_lengths) < 10:
            return {"mean": 0, "p50": 0, "p90": 0, "count": len(self._episode_lengths)}
        
        lengths = torch.tensor(self._episode_lengths, dtype=torch.float32)
        return {
            "mean": lengths.mean().item(),
            "p50": lengths.median().item(),
            "p90": lengths.quantile(0.9).item(),
            "count": len(self._episode_lengths),
        }
    
    def clear_episode_stats(self):
        """Clear episode length buffer and termination counters."""
        self._episode_lengths = []
        for k in self._term_counters:
            self._term_counters[k] = 0
    
    def get_termination_diagnostics(self, max_episode_length: int) -> dict:
        """Get termination diagnostics for feasibility analysis.
        
        Returns:
            dict with:
            - Episode length percentiles (p50, p90)
            - Termination reason percentages
            - Feasibility assessment
        """
        # Episode length stats
        ep_stats = self.get_episode_length_stats()
        p50 = ep_stats["p50"]
        p90 = ep_stats["p90"]
        H = max_episode_length
        
        # Termination breakdown
        total = max(self._term_counters["total"], 1)  # Avoid div by zero
        term_pcts = {
            reason: 100.0 * count / total
            for reason, count in self._term_counters.items()
            if reason != "total"
        }
        
        # Feasibility assessment (success criteria from task)
        feasibility = {
            "p50_ratio": p50 / H if H > 0 else 0,  # Target: > 0.2
            "p90_ratio": p90 / H if H > 0 else 0,  # Target: > 0.6
            "too_tilted_pct": term_pcts.get("too_tilted", 0),  # Target: < 40%
            "timeout_pct": term_pcts.get("timeout", 0),  # Target: > 0%
            "is_feasible": (
                (p50 / H > 0.2 if H > 0 else False) and
                (p90 / H > 0.6 if H > 0 else False) and
                term_pcts.get("too_tilted", 100) < 40
            ),
        }
        
        return {
            "episode_lengths": ep_stats,
            "termination_pcts": term_pcts,
            "feasibility": feasibility,
        }
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization."""
        if debug_vis:
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)  # Larger for visibility
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self._goal_markers = VisualizationMarkers(marker_cfg)
        else:
            if hasattr(self, "_goal_markers"):
                self._goal_markers.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "_goal_markers"):
            self._goal_markers.visualize(self._goal_pos)


# ==============================================================================
# L2F-Compatible Actor Network (Extended for 149-dim observations)
# ==============================================================================

class L2FActorNetwork(nn.Module):
    """Actor network matching L2F architecture with extended observation space.
    
    Architecture: 149 -> 64 (tanh) -> 64 (tanh) -> 4 (tanh)
    """
    
    HOVER_ACTION = 2.0 * math.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) / 21702.0 - 1.0
    
    def __init__(self, obs_dim: int = 149, hidden_dim: int = 64, action_dim: int = 4, init_std: float = 0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Learnable log std - start higher for more exploration
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(init_std))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        hover_bias = math.atanh(max(-0.99, min(0.99, self.HOVER_ACTION)))
        nn.init.constant_(self.fc3.bias, hover_bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        return mean
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-1.0, 1.0)
    
    def get_action_and_log_prob(self, obs: torch.Tensor):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1.0, 1.0), log_prob


class L2FCriticNetwork(nn.Module):
    """Critic network matching L2F architecture."""
    
    def __init__(self, obs_dim: int = 149, hidden_dim: int = 64):
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
# PPO Agent with Observation Normalization (IDENTICAL to train_hover.py)
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
        obs_dim: int = 149,
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
        
        self.actor = L2FActorNetwork(obs_dim, 64, action_dim).to(device)
        self.critic = L2FCriticNetwork(obs_dim, 64).to(device)
        
        self.obs_normalizer = RunningMeanStd((obs_dim,), device=device)
        self.normalize_observations = True
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def normalize_obs(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.normalize_observations:
            return obs
        if update:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=False)
            return self.actor.get_action(obs_norm, deterministic)
    
    def get_action_and_value(self, obs: torch.Tensor):
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=True)
            action, log_prob = self.actor.get_action_and_log_prob(obs_norm)
            value = self.critic(obs_norm)
        return action, log_prob, value
    
    def get_value(self, obs: torch.Tensor):
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=False)
            return self.critic(obs_norm)
    
    def update(self, obs: torch.Tensor, actions: torch.Tensor,
               log_probs: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor,
               minibatch_size: int = 4096):
        obs = obs.detach()
        actions = actions.detach()
        log_probs = log_probs.detach()
        returns = returns.detach()
        advantages = advantages.detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        obs_norm = self.normalize_obs(obs, update=False)
        
        total_loss = 0.0
        num_samples = obs.shape[0]
        num_updates = 0
        
        for _ in range(self.epochs):
            indices = torch.randperm(num_samples, device=obs.device)
            
            for start in range(0, num_samples, minibatch_size):
                end = min(start + minibatch_size, num_samples)
                mb_idx = indices[start:end]
                
                mb_obs = obs_norm[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs = log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                
                mean = self.actor(mb_obs)
                std = torch.exp(self.actor.log_std)
                dist = torch.distributions.Normal(mean, std)
                
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = (new_log_probs - mb_log_probs).exp()
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()
                
                values = self.critic(mb_obs)
                value_loss = ((values - mb_returns) ** 2).mean()
                
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


def sanity_test(env: CrazyfliePointNavEnv, num_steps: int = 100):
    """Run a quick sanity test to verify environment works.
    
    Tests:
    1. Environment can reset
    2. Observations have correct shape
    3. Random actions don't crash
    4. Goals are within expected bounds
    5. Rewards are finite
    """
    print("\n" + "="*60)
    print("SANITY TEST MODE")
    print("="*60)
    
    # Test 1: Reset
    print("[Test 1] Reset environment...")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"  ✓ Reset successful")
    
    # Test 2: Observation shape
    print("[Test 2] Check observation shape...")
    expected_dim = env.cfg.observation_space
    actual_dim = obs.shape[-1]
    assert actual_dim == expected_dim, f"Expected {expected_dim} dims, got {actual_dim}"
    print(f"  ✓ Observation shape correct: {obs.shape}")
    
    # Test 3: Goal distances
    print("[Test 3] Check goal distances...")
    spawn_pos = env._terrain.env_origins.clone()
    spawn_pos[:, 2] += env.cfg.init_target_height
    goal_dist = torch.norm(env._goal_pos - spawn_pos, dim=-1)
    min_dist = goal_dist.min().item()
    max_dist = goal_dist.max().item()
    assert min_dist >= env.cfg.goal_min_distance - 0.01, f"Goals too close: {min_dist}"
    assert max_dist <= env.cfg.goal_max_distance + 0.01, f"Goals too far: {max_dist}"
    print(f"  ✓ Goal distances in range: [{min_dist:.3f}, {max_dist:.3f}] m")
    
    # Test 4: Random steps
    print(f"[Test 4] Run {num_steps} random steps...")
    total_reward = 0.0
    for step in range(num_steps):
        action = torch.rand(env.num_envs, 4, device=env.device) * 2 - 1  # Random in [-1, 1]
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        # Check rewards are finite
        assert torch.isfinite(reward).all(), f"Non-finite reward at step {step}"
        total_reward += reward.mean().item()
    
    avg_reward = total_reward / num_steps
    print(f"  ✓ {num_steps} steps completed, avg reward: {avg_reward:.3f}")
    
    # Test 5: Check observation components
    print("[Test 5] Check observation components...")
    obs = obs_dict["policy"]
    
    # Position error (first 3 dims) should be clipped
    pos_error = obs[:, :3]
    assert (pos_error.abs() <= env.cfg.obs_position_clip + 0.01).all(), "Position error not clipped"
    
    # Goal relative (last 3 dims) should be clipped
    goal_rel = obs[:, -3:]
    assert (goal_rel.abs() <= env.cfg.obs_position_clip + 0.01).all(), "Goal relative not clipped"
    
    print(f"  ✓ Observation components verified")
    
    print("\n" + "="*60)
    print("ALL SANITY TESTS PASSED ✓")
    print("="*60 + "\n")


def train(env: CrazyfliePointNavEnv, agent: L2FPPOAgent, args):
    """Main training loop."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_pointnav")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    steps_per_rollout = 256  # Longer rollouts for navigation
    num_envs = env.num_envs
    
    best_reward = float("-inf")
    best_reach_rate = 0.0
    
    print(f"\n{'='*60}")
    print("Starting L2F Point Navigation PPO Training")
    print(f"{'='*60}")
    print(f"  Environments:       {num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {steps_per_rollout}")
    print(f"  Total batch size:   {steps_per_rollout * num_envs}")
    print(f"  Observation dim:    {env.cfg.observation_space}")
    print(f"  Action dim:         {env.cfg.action_space}")
    print(f"  Goal distance:      [{env.cfg.goal_min_distance}, {env.cfg.goal_max_distance}] m")
    print(f"{'='*60}\n")
    
    # Verify quaternion ordering at first reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    quat = env._robot.data.root_quat_w
    print(f"[Quaternion Check] At reset: quat[:,0].mean={quat[:,0].mean():.4f}, quat[:,3].mean={quat[:,3].mean():.4f}")
    print(f"  -> If quat[:,0] ~ 1.0, ordering is [w,x,y,z]. If quat[:,3] ~ 1.0, ordering is [x,y,z,w].")
    
    # Resume from checkpoint if requested
    start_iteration = 0
    if args.resume:
        # Find checkpoint to load
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            # Find latest checkpoint
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_dir, "final_model.pt")
        
        if os.path.exists(ckpt_path):
            print(f"\n[Resume] Loading checkpoint: {ckpt_path}")
            start_iteration, best_reward = agent.load(ckpt_path)
            print(f"[Resume] Starting from iteration {start_iteration}, best_reward={best_reward:.2f}")
        else:
            print(f"\n[Resume] No checkpoint found at {ckpt_path}, starting fresh")
    
    for iteration in range(start_iteration, start_iteration + args.max_iterations):
        # Collect rollout - DON'T reset all envs every iteration!
        # Let episodes continue naturally, only reset when done
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []
        
        episode_rewards = torch.zeros(num_envs, device=env.device)
        reach_count = 0
        episode_count = 0
        
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
            
            # Count reaches
            if "log" in env.extras and "Episode/reach_rate" in env.extras["log"]:
                reach_count += env.extras["log"]["Episode/reach_rate"] * done.sum().item()
                episode_count += done.sum().item()
            
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
        reach_rate = reach_count / max(episode_count, 1) if episode_count > 0 else 0.0
        
        # Check for best model
        is_best = mean_reward > best_reward
        if is_best:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward)
        
        if reach_rate > best_reach_rate:
            best_reach_rate = reach_rate
            agent.save(os.path.join(checkpoint_dir, "best_reach_model.pt"), iteration, best_reward)
        
        # Log progress
        if iteration % 10 == 0 or is_best:
            std = torch.exp(agent.actor.log_std).mean().item()
            star = " *BEST*" if is_best else ""
            print(f"[Iter {iteration:4d}] Reward: {mean_reward:8.3f} | Reach: {reach_rate*100:5.1f}% | Std: {std:.3f} | Loss: {loss:.4f}{star}")
        
        # Print comprehensive diagnostics every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            # Episode length stats
            ep_stats = env.get_episode_length_stats()
            H = steps_per_rollout  # Horizon
            
            # Determine training phase
            p50 = ep_stats["p50"]
            p90 = ep_stats["p90"]
            if p50 < 10:
                phase = "Phase 0: NOT LEARNABLE"
            elif p50 < 0.2 * H:
                phase = "Phase 1: Barely learnable"
            elif p50 < 0.5 * H:
                phase = "Phase 2: Learnable"
            else:
                phase = "Phase 3: Healthy training"
            
            print(f"\n  === DIAGNOSTICS (H={H}) ===")
            print(f"  Episode Length: mean={ep_stats['mean']:.1f} p50={p50:.1f} p90={p90:.1f} (n={ep_stats['count']})")
            print(f"  Phase: {phase}")
            print(f"  Targets: Phase2 needs p50>{0.2*H:.0f}, p90>{0.6*H:.0f} | Phase3 needs p50>{0.6*H:.0f}")
            
            # Termination breakdown
            tc = env._term_counters
            total = max(tc["total"], 1)
            print(f"  Terminations: xy:{tc['xy_exceeded']/total*100:.1f}% low:{tc['too_low']/total*100:.1f}% "
                  f"high:{tc['too_high']/total*100:.1f}% tilt:{tc['too_tilted']/total*100:.1f}% "
                  f"linvel:{tc['lin_vel_exceeded']/total*100:.1f}% angvel:{tc['ang_vel_exceeded']/total*100:.1f}% "
                  f"goal:{tc['goal_reached']/total*100:.1f}% timeout:{tc['timeout']/total*100:.1f}%")
            
            # Actionable guidance
            max_term = max(tc["xy_exceeded"], tc["too_low"], tc["too_high"], 
                          tc["too_tilted"], tc["lin_vel_exceeded"], tc["ang_vel_exceeded"])
            if max_term / total > 0.4:
                dominant = max(tc.items(), key=lambda x: x[1] if x[0] != "total" else 0)[0]
                print(f"  WARNING: '{dominant}' dominates ({max_term/total*100:.0f}%). Consider relaxing threshold.")
            
            print(f"  ==========================\n")
            
            # Clear stats for next window
            env.clear_episode_stats()
        
        # Save checkpoint periodically
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward)
    
    # Save final model
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), args.max_iterations, best_reward)
    print(f"\nTraining complete! Best reward: {best_reward:.3f}, Best reach rate: {best_reach_rate*100:.1f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def play(env: CrazyfliePointNavEnv, agent: L2FPPOAgent, checkpoint_path: str):
    """Run trained policy with visualization and data logging."""
    
    iteration, best_reward = agent.load(checkpoint_path)
    print(f"\n[Play Mode] Loaded checkpoint from iteration {iteration}")
    print(f"[Play Mode] Best training reward: {best_reward:.3f}")
    print("[Play Mode] Press Ctrl+C to stop\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    step_count = 0
    episode_reward = 0.0
    reach_count = 0
    episode_count = 0
    
    # Initialize flight data logger
    logger = FlightDataLogger()
    
    # Create eval directory structure with timestamp
    run_tag = int(time.time())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "eval", "pointnav", f"pointnav_{run_tag}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # File paths for periodic saving (overwrite each time)
    csv_filename = os.path.join(eval_dir, "pointnav_eval_latest.csv")
    title_prefix = "Point Navigation Evaluation"
    
    try:
        while simulation_app.is_running():
            action = agent.get_action(obs, deterministic=True)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            
            done = terminated | truncated
            episode_reward += reward.mean().item()
            step_count += 1
            
            # Log flight data
            logger.log_step(env, env_idx=0)
            
            # Track reaches
            if done.any():
                reaches = env._goal_reached[done].sum().item()
                reach_count += reaches
                episode_count += done.sum().item()
            
            # Save every 500 steps
            if step_count % 500 == 0:
                reach_rate = reach_count / max(episode_count, 1) * 100
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | Reach rate: {reach_rate:.1f}% | Saving...")
                logger.save_and_plot(csv_filename, title_prefix=title_prefix, output_dir=eval_dir)
            elif step_count % 100 == 0:
                reach_rate = reach_count / max(episode_count, 1) * 100
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | Reach rate: {reach_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n[Play Mode] Stopped by user")
        reach_rate = reach_count / max(episode_count, 1) * 100
        print(f"Final reach rate: {reach_rate:.1f}% ({reach_count}/{episode_count})")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    # Create config
    cfg = CrazyfliePointNavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    # Create environment
    env = CrazyfliePointNavEnv(cfg)
    
    if args.sanity_test:
        # Sanity test mode
        sanity_test(env)
        env.close()
        simulation_app.close()
        return
    
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
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_pointnav")
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
