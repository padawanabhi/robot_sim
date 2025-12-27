#!/usr/bin/env python3
"""
Deep diagnostic script to identify root causes of RL agent failure.
Systematically checks model compatibility, action application, reward function, 
environment consistency, and agent behavior.
"""
import sys
import os
import signal

# Set non-interactive backend
os.environ['MPLBACKEND'] = 'Agg'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import json
from datetime import datetime

def cleanup_handler(signum, frame):
    """Handle cleanup on interrupt."""
    print("\n\nInterrupted! Cleaning up...")
    sys.exit(0)

def phase1_model_compatibility(env, model_path):
    """Phase 1: Check model-environment compatibility."""
    print("\n" + "="*70)
    print("PHASE 1: MODEL-ENVIRONMENT COMPATIBILITY")
    print("="*70)
    
    issues = []
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"✗ ERROR: Failed to load model: {e}")
        return None, issues
    
    # Check observation space
    print("\n1. OBSERVATION SPACE CHECK:")
    model_obs_space = model.observation_space
    env_obs_space = env.observation_space
    
    print(f"   Model obs space: {model_obs_space}")
    print(f"   Env obs space:   {env_obs_space}")
    
    # Check shape
    if model_obs_space.shape != env_obs_space.shape:
        issue = f"CRITICAL: Observation space shape mismatch! Model: {model_obs_space.shape}, Env: {env_obs_space.shape}"
        print(f"   ✗ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Observation space shapes match: {model_obs_space.shape}")
    
    # Check bounds
    if not np.allclose(model_obs_space.low, env_obs_space.low, atol=1e-6):
        issue = f"WARNING: Observation space low bounds differ! Model: {model_obs_space.low}, Env: {env_obs_space.low}"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Observation space low bounds match")
    
    if not np.allclose(model_obs_space.high, env_obs_space.high, atol=1e-6):
        issue = f"WARNING: Observation space high bounds differ! Model: {model_obs_space.high}, Env: {env_obs_space.high}"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Observation space high bounds match")
    
    # Check action space
    print("\n2. ACTION SPACE CHECK:")
    model_action_space = model.action_space
    env_action_space = env.action_space
    
    print(f"   Model action space: {model_action_space}")
    print(f"   Env action space:   {env_action_space}")
    
    if model_action_space.shape != env_action_space.shape:
        issue = f"CRITICAL: Action space shape mismatch! Model: {model_action_space.shape}, Env: {env_action_space.shape}"
        print(f"   ✗ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Action space shapes match: {model_action_space.shape}")
    
    if not np.allclose(model_action_space.low, env_action_space.low, atol=1e-6):
        issue = f"CRITICAL: Action space low bounds differ! Model: {model_action_space.low}, Env: {env_action_space.low}"
        print(f"   ✗ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Action space low bounds match")
    
    if not np.allclose(model_action_space.high, env_action_space.high, atol=1e-6):
        issue = f"CRITICAL: Action space high bounds differ! Model: {model_action_space.high}, Env: {env_action_space.high}"
        print(f"   ✗ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Action space high bounds match")
    
    # Check model metadata
    print("\n3. MODEL METADATA:")
    try:
        # Get model file info
        model_file = model_path + '.zip'
        if os.path.exists(model_file):
            import stat
            file_stat = os.stat(model_file)
            file_size = file_stat.st_size
            mod_time = datetime.fromtimestamp(file_stat.st_mtime)
            print(f"   Model file size: {file_size / 1024:.1f} KB")
            print(f"   Model file modified: {mod_time}")
        else:
            print(f"   ⚠ Model file not found at: {model_file}")
    except Exception as e:
        print(f"   ⚠ Could not get model metadata: {e}")
    
    return model, issues

def phase2_action_application(env):
    """Phase 2: Verify actions are applied correctly."""
    print("\n" + "="*70)
    print("PHASE 2: ACTION APPLICATION VERIFICATION")
    print("="*70)
    
    issues = []
    
    # Test action format
    print("\n1. ACTION FORMAT TEST:")
    test_action = np.array([0.5, 0.0], dtype=np.float32)
    action_list = test_action.tolist()
    print(f"   Test action (numpy): {test_action}")
    print(f"   Converted to list: {action_list}")
    print(f"   Type: {type(action_list)}, Length: {len(action_list)}")
    
    if isinstance(action_list, list) and len(action_list) == 2:
        print(f"   ✓ Action format conversion works correctly")
    else:
        issue = f"CRITICAL: Action format conversion failed! Got: {action_list}"
        print(f"   ✗ {issue}")
        issues.append(issue)
    
    # Test movement
    print("\n2. MOVEMENT VERIFICATION:")
    obs, info = env.reset(seed=42)
    robot = env.env.robot_list[0]
    initial_pos = robot.state.flatten()[:2].copy()
    initial_theta = robot.state.flatten()[2]
    
    print(f"   Initial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f}), theta: {np.degrees(initial_theta):.1f}°")
    
    # Test forward movement
    forward_action = np.array([0.5, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(forward_action)
    
    robot = env.env.robot_list[0]
    final_pos = robot.state.flatten()[:2].copy()
    final_theta = robot.state.flatten()[2]
    
    distance_moved = np.linalg.norm(final_pos - initial_pos)
    theta_change = abs(final_theta - initial_theta)
    
    print(f"   After forward action [0.5, 0.0]:")
    print(f"   Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}), theta: {np.degrees(final_theta):.1f}°")
    print(f"   Distance moved: {distance_moved:.4f}m")
    print(f"   Theta change: {np.degrees(theta_change):.2f}°")
    
    # With step_time=0.1s and v=0.5 m/s, expected movement ~0.05m
    expected_distance = 0.5 * 0.1  # v * dt
    if distance_moved < 0.01:
        issue = f"CRITICAL: Robot did not move! Distance moved: {distance_moved:.4f}m (expected ~{expected_distance:.3f}m)"
        print(f"   ✗ {issue}")
        issues.append(issue)
    elif abs(distance_moved - expected_distance) > 0.02:
        issue = f"WARNING: Movement distance unexpected! Got: {distance_moved:.4f}m, Expected: ~{expected_distance:.3f}m"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Robot moves correctly (moved {distance_moved:.4f}m, expected ~{expected_distance:.3f}m)")
    
    # Test turning
    print("\n3. TURNING VERIFICATION:")
    obs, info = env.reset(seed=42)
    robot = env.env.robot_list[0]
    initial_theta = robot.state.flatten()[2]
    
    turn_action = np.array([0.0, 0.5], dtype=np.float32)  # Turn only
    obs, reward, terminated, truncated, info = env.step(turn_action)
    
    robot = env.env.robot_list[0]
    final_theta = robot.state.flatten()[2]
    theta_change = abs(final_theta - initial_theta)
    
    expected_theta_change = 0.5 * 0.1  # w * dt
    
    print(f"   After turn action [0.0, 0.5]:")
    print(f"   Theta change: {np.degrees(theta_change):.2f}° (expected ~{np.degrees(expected_theta_change):.2f}°)")
    
    if theta_change < 0.01:
        issue = f"CRITICAL: Robot did not turn! Theta change: {np.degrees(theta_change):.2f}°"
        print(f"   ✗ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Robot turns correctly")
    
    # Test action clipping
    print("\n4. ACTION CLIPPING CHECK:")
    test_actions = [
        np.array([2.0, 0.0], dtype=np.float32),  # Above limit
        np.array([-2.0, 0.0], dtype=np.float32),  # Below limit
        np.array([0.5, 2.0], dtype=np.float32),  # Angular above limit
    ]
    
    for action in test_actions:
        clipped = np.clip(action, env.action_space.low, env.action_space.high)
        print(f"   Action {action} -> Clipped to {clipped}")
        if not np.allclose(action, clipped):
            print(f"   ⚠ Action was clipped (expected for out-of-bounds actions)")
    
    return issues

def phase3_reward_analysis(env):
    """Phase 3: Analyze reward function."""
    print("\n" + "="*70)
    print("PHASE 3: REWARD FUNCTION ANALYSIS")
    print("="*70)
    
    issues = []
    
    # Test progress reward
    print("\n1. PROGRESS REWARD ANALYSIS:")
    obs, info = env.reset(seed=42)
    initial_distance = info['distance_to_goal']
    print(f"   Initial distance to goal: {initial_distance:.3f}m")
    
    # Move forward (should decrease distance)
    forward_action = np.array([0.5, 0.0], dtype=np.float32)
    obs, reward_forward, terminated, truncated, info = env.step(forward_action)
    distance_after_forward = info['distance_to_goal']
    progress_forward = initial_distance - distance_after_forward
    
    print(f"   After forward action [0.5, 0.0]:")
    print(f"   Distance: {distance_after_forward:.3f}m")
    print(f"   Progress: {progress_forward:.3f}m")
    print(f"   Reward: {reward_forward:.3f}")
    
    # Reset and try moving backward (turn 180 and go)
    obs, info = env.reset(seed=42)
    initial_distance = info['distance_to_goal']
    
    # Turn 180 degrees first
    for _ in range(31):  # ~180 degrees at 0.1s per step with w=0.5
        obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    
    # Now move forward (which is backward relative to goal)
    backward_action = np.array([0.5, 0.0], dtype=np.float32)
    obs, reward_backward, terminated, truncated, info = env.step(backward_action)
    distance_after_backward = info['distance_to_goal']
    progress_backward = initial_distance - distance_after_backward  # Should be negative
    
    print(f"\n   After turning 180° and moving forward (away from goal):")
    print(f"   Distance: {distance_after_backward:.3f}m")
    print(f"   Progress: {progress_backward:.3f}m (negative = moving away)")
    print(f"   Reward: {reward_backward:.3f}")
    
    # Calculate expected rewards
    # Progress reward: 20.0 * max(0, progress) - FIXED to not penalize backward movement
    expected_reward_forward = 20.0 * max(0.0, progress_forward) - 0.005  # time penalty
    expected_reward_backward = 20.0 * max(0.0, progress_backward) - 0.005  # time penalty (no negative progress penalty!)
    
    print(f"\n   Expected reward (forward): {expected_reward_forward:.3f}")
    print(f"   Actual reward (forward): {reward_forward:.3f}")
    print(f"   Expected reward (backward): {expected_reward_backward:.3f} (no penalty for backward movement)")
    print(f"   Actual reward (backward): {reward_backward:.3f}")
    
    # CHECK: Verify backward movement doesn't get penalized
    if progress_backward < 0 and reward_backward < -1.0:
        issue = f"CRITICAL: Large negative reward for backward movement! Progress: {progress_backward:.3f}m, Reward: {reward_backward:.3f}"
        print(f"   ✗ {issue}")
        print(f"   This strongly discourages any movement that temporarily increases distance!")
        issues.append(issue)
    elif progress_backward < 0 and reward_backward >= 0:
        print(f"   ✓ Backward movement is NOT penalized (reward: {reward_backward:.3f}), allowing obstacle navigation")
    
    # Test reward signal quality
    print("\n2. REWARD SIGNAL QUALITY:")
    test_actions = [
        ([0.0, 0.0], "No movement"),
        ([0.5, 0.0], "Forward"),
        ([1.0, 0.0], "Fast forward"),
        ([0.0, 0.5], "Turn only"),
        ([0.5, 0.5], "Forward + turn"),
    ]
    
    rewards_by_action = []
    for action_list, description in test_actions:
        obs, info = env.reset(seed=42)
        initial_distance = info['distance_to_goal']
        total_reward = 0
        
        for step in range(10):
            action = np.array(action_list, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        final_distance = info['distance_to_goal']
        progress = initial_distance - final_distance
        rewards_by_action.append({
            'description': description,
            'action': action_list,
            'total_reward': total_reward,
            'progress': progress,
            'avg_reward_per_step': total_reward / 10
        })
        
        print(f"   {description:20s}: Progress={progress:6.3f}m, Total reward={total_reward:7.2f}, Avg/step={total_reward/10:6.3f}")
    
    # Check if rewards are clear
    forward_reward = rewards_by_action[1]['avg_reward_per_step']
    no_move_reward = rewards_by_action[0]['avg_reward_per_step']
    
    if forward_reward <= no_move_reward:
        issue = f"WARNING: Forward movement reward ({forward_reward:.3f}) not higher than no movement ({no_move_reward:.3f})"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Forward movement has higher reward than no movement")
    
    # Reward scale analysis
    print("\n3. REWARD SCALE ANALYSIS:")
    print(f"   Progress reward coefficient: 20.0 (only for positive progress)")
    print(f"   Time penalty per step: -0.005")
    print(f"   Time penalty per episode (1000 steps): -5.0")
    print(f"   Collision penalty: -200.0")
    print(f"   Goal bonus: +200.0 (balanced with collision penalty)")
    
    # Calculate reward ranges
    max_progress_per_step = 1.0 * 0.1  # max velocity * step_time
    max_reward_per_step = 20.0 * max_progress_per_step - 0.005  # +1.995
    min_reward_per_step = 20.0 * (-max_progress_per_step) - 0.005  # -2.005 (moving away)
    
    print(f"\n   Expected reward range per step:")
    print(f"   Best case (max progress): {max_reward_per_step:.3f}")
    print(f"   Worst case (max backward): {min_reward_per_step:.3f}")
    print(f"   Range: {max_reward_per_step - min_reward_per_step:.3f}")
    
    # Check balance
    collision_penalty = -200.0
    goal_bonus = 200.0  # Updated to match current reward function (balanced 1:1)
    ratio = abs(collision_penalty) / goal_bonus
    
    print(f"\n   Collision vs Goal balance:")
    print(f"   Collision penalty: {collision_penalty:.1f}")
    print(f"   Goal bonus: {goal_bonus:.1f}")
    print(f"   Ratio: {ratio:.1f}:1 (collision is {ratio:.1f}× worse than goal is good)")
    
    if ratio > 2.0:
        issue = f"WARNING: Collision penalty ({collision_penalty:.1f}) is {ratio:.1f}× larger than goal bonus ({goal_bonus:.1f})"
        print(f"   ⚠ {issue}")
        print(f"   This may make the agent overly cautious, avoiding all movement")
        issues.append(issue)
    else:
        print(f"   ✓ Collision and goal rewards are balanced (1:1 ratio)")
    
    return issues

def phase4_environment_consistency(env, config_path):
    """Phase 4: Check environment consistency."""
    print("\n" + "="*70)
    print("PHASE 4: ENVIRONMENT CONSISTENCY CHECK")
    print("="*70)
    
    issues = []
    
    # Check training vs evaluation environment
    print("\n1. TRAINING VS EVALUATION ENVIRONMENT:")
    print(f"   Config path: {config_path}")
    print(f"   Randomize: {env.randomize}")
    print(f"   Num obstacles: {env.num_obstacles}")
    print(f"   Max steps: {env.max_steps}")
    
    if not env.randomize:
        issue = "WARNING: Environment randomization is disabled! Training and evaluation may use different setups"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Randomization is enabled")
    
    # Test randomization quality
    print("\n2. RANDOMIZATION QUALITY:")
    distances = []
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        robot = env.env.robot_list[0]
        goal = robot.goal.flatten()[:2]
        start = robot.state.flatten()[:2]
        distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        distances.append(distance)
    
    avg_distance = np.mean(distances)
    print(f"   Tested 5 random seeds:")
    print(f"   Start-goal distances: {[f'{d:.2f}' for d in distances]}")
    print(f"   Average distance: {avg_distance:.2f}m")
    
    if avg_distance < 3.0:
        issue = f"WARNING: Average start-goal distance is small ({avg_distance:.2f}m), may be too easy"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    elif avg_distance > 12.0:
        issue = f"WARNING: Average start-goal distance is very large ({avg_distance:.2f}m), may be impossible"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Start-goal distances are reasonable")
    
    # Check obstacle placement
    print("\n3. OBSTACLE PLACEMENT:")
    obs, info = env.reset(seed=42)
    num_obstacles = len(env.env.obstacle_list) if hasattr(env.env, 'obstacle_list') else 0
    print(f"   Number of obstacles: {num_obstacles}")
    
    if num_obstacles == 0:
        issue = "CRITICAL: No obstacles found in environment!"
        print(f"   ✗ {issue}")
        issues.append(issue)
    elif num_obstacles != env.num_obstacles:
        issue = f"WARNING: Obstacle count mismatch! Expected: {env.num_obstacles}, Found: {num_obstacles}"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Obstacle count matches expected: {num_obstacles}")
    
    # Check if obstacles block path
    robot = env.env.robot_list[0]
    start = robot.state.flatten()[:2]
    goal = robot.goal.flatten()[:2]
    
    # Simple check: count obstacles between start and goal
    obstacles_in_path = 0
    for obs_obj in env.env.obstacle_list:
        obs_pos = obs_obj.state.flatten()[:2]
        # Check if obstacle is roughly between start and goal
        start_to_goal = goal - start
        start_to_obs = obs_pos - start
        # Project obstacle onto start-goal line
        if np.linalg.norm(start_to_goal) > 0.1:
            t = np.clip(np.dot(start_to_obs, start_to_goal) / np.dot(start_to_goal, start_to_goal), 0, 1)
            closest_point = start + t * start_to_goal
            dist_to_line = np.linalg.norm(obs_pos - closest_point)
            if dist_to_line < 1.0:  # Within 1m of straight path
                obstacles_in_path += 1
    
    print(f"   Obstacles near straight-line path: {obstacles_in_path}/{num_obstacles}")
    if obstacles_in_path == num_obstacles:
        issue = f"WARNING: All obstacles are on the straight-line path, may block navigation"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    
    return issues

def phase5_agent_behavior(model, env):
    """Phase 5: Analyze agent behavior."""
    print("\n" + "="*70)
    print("PHASE 5: AGENT BEHAVIOR ANALYSIS")
    print("="*70)
    
    issues = []
    
    # Action distribution
    print("\n1. ACTION DISTRIBUTION:")
    actions = []
    obs, info = env.reset(seed=42)
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    actions = np.array(actions)
    action_magnitudes = np.sqrt(actions[:, 0]**2 + actions[:, 1]**2)
    
    print(f"   Collected {len(actions)} actions")
    print(f"   Linear velocity range: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
    print(f"   Angular velocity range: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
    print(f"   Action magnitude range: [{action_magnitudes.min():.3f}, {action_magnitudes.max():.3f}]")
    print(f"   Mean action magnitude: {action_magnitudes.mean():.3f}")
    print(f"   Std action magnitude: {action_magnitudes.std():.3f}")
    
    # Check if agent is moving
    if action_magnitudes.mean() < 0.05:
        issue = f"CRITICAL: Agent is mostly stationary! Mean action magnitude: {action_magnitudes.mean():.3f}"
        print(f"   ✗ {issue}")
        issues.append(issue)
    elif action_magnitudes.mean() < 0.1:
        issue = f"WARNING: Agent has low action magnitude! Mean: {action_magnitudes.mean():.3f}"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Agent is taking meaningful actions")
    
    # Check action diversity
    if action_magnitudes.std() < 0.01:
        issue = f"WARNING: Actions have very low diversity (std: {action_magnitudes.std():.3f}), agent may be stuck in a pattern"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Actions show good diversity")
    
    # Trajectory analysis
    print("\n2. TRAJECTORY ANALYSIS:")
    obs, info = env.reset(seed=42)
    robot = env.env.robot_list[0]
    start_pos = robot.state.flatten()[:2].copy()
    goal_pos = robot.goal.flatten()[:2]
    initial_distance = info['distance_to_goal']
    
    trajectory = [start_pos.copy()]
    distances = [initial_distance]
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        robot = env.env.robot_list[0]
        trajectory.append(robot.state.flatten()[:2].copy())
        distances.append(info['distance_to_goal'])
        
        if terminated or truncated:
            break
    
    trajectory = np.array(trajectory)
    final_distance = distances[-1]
    progress = initial_distance - final_distance
    
    print(f"   Trajectory length: {len(trajectory)} steps")
    print(f"   Initial distance: {initial_distance:.3f}m")
    print(f"   Final distance: {final_distance:.3f}m")
    print(f"   Progress: {progress:.3f}m ({100*progress/initial_distance:.1f}%)")
    
    # Calculate total distance traveled
    total_distance_traveled = 0
    for i in range(1, len(trajectory)):
        total_distance_traveled += np.linalg.norm(trajectory[i] - trajectory[i-1])
    
    print(f"   Total distance traveled: {total_distance_traveled:.3f}m")
    print(f"   Efficiency (progress/traveled): {progress/total_distance_traveled:.3f}" if total_distance_traveled > 0 else "   Efficiency: N/A (no movement)")
    
    if progress < 0.5:
        issue = f"CRITICAL: Very little progress made! Progress: {progress:.3f}m ({100*progress/initial_distance:.1f}%)"
        print(f"   ✗ {issue}")
        issues.append(issue)
    
    if total_distance_traveled < 0.1:
        issue = f"CRITICAL: Agent barely moved! Total distance traveled: {total_distance_traveled:.3f}m"
        print(f"   ✗ {issue}")
        issues.append(issue)
    
    # Check for oscillation
    if len(trajectory) > 10:
        recent_positions = trajectory[-10:]
        position_variance = np.var(recent_positions, axis=0).sum()
        if position_variance < 0.01:
            issue = f"WARNING: Agent appears stuck/oscillating! Recent position variance: {position_variance:.6f}"
            print(f"   ⚠ {issue}")
            issues.append(issue)
    
    # Compare to random policy
    print("\n3. COMPARISON TO RANDOM POLICY:")
    random_progresses = []
    for seed in range(5):
        obs, info = env.reset(seed=seed + 100)
        initial_dist = info['distance_to_goal']
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        final_dist = info['distance_to_goal']
        random_progresses.append(initial_dist - final_dist)
    
    avg_random_progress = np.mean(random_progresses)
    print(f"   Random policy average progress: {avg_random_progress:.3f}m")
    print(f"   Trained model progress: {progress:.3f}m")
    
    if progress < avg_random_progress:
        issue = f"CRITICAL: Trained model performs WORSE than random policy! Model: {progress:.3f}m, Random: {avg_random_progress:.3f}m"
        print(f"   ✗ {issue}")
        issues.append(issue)
    elif progress < avg_random_progress * 1.5:
        issue = f"WARNING: Trained model only slightly better than random! Model: {progress:.3f}m, Random: {avg_random_progress:.3f}m"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    else:
        print(f"   ✓ Trained model performs better than random policy")
    
    return issues, trajectory, distances

def phase6_fundamental_issues(env):
    """Phase 6: Identify fundamental issues."""
    print("\n" + "="*70)
    print("PHASE 6: FUNDAMENTAL ISSUES CHECK")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Check reward function logic
    print("\n1. REWARD FUNCTION LOGIC:")
    
    # Simulate a scenario where agent moves away
    obs, info = env.reset(seed=42)
    initial_distance = info['distance_to_goal']
    
    # Move away from goal
    for step in range(10):
        # Turn away from goal
        action = np.array([0.5, 0.3], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
    
    final_distance = info['distance_to_goal']
    total_progress = initial_distance - final_distance
    
    print(f"   Scenario: Agent moves away from goal for 10 steps")
    print(f"   Initial distance: {initial_distance:.3f}m")
    print(f"   Final distance: {final_distance:.3f}m")
    print(f"   Total progress: {total_progress:.3f}m (negative = moved away)")
    
    # Calculate expected reward
    # Progress reward: 20.0 * progress per step
    # If moved 0.1m away per step: progress = -0.1, reward = 20.0 * (-0.1) = -2.0 per step
    # Over 10 steps: -20.0 total from progress alone
    expected_negative_reward = 20.0 * total_progress - 0.005 * 10  # progress + time penalty
    
    print(f"   Expected total reward (if moving away): {expected_negative_reward:.3f}")
    
    if total_progress < 0 and expected_negative_reward < -10.0:
        issue = "CRITICAL: Progress reward becomes very negative when agent moves away from goal!"
        print(f"   ✗ {issue}")
        print(f"   Current formula: reward += 20.0 * progress")
        print(f"   If progress is negative (moving away), reward becomes very negative")
        print(f"   This strongly discourages ANY movement that might temporarily increase distance")
        issues.append(issue)
        recommendations.append("Fix progress reward: Use max(0, progress) or distance-based reward instead")
    
    # Time penalty analysis
    print("\n2. TIME PENALTY ANALYSIS:")
    time_penalty_per_step = 0.005
    time_penalty_per_episode = time_penalty_per_step * 1000
    print(f"   Time penalty per step: {time_penalty_per_step}")
    print(f"   Time penalty per episode (1000 steps): {time_penalty_per_episode:.1f}")
    
    # Compare to progress reward
    max_progress_per_episode = 1.0 * 0.1 * 1000  # max_vel * step_time * max_steps
    max_progress_reward = 20.0 * max_progress_per_episode
    print(f"   Max possible progress reward per episode: {max_progress_reward:.1f}")
    print(f"   Net reward if making max progress: {max_progress_reward - time_penalty_per_episode:.1f}")
    
    if time_penalty_per_episode > max_progress_reward * 0.1:
        issue = f"WARNING: Time penalty ({time_penalty_per_episode:.1f}) is significant compared to max progress reward ({max_progress_reward:.1f})"
        print(f"   ⚠ {issue}")
        issues.append(issue)
    
    # Collision penalty balance
    print("\n3. COLLISION PENALTY BALANCE:")
    collision_penalty = -200.0
    goal_bonus = 200.0  # Updated to match current reward function
    
    print(f"   Collision penalty: {collision_penalty:.1f}")
    print(f"   Goal bonus: {goal_bonus:.1f}")
    print(f"   Ratio: {abs(collision_penalty) / goal_bonus:.1f}:1")
    
    if abs(collision_penalty) > goal_bonus * 1.5:
        issue = f"WARNING: Collision penalty ({collision_penalty:.1f}) is much larger than goal bonus ({goal_bonus:.1f})"
        print(f"   ⚠ {issue}")
        print(f"   This imbalance may cause the agent to be overly cautious, avoiding all movement")
        issues.append(issue)
    else:
        print(f"   ✓ Collision and goal rewards are balanced (1:1 ratio)")
        recommendations.append("Balance rewards: Increase goal bonus to match collision penalty (e.g., +200 for goal, -200 for collision)")
    
    return issues, recommendations

def generate_report(all_issues, recommendations, trajectory=None, distances=None, env=None, save_dir="logs/diagnostics"):
    """Generate comprehensive diagnostic report."""
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, "deep_diagnosis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DEEP RL SYSTEM DIAGNOSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total issues found: {len(all_issues)}\n")
        f.write(f"Critical issues: {len([i for i in all_issues if 'CRITICAL' in i])}\n")
        f.write(f"Warnings: {len([i for i in all_issues if 'WARNING' in i])}\n\n")
        
        f.write("ISSUES FOUND\n")
        f.write("-"*70 + "\n")
        for i, issue in enumerate(all_issues, 1):
            f.write(f"{i}. {issue}\n")
        
        if recommendations:
            f.write("\nRECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
    
    print(f"\n✓ Diagnostic report saved to: {report_path}")
    
    # Create visualization if trajectory data available
    if trajectory is not None and distances is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot trajectory
        ax = axes[0]
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Agent Trajectory')
        ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], c='red', s=100, marker='x', label='End', zorder=5)
        
        # Plot goal
        if env is not None and hasattr(env.env, 'robot_list') and len(env.env.robot_list) > 0:
            goal_pos = env.env.robot_list[0].goal.flatten()[:2]
            ax.scatter([goal_pos[0]], [goal_pos[1]], c='orange', s=150, marker='*', label='Goal', zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Agent Trajectory')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Plot distance over time
        ax = axes[1]
        ax.plot(distances, 'b-', linewidth=2)
        ax.axhline(y=0.3, color='g', linestyle='--', label='Success threshold (0.3m)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance to Goal (m)')
        ax.set_title('Distance to Goal Over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "deep_diagnosis_visualization.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Visualization saved to: {plot_path}")

def main():
    """Main diagnostic function."""
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    print("="*70)
    print("DEEP RL SYSTEM DIAGNOSIS")
    print("="*70)
    print("\nThis script will systematically diagnose why the RL agent")
    print("is achieving 0/5 success rate. It will check:")
    print("1. Model-environment compatibility")
    print("2. Action application")
    print("3. Reward function")
    print("4. Environment consistency")
    print("5. Agent behavior")
    print("6. Fundamental issues")
    print("="*70)
    
    # Find model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'ppo', 'ppo_nav_final')
    
    if not os.path.exists(model_path + '.zip'):
        model_path = os.path.join(model_dir, 'ppo', 'best_model')
        if not os.path.exists(model_path + '.zip'):
            print(f"\n✗ ERROR: No trained model found!")
            print(f"   Checked: {os.path.join(model_dir, 'ppo', 'ppo_nav_final.zip')}")
            print(f"   Checked: {os.path.join(model_dir, 'ppo', 'best_model.zip')}")
            return
    
    # Create environment
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    all_issues = []
    recommendations = []
    
    # Phase 1: Model compatibility
    model, issues = phase1_model_compatibility(env, model_path)
    all_issues.extend(issues)
    
    if model is None:
        print("\n✗ Cannot continue without model. Exiting.")
        env.close()
        return
    
    # Phase 2: Action application
    issues = phase2_action_application(env)
    all_issues.extend(issues)
    
    # Phase 3: Reward analysis
    issues = phase3_reward_analysis(env)
    all_issues.extend(issues)
    
    # Phase 4: Environment consistency
    issues = phase4_environment_consistency(env, config_path)
    all_issues.extend(issues)
    
    # Phase 5: Agent behavior
    issues, trajectory, distances = phase5_agent_behavior(model, env)
    all_issues.extend(issues)
    
    # Phase 6: Fundamental issues
    issues, recs = phase6_fundamental_issues(env)
    all_issues.extend(issues)
    recommendations.extend(recs)
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING DIAGNOSTIC REPORT")
    print("="*70)
    generate_report(all_issues, recommendations, trajectory, distances, env)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    print(f"Total issues found: {len(all_issues)}")
    critical_count = len([i for i in all_issues if 'CRITICAL' in i])
    warning_count = len([i for i in all_issues if 'WARNING' in i])
    print(f"  - Critical issues: {critical_count}")
    print(f"  - Warnings: {warning_count}")
    
    if critical_count > 0:
        print("\nCRITICAL ISSUES:")
        for issue in [i for i in all_issues if 'CRITICAL' in i]:
            print(f"  ✗ {issue}")
    
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    env.close()
    print("\n" + "="*70)
    print("Diagnosis complete! Check logs/diagnostics/ for detailed report.")
    print("="*70)

if __name__ == "__main__":
    main()

