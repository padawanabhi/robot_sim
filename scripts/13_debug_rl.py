#!/usr/bin/env python3
"""Debug script to analyze RL environment and identify issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv

def analyze_environment():
    """Analyze the RL environment to identify issues."""
    print("=" * 70)
    print("RL ENVIRONMENT DEBUG ANALYSIS")
    print("=" * 70)
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=False, num_obstacles=8)
    
    # Test episode
    obs, info = env.reset()
    print(f"\n1. INITIAL STATE:")
    print(f"   Initial distance to goal: {info['distance_to_goal']:.2f}m")
    print(f"   Max steps: {env.max_steps}")
    print(f"   Step time: 0.1s (from config)")
    print(f"   Max episode time: {env.max_steps * 0.1:.1f}s")
    
    print(f"\n2. ACTION SPACE:")
    print(f"   Action space: {env.action_space}")
    print(f"   Action low: {env.action_space.low}")
    print(f"   Action high: {env.action_space.high}")
    
    # Check robot velocity limits from config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    robot_vel_max = config['robot'][0]['vel_max']
    print(f"   Robot vel_max from config: {robot_vel_max}")
    print(f"   ⚠️  ISSUE: Action space allows [-1.5, 1.5] but robot only accepts [-1, 1]!")
    
    print(f"\n3. OBSERVATION SPACE:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    print(f"   Observation meaning: [dx_norm, dy_norm, dtheta, v, w, min_obstacle_dist, distance_to_goal]")
    
    print(f"\n4. TESTING MOVEMENT:")
    # Test different actions
    test_actions = [
        [0.0, 0.0],      # No movement
        [0.5, 0.0],      # Forward
        [1.0, 0.0],      # Fast forward
        [0.0, 0.5],      # Turn only
        [0.5, 0.5],     # Forward + turn
    ]
    
    distances_traveled = []
    rewards_received = []
    
    for action in test_actions:
        env.reset()
        initial_pos = env.env.robot_list[0].state.flatten()[:2]
        total_reward = 0
        
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            total_reward += reward
            if terminated or truncated:
                break
        
        final_pos = env.env.robot_list[0].state.flatten()[:2]
        distance = np.sqrt((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2)
        distances_traveled.append(distance)
        rewards_received.append(total_reward)
        
        print(f"   Action {action}: moved {distance:.3f}m, reward: {total_reward:.2f}")
    
    print(f"\n5. REWARD ANALYSIS:")
    print(f"   Average reward per step (forward): {rewards_received[1] / 10:.3f}")
    print(f"   Time penalty per step: -0.01")
    print(f"   Net reward per step (if no progress): {rewards_received[1] / 10 - 0.01:.3f}")
    
    # Simulate a full episode
    print(f"\n6. FULL EPISODE SIMULATION:")
    env.reset()
    initial_distance = info['distance_to_goal']
    total_reward = 0
    step = 0
    
    # Use a simple forward action
    while step < env.max_steps:
        action = np.array([0.8, 0.0])  # Forward at 0.8 m/s
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"   Step {step}: distance={info['distance_to_goal']:.2f}m, "
                  f"cumulative_reward={total_reward:.2f}, obs[5]={obs[5]:.2f}")
        
        if terminated or truncated:
            break
    
    final_distance = info['distance_to_goal']
    progress = initial_distance - final_distance
    print(f"\n   Final: {step} steps, distance={final_distance:.2f}m, "
          f"progress={progress:.2f}m, total_reward={total_reward:.2f}")
    
    print(f"\n7. IDENTIFIED ISSUES:")
    issues = []
    
    # Issue 1: Action space mismatch
    if env.action_space.high[0] > robot_vel_max[0]:
        issues.append("ACTION SPACE MISMATCH: Action space allows higher velocities than robot config")
    
    # Issue 2: Max steps might be too low
    max_distance_possible = env.max_steps * 0.1 * 1.0  # steps * time * max_vel
    if max_distance_possible < initial_distance * 1.5:
        issues.append(f"MAX STEPS TOO LOW: Can only travel {max_distance_possible:.1f}m max, "
                     f"but goal is {initial_distance:.1f}m away")
    
    # Issue 3: Reward might be too sparse
    if abs(rewards_received[0]) < 0.1:  # No movement reward
        issues.append("REWARD TOO SPARSE: No movement gives very small reward signal")
    
    # Issue 4: Time penalty might be too high
    time_penalty_total = env.max_steps * 0.01
    if time_penalty_total > 2.0:
        issues.append(f"TIME PENALTY TOO HIGH: Total time penalty ({time_penalty_total:.1f}) "
                     f"might discourage exploration")
    
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    env.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_environment()

