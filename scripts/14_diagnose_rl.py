#!/usr/bin/env python3
"""Diagnose RL agent behavior to understand why it's not reaching the goal."""
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

def diagnose_episode(model, env, episode_num=1):
    """Run one episode and collect detailed diagnostics."""
    obs, info = env.reset(seed=episode_num + 42)
    
    initial_distance = info['distance_to_goal']
    print(f"\n=== Episode {episode_num} Diagnostics ===")
    print(f"Initial distance to goal: {initial_distance:.2f}m")
    
    # Collect data
    distances = [initial_distance]
    rewards = []
    actions = []
    min_obstacle_dists = []
    heading_errors = []
    positions = []
    
    robot = env.env.robot_list[0]
    initial_pos = robot.state.flatten()[:2]
    goal_pos = robot.goal.flatten()[:2]
    positions.append(initial_pos.copy())
    
    step = 0
    done = False
    
    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        distances.append(info['distance_to_goal'])
        rewards.append(reward)
        actions.append(action.copy())
        min_obstacle_dists.append(obs[5])
        heading_errors.append(abs(obs[2]))
        
        robot = env.env.robot_list[0]
        positions.append(robot.state.flatten()[:2].copy())
        
        done = terminated or truncated
        step += 1
        
        # Print every 100 steps
        if step % 100 == 0:
            print(f"  Step {step}: distance={info['distance_to_goal']:.2f}m, "
                  f"reward={reward:.2f}, action=[{action[0]:.2f}, {action[1]:.2f}], "
                  f"min_obs_dist={obs[5]:.2f}m, heading_err={np.degrees(abs(obs[2])):.1f}°")
    
    final_distance = info['distance_to_goal']
    total_reward = sum(rewards)
    progress = initial_distance - final_distance
    
    # Check obstacle detection
    robot = env.env.robot_list[0]
    final_robot_pos = robot.state.flatten()[:2]
    num_obstacles = len(env.env.obstacle_list) if hasattr(env.env, 'obstacle_list') else 0
    
    print(f"\nEpisode Summary:")
    print(f"  Steps: {step}")
    print(f"  Initial distance: {initial_distance:.2f}m")
    print(f"  Final distance: {final_distance:.2f}m")
    print(f"  Progress: {progress:.2f}m ({100*progress/initial_distance:.1f}%)")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/step:.3f}")
    print(f"  Average action magnitude: {np.mean([np.sqrt(a[0]**2 + a[1]**2) for a in actions]):.3f}")
    print(f"  Average min obstacle distance: {np.mean(min_obstacle_dists):.3f}m")
    print(f"  Average heading error: {np.degrees(np.mean(heading_errors)):.1f}°")
    print(f"  Number of obstacles in environment: {num_obstacles}")
    print(f"  Success: {'YES' if final_distance < 0.3 else 'NO'}")
    
    # Debug obstacle detection
    if num_obstacles == 0:
        print(f"  ⚠️ WARNING: No obstacles detected in environment!")
    elif np.mean(min_obstacle_dists) >= 9.9:
        print(f"  ⚠️ WARNING: Min obstacle distance is always near max (10.0m)")
        print(f"     This suggests obstacles are not being detected properly")
        # Try to manually calculate distance to first obstacle
        if num_obstacles > 0:
            first_obs = env.env.obstacle_list[0]
            obs_pos = first_obs.state.flatten()[:2]
            manual_dist = np.sqrt((final_robot_pos[0] - obs_pos[0])**2 + (final_robot_pos[1] - obs_pos[1])**2)
            print(f"     Manual distance to first obstacle: {manual_dist:.2f}m")
    
    # Analyze behavior
    print(f"\nBehavior Analysis:")
    
    # Check if agent is moving
    avg_action_mag = np.mean([np.sqrt(a[0]**2 + a[1]**2) for a in actions])
    if avg_action_mag < 0.1:
        print(f"  ⚠️ Agent is mostly stationary (avg action magnitude: {avg_action_mag:.3f})")
    else:
        print(f"  ✓ Agent is moving (avg action magnitude: {avg_action_mag:.3f})")
    
    # Check progress
    if progress < 0.5:
        print(f"  ⚠️ Very little progress made ({progress:.2f}m)")
    elif progress < initial_distance * 0.3:
        print(f"  ⚠️ Limited progress ({100*progress/initial_distance:.1f}% of initial distance)")
    else:
        print(f"  ✓ Good progress made ({100*progress/initial_distance:.1f}% of initial distance)")
    
    # Check if stuck
    if step == 1000 and final_distance > initial_distance * 0.7:
        print(f"  ⚠️ Agent may be stuck or oscillating (reached max steps, still far from goal)")
    
    # Check obstacle avoidance
    avg_obs_dist = np.mean(min_obstacle_dists)
    if avg_obs_dist < 0.3:
        print(f"  ⚠️ Agent is very close to obstacles (avg distance: {avg_obs_dist:.3f}m)")
    elif avg_obs_dist < 0.5:
        print(f"  ⚠️ Agent is close to obstacles (avg distance: {avg_obs_dist:.3f}m)")
    else:
        print(f"  ✓ Agent maintains safe distance from obstacles (avg: {avg_obs_dist:.3f}m)")
    
    # Check heading
    avg_heading_err = np.degrees(np.mean(heading_errors))
    if avg_heading_err > 45:
        print(f"  ⚠️ Agent has poor heading alignment (avg error: {avg_heading_err:.1f}°)")
    else:
        print(f"  ✓ Agent maintains good heading (avg error: {avg_heading_err:.1f}°)")
    
    return {
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'progress': progress,
        'total_reward': total_reward,
        'steps': step,
        'success': final_distance < 0.3,
        'distances': distances,
        'rewards': rewards,
        'actions': actions,
        'positions': positions,
        'goal_pos': goal_pos
    }

def visualize_diagnosis(diagnosis, save_path=None):
    """Create diagnostic visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Trajectory
    ax = axes[0, 0]
    positions = np.array(diagnosis['positions'])
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax.scatter([positions[0, 0]], [positions[0, 1]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([positions[-1, 0]], [positions[-1, 1]], c='red', s=100, marker='x', label='End', zorder=5)
    ax.scatter([diagnosis['goal_pos'][0]], [diagnosis['goal_pos'][1]], c='orange', s=150, marker='*', label='Goal', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"Trajectory (Progress: {diagnosis['progress']:.2f}m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Plot 2: Distance over time
    ax = axes[0, 1]
    distances = diagnosis['distances']
    ax.plot(distances, 'b-', linewidth=2)
    ax.axhline(y=0.3, color='g', linestyle='--', label='Success threshold (0.3m)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Goal (m)')
    ax.set_title('Distance to Goal Over Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Reward over time
    ax = axes[1, 0]
    rewards = diagnosis['rewards']
    ax.plot(rewards, 'g-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward Over Time (Total: {diagnosis["total_reward"]:.2f})')
    ax.grid(True)
    
    # Plot 4: Action magnitude over time
    ax = axes[1, 1]
    actions = diagnosis['actions']
    action_magnitudes = [np.sqrt(a[0]**2 + a[1]**2) for a in actions]
    ax.plot(action_magnitudes, 'r-', linewidth=1, alpha=0.7, label='Action Magnitude')
    ax.axhline(y=0.05, color='k', linestyle='--', alpha=0.3, label='Movement threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Action Magnitude')
    ax.set_title(f'Action Magnitude (Avg: {np.mean(action_magnitudes):.3f})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nDiagnostic plot saved to: {save_path}")
    else:
        save_path = f"logs/diagnostics/episode_{diagnosis.get('episode_num', 1)}_diagnosis.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nDiagnostic plot saved to: {save_path}")
    
    plt.close()

def main():
    """Main diagnostic function."""
    # Load model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'ppo', 'best_model')
    
    if not os.path.exists(model_path + '.zip'):
        model_path = os.path.join(model_dir, 'ppo', 'ppo_nav_final')
        if not os.path.exists(model_path + '.zip'):
            print("No trained model found. Run training first.")
            return
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    # Run diagnostics on a few episodes
    num_episodes = 3
    diagnoses = []
    
    for ep in range(num_episodes):
        diagnosis = diagnose_episode(model, env, episode_num=ep + 1)
        diagnosis['episode_num'] = ep + 1
        diagnoses.append(diagnosis)
        
        # Visualize
        save_path = f"logs/diagnostics/episode_{ep+1}_diagnosis.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_diagnosis(diagnosis, save_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS EPISODES")
    print(f"{'='*70}")
    
    avg_progress = np.mean([d['progress'] for d in diagnoses])
    avg_final_dist = np.mean([d['final_distance'] for d in diagnoses])
    avg_reward = np.mean([d['total_reward'] for d in diagnoses])
    success_count = sum([d['success'] for d in diagnoses])
    
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average progress: {avg_progress:.2f}m")
    print(f"Average final distance: {avg_final_dist:.2f}m")
    print(f"Average total reward: {avg_reward:.2f}")
    
    # Common issues
    print(f"\nCommon Issues Detected:")
    issues = []
    
    if avg_progress < 1.0:
        issues.append(f"⚠️ Very little progress on average ({avg_progress:.2f}m)")
    
    if avg_final_dist > 5.0:
        issues.append(f"⚠️ Still very far from goal on average ({avg_final_dist:.2f}m)")
    
    if success_count == 0:
        issues.append("⚠️ No successful episodes")
    
    if len(issues) == 0:
        print("  ✓ No major issues detected")
    else:
        for issue in issues:
            print(f"  {issue}")
    
    env.close()

if __name__ == "__main__":
    main()

