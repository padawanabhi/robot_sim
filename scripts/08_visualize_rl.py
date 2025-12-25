#!/usr/bin/env python3
"""Visualize RL agent performance with proper plotting."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv
import numpy as np
import matplotlib
# Use non-interactive backend to prevent hangs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import time

def visualize_episode(model, env, episode_num=1, save_path=None):
    """Run one episode and visualize it properly."""
    obs, info = env.reset()
    done = False
    step = 0
    
    # Store trajectory
    trajectory = []
    robot_poses = []
    goal_pos = None
    
    print(f"\nRunning Episode {episode_num}...")
    print(f"Initial distance to goal: {info['distance_to_goal']:.2f}m")
    
    while not done and step < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        # Get robot and goal positions
        robot = env.env.robot_list[0]
        pose = robot.state.flatten()[:3]
        goal = robot.goal.flatten()[:3]
        
        if goal_pos is None:
            goal_pos = goal[:2]
        
        robot_poses.append(pose.copy())
        trajectory.append({
            'step': step,
            'pose': pose.copy(),
            'reward': reward,
            'distance': info['distance_to_goal']
        })
    
    success = info['distance_to_goal'] < 0.3
    total_reward = sum(t['reward'] for t in trajectory)
    
    print(f"Episode {episode_num}: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Steps: {step}, Total Reward: {total_reward:.2f}, Final Distance: {info['distance_to_goal']:.2f}m")
    
    # Get obstacles - use same approach as RL environment
    obstacles = []
    for obs_obj in env.env.obstacle_list:
        obs_state = obs_obj.state.flatten()
        center = obs_state[:2]
        
        # Use getattr with defaults (same as RL environment does)
        try:
            radius = getattr(obs_obj.shape, 'radius', None)
            if radius is not None:
                # Circle obstacle
                obstacles.append({
                    'type': 'circle',
                    'center': center,
                    'radius': float(radius)
                })
            else:
                # Try rectangle
                length = getattr(obs_obj.shape, 'length', 1.0)
                width = getattr(obs_obj.shape, 'width', 0.3)
                obstacles.append({
                    'type': 'rectangle',
                    'center': center,
                    'length': float(length),
                    'width': float(width),
                    'angle': float(obs_state[2]) if len(obs_state) > 2 else 0.0
                })
        except (AttributeError, TypeError):
            # Fallback: assume circle with default radius
            obstacles.append({
                'type': 'circle',
                'center': center,
                'radius': 0.5  # default radius
            })
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw obstacles
    for obs in obstacles:
        if obs['type'] == 'circle':
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.6)
            ax.add_patch(circle)
        else:
            # Rectangle
            center = obs['center']
            angle = obs['angle']
            length = obs['length']
            width = obs['width']
            
            # Create rectangle
            rect = Rectangle(
                (center[0] - length/2, center[1] - width/2),
                length, width,
                angle=np.degrees(angle),
                color='gray', alpha=0.6
            )
            ax.add_patch(rect)
    
    # Draw goal
    goal_circle = Circle(goal_pos, 0.3, color='green', alpha=0.7, label='Goal')
    ax.add_patch(goal_circle)
    
    # Draw trajectory
    poses = np.array(robot_poses)
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=2, alpha=0.6, label='Path')
    
    # Draw start position
    ax.plot(poses[0, 0], poses[0, 1], 'go', markersize=10, label='Start')
    
    # Draw end position
    end_color = 'green' if success else 'red'
    ax.plot(poses[-1, 0], poses[-1, 1], 'o', color=end_color, markersize=10, label='End')
    
    # Draw robot at final position
    final_pose = poses[-1]
    robot_circle = Circle(final_pose[:2], 0.2, color='blue', alpha=0.5)
    ax.add_patch(robot_circle)
    
    # Draw heading arrow
    arrow_length = 0.3
    ax.arrow(
        final_pose[0], final_pose[1],
        arrow_length * np.cos(final_pose[2]),
        arrow_length * np.sin(final_pose[2]),
        head_width=0.1, head_length=0.1, fc='blue', ec='blue'
    )
    
    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Episode {episode_num}: {"SUCCESS" if success else "FAILED"} - '
                f'{step} steps, Reward: {total_reward:.2f}')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # Close immediately without showing (prevents hangs on macOS)
    plt.close()
    
    return success, total_reward, step

def main():
    """Main visualization function."""
    # Load model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'ppo_nav_final')
    
    if not os.path.exists(model_path + '.zip'):
        model_path = os.path.join(model_dir, 'best_model')
        if not os.path.exists(model_path + '.zip'):
            print("No trained model found. Run training first.")
            return
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment (headless for data collection)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    # Create output directory
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize multiple episodes
    num_episodes = 5
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"\nVisualizing {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        save_path = os.path.join(viz_dir, f'episode_{ep+1}.png')
        success, reward, length = visualize_episode(model, env, ep+1, save_path)
        
        success_count += int(success)
        total_rewards.append(reward)
        episode_lengths.append(length)
        
        # No delay needed since we're not displaying
    
    print(f"\n=== Visualization Results ===")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"\nVisualizations saved to: {viz_dir}")
    
    env.close()

if __name__ == "__main__":
    main()

