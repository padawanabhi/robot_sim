#!/usr/bin/env python3
"""Compare RL agent vs classical controllers side by side."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
import glob
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv

from scripts.go_to_goal_controller import GoToGoalController
import numpy as np
import yaml
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def run_controller_episode(controller_type, config_path, episode_num=1, algorithm='ppo'):
    """
    Run one episode with a specific controller.
    
    Args:
        controller_type: 'rl', 'go_to_goal', or 'potential_field'
        config_path: Path to world config
        episode_num: Episode number for randomization seed
    
    Returns:
        dict with episode results
    """
    import irsim
    
    # Create environment
    if controller_type == 'rl':
        env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
        # Load RL model based on algorithm
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Try algorithm-specific paths
        model_path = None
        if algorithm.lower() == 'ppo':
            # Match evaluation script priority: try final model first (most recently trained)
            model_path = os.path.join(model_dir, 'ppo', 'ppo_nav_final')
            if not os.path.exists(model_path + '.zip'):
                # Fall back to best_model (best performing checkpoint during training)
                model_path = os.path.join(model_dir, 'ppo', 'best_model')
            if not os.path.exists(model_path + '.zip'):
                # Try improved PPO
                model_path = os.path.join(model_dir, 'ppo_improved', 'ppo_improved_final')
            if not os.path.exists(model_path + '.zip'):
                # Try root directory
                model_path = os.path.join(model_dir, 'ppo_nav_final')
            if not os.path.exists(model_path + '.zip'):
                model_path = os.path.join(model_dir, 'best_model')
            if os.path.exists(model_path + '.zip'):
                model = PPO.load(model_path)
            else:
                print(f"Warning: PPO model not found")
                return None
        elif algorithm.lower() == 'sac':
            model_path = os.path.join(model_dir, 'sac', 'sac_nav_final')
            if not os.path.exists(model_path + '.zip'):
                model_path = os.path.join(model_dir, 'best_model')
            if os.path.exists(model_path + '.zip'):
                model = SAC.load(model_path)
            else:
                print(f"Warning: SAC model not found")
                return None
        elif algorithm.lower() == 'td3':
            model_path = os.path.join(model_dir, 'td3', 'td3_nav_final')
            if not os.path.exists(model_path + '.zip'):
                model_path = os.path.join(model_dir, 'best_model')
            if os.path.exists(model_path + '.zip'):
                model = TD3.load(model_path)
            else:
                print(f"Warning: TD3 model not found")
                return None
        else:
            print(f"Warning: Unknown algorithm {algorithm}")
            return None
        
        # Use same seed scheme as evaluation script for consistency
        obs, info = env.reset(seed=episode_num + 42)
    else:
        # For classical controllers, create a randomized environment similar to RL
        # Load base config and randomize it
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Use same seed scheme as evaluation script for consistency
        np.random.seed(episode_num + 42)
        
        # Randomize robot start
        start_x = np.random.uniform(1.0, 2.5)
        start_y = np.random.uniform(1.0, 2.5)
        start_theta = np.random.uniform(-np.pi, np.pi)
        config['robot'][0]['state'] = [start_x, start_y, start_theta]
        
        # Randomize goal
        goal_x = np.random.uniform(7.0, 9.0)
        goal_y = np.random.uniform(7.0, 9.0)
        goal_theta = np.random.uniform(-np.pi, np.pi)
        config['robot'][0]['goal'] = [goal_x, goal_y, goal_theta]
        
        # Ensure goal is far from start
        while np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2) < 5.0:
            goal_x = np.random.uniform(7.0, 9.0)
            goal_y = np.random.uniform(7.0, 9.0)
            config['robot'][0]['goal'] = [goal_x, goal_y, goal_theta]
        
        # Add random obstacles (8 obstacles)
        obstacles = []
        for _ in range(8):
            obs_x = np.random.uniform(1.0, 9.0)
            obs_y = np.random.uniform(1.0, 9.0)
            
            if np.random.random() < 0.7:  # 70% circles
                radius = np.random.uniform(0.25, 0.5)
                obstacles.append({
                    'shape': {'name': 'circle', 'radius': radius},
                    'state': [obs_x, obs_y],
                    'color': 'red'
                })
            else:  # 30% rectangles
                length = np.random.uniform(1.0, 2.0)
                width = np.random.uniform(0.2, 0.4)
                angle = np.random.uniform(0, 2*np.pi)
                obstacles.append({
                    'shape': {'name': 'rectangle', 'length': length, 'width': width},
                    'state': [obs_x, obs_y, angle],
                    'color': 'gray'
                })
        
        config['obstacle'] = obstacles
        
        # Save temp config and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        env = irsim.make(temp_config_path)
        
        # Clean up temp file
        import time
        time.sleep(0.1)
        try:
            os.unlink(temp_config_path)
        except:
            pass
    
    # Store trajectory
    robot_poses = []
    goal_pos = None
    step = 0
    total_reward = 0
    done = False
    
    # Initialize controller
    if controller_type == 'go_to_goal':
        controller = GoToGoalController()
    elif controller_type == 'potential_field':
        from scripts.potential_field_controller import PotentialFieldController
        controller = PotentialFieldController()
    
    # Get initial state
    if controller_type == 'rl':
        robot = env.env.robot_list[0]
        obstacles_list = env.env.obstacle_list
    else:
        robot = env.robot_list[0]
        obstacles_list = env.obstacle_list
    
    initial_pose = robot.state.flatten()[:3]
    goal_pose = robot.goal.flatten()[:3]
    goal_pos = goal_pose[:2]
    
    while not done and step < 1000:  # Match evaluation script max_steps
        # Get current state
        if controller_type == 'rl':
            robot = env.env.robot_list[0]
            pose = robot.state.flatten()[:3]
        else:
            robot = env.robot_list[0]
            pose = robot.state.flatten()[:3]
        
        robot_poses.append(pose.copy())
        
        # Compute action
        if controller_type == 'rl':
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        elif controller_type == 'go_to_goal':
            obstacles = obstacles_list
            action = controller.compute(pose, goal_pose, obstacles)
            env.step([action])
            done = env.done()
        elif controller_type == 'potential_field':
            pos = pose[:2]
            theta = pose[2]
            goal = goal_pose[:2]
            obstacles = obstacles_list
            action = controller.compute(pos, theta, goal, obstacles)
            env.step([action])
            done = env.done()
        
        step += 1
    
    # Get final state (use consistent calculation for all)
    if controller_type == 'rl':
        final_pose = env.env.robot_list[0].state.flatten()[:3]
        # Calculate distance consistently (same as other controllers)
        final_distance = np.sqrt((goal_pose[0]-final_pose[0])**2 + (goal_pose[1]-final_pose[1])**2)
        success = final_distance < 0.3
    else:
        final_pose = env.robot_list[0].state.flatten()[:3]
        final_distance = np.sqrt((goal_pose[0]-final_pose[0])**2 + (goal_pose[1]-final_pose[1])**2)
        success = final_distance < 0.3
    
    obstacles = []
    for obs_obj in obstacles_list:
        obs_state = obs_obj.state.flatten()
        center = obs_state[:2]
        
        try:
            radius = getattr(obs_obj.shape, 'radius', None)
            if radius is not None:
                obstacles.append({
                    'type': 'circle',
                    'center': center,
                    'radius': float(radius)
                })
            else:
                length = getattr(obs_obj.shape, 'length', 1.0)
                width = getattr(obs_obj.shape, 'width', 0.3)
                obstacles.append({
                    'type': 'rectangle',
                    'center': center,
                    'length': float(length),
                    'width': float(width),
                    'angle': float(obs_state[2]) if len(obs_state) > 2 else 0.0
                })
        except:
            obstacles.append({
                'type': 'circle',
                'center': center,
                'radius': 0.5
            })
    
    if controller_type == 'rl':
        env.close()
    else:
        try:
            env.end()
        except:
            pass
    
    return {
        'controller': controller_type,
        'poses': np.array(robot_poses),
        'goal': goal_pos,
        'obstacles': obstacles,
        'success': success,
        'steps': step,
        'final_distance': final_distance,
        'reward': total_reward if controller_type == 'rl' else None
    }

def compare_controllers(num_episodes=5, include_all_rl=True):
    """Compare all controllers on the same scenarios."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    
    # Build controller list
    controllers = ['go_to_goal', 'potential_field']
    rl_algorithms = []
    
    if include_all_rl:
        # Check which RL algorithms have trained models
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if os.path.exists(os.path.join(model_dir, 'ppo_improved', 'ppo_improved_final.zip')) or \
           os.path.exists(os.path.join(model_dir, 'ppo', 'ppo_nav_final.zip')) or \
           os.path.exists(os.path.join(model_dir, 'ppo_nav_final.zip')):
            rl_algorithms.append('ppo')
        if os.path.exists(os.path.join(model_dir, 'sac', 'sac_nav_final.zip')):
            rl_algorithms.append('sac')
        if os.path.exists(os.path.join(model_dir, 'td3', 'td3_nav_final.zip')):
            rl_algorithms.append('td3')
    
    # Add RL controllers
    for alg in rl_algorithms:
        controllers.append(f'rl_{alg}')
    
    results = {ctrl: [] for ctrl in controllers}
    
    print(f"\nComparing controllers on {num_episodes} episodes...")
    print("=" * 70)
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1}/{num_episodes}")
        print("-" * 70)
        
        for ctrl in controllers:
            try:
                if ctrl.startswith('rl_'):
                    algorithm = ctrl.split('_')[1]
                    result = run_controller_episode('rl', config_path, episode_num=ep, algorithm=algorithm)
                else:
                    result = run_controller_episode(ctrl, config_path, episode_num=ep)
                
                if result:
                    results[ctrl].append(result)
                    status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
                    print(f"{ctrl:20s}: {status:10s} | Steps: {result['steps']:3d} | "
                          f"Final dist: {result['final_distance']:.2f}m")
                else:
                    print(f"{ctrl:20s}: Model not found")
            except Exception as e:
                print(f"{ctrl:20s}: Error - {e}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for ctrl in controllers:
        if len(results[ctrl]) > 0:
            successes = sum(1 for r in results[ctrl] if r['success'])
            avg_steps = np.mean([r['steps'] for r in results[ctrl]])
            avg_distance = np.mean([r['final_distance'] for r in results[ctrl]])
            
            ctrl_name = ctrl.upper().replace('_', ' ')
            print(f"\n{ctrl_name}:")
            print(f"  Success rate: {successes}/{len(results[ctrl])} ({100*successes/len(results[ctrl]):.1f}%)")
            print(f"  Avg steps: {avg_steps:.1f}")
            print(f"  Avg final distance: {avg_distance:.2f}m")
            if ctrl.startswith('rl_'):
                avg_reward = np.mean([r['reward'] for r in results[ctrl] if r['reward'] is not None])
                print(f"  Avg reward: {avg_reward:.2f}")
    
    # Create comparison visualizations
    create_comparison_plots(results, num_episodes)
    
    return results

def create_comparison_plots(results, num_episodes):
    """Create side-by-side comparison plots."""
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'comparisons')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get all controllers that have results
    controllers = [ctrl for ctrl in results.keys() if len(results[ctrl]) > 0]
    controller_names = {
        'go_to_goal': 'Go-to-Goal',
        'potential_field': 'Potential Field',
        'rl_ppo': 'RL Agent (PPO)',
        'rl_sac': 'RL Agent (SAC)',
        'rl_td3': 'RL Agent (TD3)'
    }
    
    # Create one plot per episode showing all controllers
    max_episodes = max([len(results[ctrl]) for ctrl in controllers] + [num_episodes])
    for ep in range(min(num_episodes, max_episodes)):
        num_ctrls = len([ctrl for ctrl in controllers if ep < len(results[ctrl])])
        if num_ctrls == 0:
            continue
        
        fig, axes = plt.subplots(1, num_ctrls, figsize=(6*num_ctrls, 6))
        if num_ctrls == 1:
            axes = [axes]
        
        plot_idx = 0
        for ctrl in controllers:
            if ep < len(results[ctrl]):
                result = results[ctrl][ep]
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Draw obstacles
                for obs in result['obstacles']:
                    if obs['type'] == 'circle':
                        circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.6)
                        ax.add_patch(circle)
                    else:
                        rect = Rectangle(
                            (obs['center'][0] - obs['length']/2, obs['center'][1] - obs['width']/2),
                            obs['length'], obs['width'],
                            angle=np.degrees(obs['angle']),
                            color='gray', alpha=0.6
                        )
                        ax.add_patch(rect)
                
                # Draw goal
                goal_circle = Circle(result['goal'], 0.3, color='green', alpha=0.7)
                ax.add_patch(goal_circle)
                
                # Draw trajectory
                poses = result['poses']
                ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=2, alpha=0.6)
                
                # Draw start and end
                ax.plot(poses[0, 0], poses[0, 1], 'go', markersize=10)
                end_color = 'green' if result['success'] else 'red'
                ax.plot(poses[-1, 0], poses[-1, 1], 'o', color=end_color, markersize=10)
                
                # Set properties
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ctrl_display_name = controller_names.get(ctrl, ctrl.replace('_', ' ').title())
                ax.set_title(f"{ctrl_display_name}\n"
                            f"{'SUCCESS' if result['success'] else 'FAILED'} - "
                            f"{result['steps']} steps",
                            fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(viz_dir, f'comparison_episode_{ep+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {save_path}")

if __name__ == "__main__":
    compare_controllers(num_episodes=5)

