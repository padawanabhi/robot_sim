#!/usr/bin/env python3
"""Evaluate trained RL agent."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
# Import to register the environment
# Note: Using importlib because module name starts with number
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for evaluation
import matplotlib.pyplot as plt

def evaluate(headless=False):
    """Evaluate trained model."""
    # Load model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'ppo_nav_final')
    
    if not os.path.exists(model_path + '.zip'):
        # Try best model
        model_path = os.path.join(model_dir, 'best_model')
        if not os.path.exists(model_path + '.zip'):
            print("No trained model found. Run training first.")
            return
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment - use headless mode to avoid crashes
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    render_mode = None if headless else 'human'
    env = DiffDriveNavEnv(config_path=config_path, render_mode=render_mode, randomize=True, num_obstacles=8)
    
    # Enable matplotlib interactive mode only if not headless
    if not headless:
        try:
            plt.ion()
        except:
            headless = True  # Fallback to headless if matplotlib fails
    
    # Run episodes
    num_episodes = 5
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    if headless:
        print("Running in headless mode (no visualization)")
    
    for ep in range(num_episodes):
        try:
            obs, info = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            print(f"\n--- Episode {ep+1} ---")
            print(f"Initial distance to goal: {info['distance_to_goal']:.2f}m")
            
            while not done and step < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
                
                # Only render if not headless and occasionally to reduce overhead
                if not headless and step % 5 == 0:
                    try:
                        env.render()
                        # Set axis limits and update display
                        try:
                            fig = plt.gcf()
                            if fig:
                                ax = fig.gca()
                                if ax:
                                    ax.set_xlim(0, 10)
                                    ax.set_ylim(0, 10)
                            plt.pause(0.01)
                        except:
                            pass
                    except:
                        pass  # Silently fail if rendering causes issues
            
            success = info['distance_to_goal'] < 0.3
            success_count += int(success)
            total_rewards.append(total_reward)
            episode_lengths.append(step)
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status} | Steps: {step}, Reward: {total_reward:.2f}, "
                  f"Final distance: {info['distance_to_goal']:.2f}m")
            
            # Small delay between episodes
            import time
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in episode {ep+1}: {e}")
            continue
    
    print(f"\n=== Evaluation Results ===")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    
    try:
        env.close()
    except:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', 
                       help='Run evaluation without visualization (prevents crashes)')
    args = parser.parse_args()
    
    evaluate(headless=args.headless)

