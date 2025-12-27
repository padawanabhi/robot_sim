#!/usr/bin/env python3
"""Evaluate trained RL agent."""
import sys
import os
import signal

# Set non-interactive backend BEFORE any matplotlib imports to prevent hangs
os.environ['MPLBACKEND'] = 'Agg'

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
matplotlib.use('Agg')  # Use non-interactive backend to prevent hangs
import matplotlib.pyplot as plt

def cleanup_handler(signum, frame):
    """Handle cleanup on interrupt."""
    print("\n\nInterrupted! Cleaning up...")
    sys.exit(0)

def evaluate(headless=True):
    """Evaluate trained model."""
    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Load model - prefer best_model (best performing during training) over final model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Try final model first (most recently trained)
    model_path = os.path.join(model_dir, 'ppo', 'ppo_nav_final')
    if not os.path.exists(model_path + '.zip'):
        # Fall back to best_model (best performing checkpoint during training)
        model_path = os.path.join(model_dir, 'ppo', 'best_model')
        if not os.path.exists(model_path + '.zip'):
            # Try root models directory
            model_path = os.path.join(model_dir, 'ppo_nav_final')
            if not os.path.exists(model_path + '.zip'):
                # Try root best_model
                model_path = os.path.join(model_dir, 'best_model')
                if not os.path.exists(model_path + '.zip'):
                    print("No trained model found. Run training first.")
                    return
    
    print(f"Loading model from {model_path}...")
    
    # #region agent log
    import json
    import stat
    from datetime import datetime
    try:
        model_file = model_path + '.zip'
        if os.path.exists(model_file):
            file_stat = os.stat(model_file)
            mod_time = datetime.fromtimestamp(file_stat.st_mtime)
            file_size = file_stat.st_size
            with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "eval-run",
                    "hypothesisId": "H2",
                    "location": "07_evaluate_rl.py:model_load",
                    "message": "Model file info",
                    "data": {"model_path": model_path, "mod_time": mod_time.isoformat(), "file_size_kb": file_size / 1024},
                    "timestamp": int(__import__('time').time() * 1000)
                }) + '\n')
    except Exception as e:
        with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "eval-run",
                "hypothesisId": "H2",
                "location": "07_evaluate_rl.py:model_load:error",
                "message": "Error getting model file info",
                "data": {"error": str(e)},
                "timestamp": int(__import__('time').time() * 1000)
            }) + '\n')
    # #endregion
    
    try:
        model = PPO.load(model_path)
        
        # #region agent log
        try:
            with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "eval-run",
                    "hypothesisId": "H3",
                    "location": "07_evaluate_rl.py:model_loaded",
                    "message": "Model loaded successfully",
                    "data": {
                        "obs_space_shape": list(model.observation_space.shape),
                        "action_space_shape": list(model.action_space.shape),
                        "action_space_low": [float(x) for x in model.action_space.low],
                        "action_space_high": [float(x) for x in model.action_space.high]
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + '\n')
        except: pass
        # #endregion
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment - always use headless mode to prevent hangs
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    # #region agent log
    try:
        with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "eval-run",
                "hypothesisId": "H3",
                "location": "07_evaluate_rl.py:env_created",
                "message": "Environment created",
                "data": {
                    "env_obs_space_shape": list(env.observation_space.shape),
                    "env_action_space_shape": list(env.action_space.shape),
                    "env_action_space_low": [float(x) for x in env.action_space.low],
                    "env_action_space_high": [float(x) for x in env.action_space.high]
                },
                "timestamp": int(__import__('time').time() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    # Run episodes
    num_episodes = 5
    success_count = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"\nRunning {num_episodes} evaluation episodes (headless mode)...")
    
    try:
        for ep in range(num_episodes):
            try:
                obs, info = env.reset(seed=ep + 42)  # Use seed for reproducibility
                done = False
                total_reward = 0
                step = 0
                
                print(f"\n--- Episode {ep+1} ---")
                print(f"Initial distance to goal: {info['distance_to_goal']:.2f}m")
                
                # Track action statistics for episode
                actions_this_episode = []
                rewards_this_episode = []
                distances_this_episode = [info['distance_to_goal']]
                
                while not done and step < 1000:  # Match max_steps
                    action, _ = model.predict(obs, deterministic=True)
                    actions_this_episode.append([float(action[0]), float(action[1])])
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    total_reward += reward
                    rewards_this_episode.append(float(reward))
                    distances_this_episode.append(float(info['distance_to_goal']))
                    done = terminated or truncated
                    step += 1
                
                success = info['distance_to_goal'] < 0.3
                success_count += int(success)
                total_rewards.append(total_reward)
                episode_lengths.append(step)
                
                # #region agent log
                # Log episode summary instead of every step
                import json
                import numpy as np
                try:
                    actions_arr = np.array(actions_this_episode)
                    with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "eval-run",
                            "hypothesisId": "H6",
                            "location": "07_evaluate_rl.py:episode_summary",
                            "message": "Episode summary",
                            "data": {
                                "episode": ep+1,
                                "success": bool(success),
                                "total_steps": int(step),
                                "total_reward": float(total_reward),
                                "initial_distance": float(distances_this_episode[0]),
                                "final_distance": float(distances_this_episode[-1]),
                                "progress": float(distances_this_episode[0] - distances_this_episode[-1]),
                                "avg_reward": float(np.mean(rewards_this_episode)) if rewards_this_episode else 0.0,
                                "action_stats": {
                                    "mean_linear": float(np.mean(actions_arr[:, 0])) if len(actions_arr) > 0 else 0.0,
                                    "mean_angular": float(np.mean(actions_arr[:, 1])) if len(actions_arr) > 0 else 0.0,
                                    "std_linear": float(np.std(actions_arr[:, 0])) if len(actions_arr) > 0 else 0.0,
                                    "std_angular": float(np.std(actions_arr[:, 1])) if len(actions_arr) > 0 else 0.0,
                                    "mean_magnitude": float(np.mean(np.sqrt(actions_arr[:, 0]**2 + actions_arr[:, 1]**2))) if len(actions_arr) > 0 else 0.0
                                }
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except Exception as e:
                    pass
                # #endregion
                
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"{status} | Steps: {step}, Reward: {total_reward:.2f}, "
                      f"Final distance: {info['distance_to_goal']:.2f}m")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Cleaning up...")
                break
            except Exception as e:
                print(f"Error in episode {ep+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n=== Evaluation Results ===")
        if len(total_rewards) > 0:
            print(f"Success rate: {success_count}/{len(total_rewards)} ({100*success_count/len(total_rewards):.1f}%)")
            print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
        else:
            print("No episodes completed successfully.")
    
    finally:
        # Always cleanup
        try:
            env.close()
        except:
            pass
        print("\nEvaluation complete.")

if __name__ == "__main__":
    # Always run headless to prevent hangs and crashes
    evaluate(headless=True)

