#!/usr/bin/env python3
"""Evaluate and compare all trained RL algorithms."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv
import numpy as np

def evaluate_algorithm(algorithm, num_episodes=10):
    """Evaluate a single algorithm."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Try algorithm-specific directory first, then root
    algo_dir = os.path.join(model_dir, algorithm.lower())
    model_path = os.path.join(algo_dir, f'{algorithm.lower()}_nav_final')
    
    if not os.path.exists(model_path + '.zip'):
        # Try best model
        model_path = os.path.join(algo_dir, 'best_model')
        if not os.path.exists(model_path + '.zip'):
            # Try root directory
            model_path = os.path.join(model_dir, f'{algorithm.lower()}_nav_final')
            if not os.path.exists(model_path + '.zip'):
                print(f"  Model not found for {algorithm.upper()}")
                return None
    
    print(f"  Loading {algorithm.upper()} model from {model_path}...")
    
    # Load model
    if algorithm.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algorithm.lower() == 'sac':
        model = SAC.load(model_path)
    elif algorithm.lower() == 'td3':
        model = TD3.load(model_path)
    else:
        return None
    
    # Create environment
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    # Run episodes
    success_count = 0
    total_rewards = []
    episode_lengths = []
    final_distances = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        
        success = info['distance_to_goal'] < 0.3
        success_count += int(success)
        total_rewards.append(total_reward)
        episode_lengths.append(step)
        final_distances.append(info['distance_to_goal'])
    
    env.close()
    
    return {
        'algorithm': algorithm.upper(),
        'success_rate': success_count / num_episodes,
        'success_count': success_count,
        'num_episodes': num_episodes,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_steps': np.mean(episode_lengths),
        'std_steps': np.std(episode_lengths),
        'avg_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances)
    }

def evaluate_all_algorithms(num_episodes=10):
    """Evaluate all trained algorithms."""
    algorithms = ['ppo', 'sac', 'td3']
    
    print("=" * 70)
    print("Evaluating All RL Algorithms")
    print("=" * 70)
    print(f"Episodes per algorithm: {num_episodes}")
    print("=" * 70)
    
    results = {}
    
    for algo in algorithms:
        print(f"\nEvaluating {algo.upper()}...")
        result = evaluate_algorithm(algo, num_episodes)
        if result:
            results[algo] = result
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Algorithm':<12} {'Success Rate':<15} {'Avg Reward':<15} {'Avg Steps':<12} {'Avg Final Dist':<15}")
    print("-" * 70)
    
    for algo in algorithms:
        if algo in results:
            r = results[algo]
            print(f"{r['algorithm']:<12} {r['success_rate']*100:>6.1f}% ({r['success_count']}/{r['num_episodes']})  "
                  f"{r['avg_reward']:>8.2f} ± {r['std_reward']:>5.2f}  "
                  f"{r['avg_steps']:>6.1f} ± {r['std_steps']:>4.1f}  "
                  f"{r['avg_final_distance']:>6.2f} ± {r['std_final_distance']:>5.2f}")
        else:
            print(f"{algo.upper():<12} {'Model not found':<15}")
    
    print("=" * 70)
    
    # Find best algorithm
    if results:
        best_success = max(results.values(), key=lambda x: x['success_rate'])
        best_reward = max(results.values(), key=lambda x: x['avg_reward'])
        
        print(f"\nBest Success Rate: {best_success['algorithm']} ({best_success['success_rate']*100:.1f}%)")
        print(f"Best Average Reward: {best_reward['algorithm']} ({best_reward['avg_reward']:.2f})")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes per algorithm (default: 10)')
    args = parser.parse_args()
    
    evaluate_all_algorithms(num_episodes=args.episodes)

