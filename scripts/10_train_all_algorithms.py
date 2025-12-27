#!/usr/bin/env python3
"""Train all RL algorithms and compare them."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import argparse

def train_all_algorithms(timesteps=500_000):
    """Train PPO, SAC, and TD3 algorithms."""
    algorithms = ['ppo', 'sac', 'td3']
    
    print("=" * 60)
    print("Training All RL Algorithms")
    print("=" * 60)
    print(f"Timesteps per algorithm: {timesteps:,}")
    print(f"Total training time: ~{len(algorithms) * timesteps / 10000:.0f} minutes")
    print("\nNote: Using single environment with GPU/MPS acceleration for macOS stability")
    print("=" * 60)
    
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Training {algo.upper()}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '06_train_rl.py'),
            '--algorithm', algo,
            '--timesteps', str(timesteps)
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nWarning: {algo.upper()} training failed with exit code {result.returncode}")
        else:
            print(f"\n{algo.upper()} training completed successfully!")
    
    print("\n" + "=" * 60)
    print("All training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Evaluate all algorithms: python scripts/11_evaluate_all_algorithms.py")
    print("2. Compare with classical controllers: python scripts/09_compare_controllers.py")
    print("3. View TensorBoard: tensorboard --logdir logs/tensorboard")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train all RL algorithms')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Number of training timesteps per algorithm (default: 500000)')
    args = parser.parse_args()
    
    train_all_algorithms(timesteps=args.timesteps)

