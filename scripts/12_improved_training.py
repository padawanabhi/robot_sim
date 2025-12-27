#!/usr/bin/env python3
"""Train improved RL model with better reward function and longer training."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import importlib.util
import torch
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv
import argparse

def make_env(config_path, render_mode=None, randomize=True):
    """Create and wrap environment."""
    env = DiffDriveNavEnv(config_path=config_path, render_mode=render_mode, randomize=randomize, num_obstacles=8)
    env = Monitor(env)
    return env

def get_device():
    """Detect and return the best available device (GPU/MPS/CPU)."""
    if torch.backends.mps.is_available():
        print("✓ Using Apple Silicon GPU (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        print("✓ Using CUDA GPU")
        return "cuda"
    else:
        print("⚠ Using CPU (no GPU detected)")
        return "cpu"

def train_improved(timesteps=500_000):
    """Train with improved reward function and longer duration."""
    print("=" * 70)
    print("Improved RL Training")
    print("=" * 70)
    print("Improvements:")
    print("  - Better reward function (more encouraging of movement)")
    print("  - Improved observation space (normalized directions)")
    print("  - Longer training duration")
    print("=" * 70)
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    
    # Use single environment to avoid crashes on macOS
    # GPU acceleration will make up for the lack of parallel environments
    print("Creating single environment (GPU-accelerated)...")
    env = make_env(config_path, randomize=True)
    print("Environment ready for training!")
    
    # Detect and use GPU if available
    device = get_device()
    
    # Create directories
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ppo_improved')
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    
    # Create evaluation environment
    eval_env = make_env(config_path, randomize=True)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='ppo_improved'
    )
    
    # Create improved model
    print(f"\nCreating improved PPO model...")
    tensorboard_log = os.path.join(log_dir, 'tensorboard', 'PPO_IMPROVED')
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Back to 2048 since we're using single env (GPU will speed up)
        batch_size=64,
        n_epochs=10,   # Back to 10 for better learning
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        tensorboard_log=tensorboard_log,
        device=device  # Use GPU/MPS if available
    )
    
    # Train
    print(f"\nStarting improved training for {timesteps:,} timesteps...")
    print(f"Monitor with: tensorboard --logdir logs/tensorboard")
    print("=" * 70)
    
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'ppo_improved_final')
    model.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}.zip")
    print("=" * 70)
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train improved RL model')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Number of training timesteps (default: 500000)')
    args = parser.parse_args()
    
    train_improved(timesteps=args.timesteps)

