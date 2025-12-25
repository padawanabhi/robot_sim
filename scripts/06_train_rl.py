#!/usr/bin/env python3
"""Train RL agent for navigation."""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Import to register the environment
# Note: Using importlib because module name starts with number
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv

def make_env(config_path, render_mode=None, randomize=True):
    """Create and wrap environment."""
    env = DiffDriveNavEnv(config_path=config_path, render_mode=render_mode, randomize=randomize, num_obstacles=8)
    env = Monitor(env)
    return env

def train():
    """Main training function."""
    print("Creating environment...")
    
    # Get config path (use complex world for training)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    # Use randomization for better generalization
    env = make_env(config_path, randomize=True)
    
    # Verify environment (skip check to avoid crashes, environment is already tested)
    print("Skipping environment check (already verified)...")
    # Uncomment below if you want to run the check (may cause crashes on macOS)
    # check_env(env, warn=True)
    print("Environment ready for training!")
    
    # Create directories
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create evaluation environment (also randomized for fair evaluation)
    eval_env = make_env(config_path, randomize=True)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(model_dir, 'checkpoints'),
        name_prefix='ppo_nav'
    )
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    
    # Train
    print("Starting training...")
    print("Monitor with: tensorboard --logdir logs/tensorboard")
    
    total_timesteps = 100_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False  # Disable progress bar to avoid dependency issues
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'ppo_nav_final'))
    print("Training complete! Model saved to models/ppo_nav_final.zip")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train()

