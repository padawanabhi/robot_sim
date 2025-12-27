#!/usr/bin/env python3
"""Train RL agent for navigation."""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import argparse
import torch
import json
import os

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

class TrainingLogCallback(BaseCallback):
    """Minimal callback to log training progress at checkpoints."""
    def __init__(self, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_path = '/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log'
    
    def _on_step(self) -> bool:
        # Only log at checkpoints to avoid flooding
        if self.num_timesteps % self.log_freq == 0:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "training-run",
                        "hypothesisId": "H3",
                        "location": "06_train_rl.py:TrainingLogCallback",
                        "message": "Training checkpoint",
                        "data": {
                            "num_timesteps": int(self.num_timesteps),
                            "episode_rewards": float(self.locals.get('infos', [{}])[0].get('episode', {}).get('r', 0.0)) if self.locals.get('infos') else None,
                            "episode_length": float(self.locals.get('infos', [{}])[0].get('episode', {}).get('l', 0.0)) if self.locals.get('infos') else None
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
        return True

def train(algorithm='ppo', timesteps=300_000):
    """
    Main training function.
    
    Args:
        algorithm: 'ppo', 'sac', or 'td3'
        timesteps: Number of training timesteps
    """
    print(f"Creating environment for {algorithm.upper()} training...")
    
    # Get config path (use complex world for training)
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
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    algo_dir = os.path.join(model_dir, algorithm.lower())
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(algo_dir, exist_ok=True)
    os.makedirs(os.path.join(algo_dir, 'checkpoints'), exist_ok=True)
    
    # Create evaluation environment (single env, also randomized for fair evaluation)
    eval_env = make_env(config_path, randomize=True)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=algo_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(algo_dir, 'checkpoints'),
        name_prefix=f'{algorithm.lower()}_nav'
    )
    
    # Minimal training log callback (logs every 10k steps)
    training_log_callback = TrainingLogCallback(log_freq=10000)
    
    # Create model based on algorithm
    print(f"Creating {algorithm.upper()} model...")
    tensorboard_log = os.path.join(log_dir, 'tensorboard', algorithm.upper())
    
    if algorithm.lower() == 'ppo':
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
            vf_coef=0.5,    # Value function coefficient
            tensorboard_log=tensorboard_log,
            device=device  # Use GPU/MPS if available
        )
    elif algorithm.lower() == 'sac':
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            tensorboard_log=tensorboard_log,
            device=device  # Use GPU/MPS if available
        )
    elif algorithm.lower() == 'td3':
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            tensorboard_log=tensorboard_log,
            device=device  # Use GPU/MPS if available
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'ppo', 'sac', or 'td3'")
    
    # Train
    print(f"Starting {algorithm.upper()} training for {timesteps:,} timesteps...")
    print(f"Monitor with: tensorboard --logdir logs/tensorboard")
    
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback, training_log_callback],
        progress_bar=False  # Disable progress bar to avoid dependency issues
    )
    
    # Save final model
    final_model_path = os.path.join(algo_dir, f'{algorithm.lower()}_nav_final')
    model.save(final_model_path)
    print(f"Training complete! Model saved to {final_model_path}.zip")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agent for navigation')
    parser.add_argument('--algorithm', type=str, default='ppo', 
                       choices=['ppo', 'sac', 'td3'],
                       help='RL algorithm to use (default: ppo)')
    parser.add_argument('--timesteps', type=int, default=300_000,
                       help='Number of training timesteps (default: 300000)')
    args = parser.parse_args()
    
    train(algorithm=args.algorithm, timesteps=args.timesteps)

