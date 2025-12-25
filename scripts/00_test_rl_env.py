#!/usr/bin/env python3
"""Quick test to verify RL environment works before training."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import to register the environment
import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv

def test_env():
    """Test the RL environment."""
    print("Testing RL environment...")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world.yaml')
    
    # Create environment with randomization enabled
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few random steps
    print("\nTesting random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={terminated or truncated}, dist_to_goal={info['distance_to_goal']:.2f}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print("  Reset environment")
    
    env.close()
    print("\n✓ RL environment test passed! Ready for training.")

if __name__ == "__main__":
    test_env()

