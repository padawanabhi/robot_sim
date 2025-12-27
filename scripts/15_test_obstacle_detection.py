#!/usr/bin/env python3
"""Test obstacle detection in RL environment."""
import sys
import os

# Set non-interactive backend
os.environ['MPLBACKEND'] = 'Agg'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
spec = importlib.util.spec_from_file_location("rl_env", os.path.join(os.path.dirname(__file__), "05_rl_environment.py"))
rl_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_env_module)
DiffDriveNavEnv = rl_env_module.DiffDriveNavEnv
import numpy as np

def test_obstacle_detection():
    """Test if obstacles are being created and detected."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world_complex.yaml')
    
    print("Testing obstacle detection...")
    print("=" * 70)
    
    # Create environment
    env = DiffDriveNavEnv(config_path=config_path, render_mode=None, randomize=True, num_obstacles=8)
    
    # Reset to get initial state
    obs, info = env.reset(seed=42)
    
    # Check obstacles
    print(f"\n1. Environment Check:")
    print(f"   Has obstacle_list: {hasattr(env.env, 'obstacle_list')}")
    
    if hasattr(env.env, 'obstacle_list'):
        num_obstacles = len(env.env.obstacle_list)
        print(f"   Number of obstacles: {num_obstacles}")
        
        if num_obstacles == 0:
            print("   ⚠️ ERROR: No obstacles found!")
            return
        
        # Get robot position
        robot = env.env.robot_list[0]
        robot_pos = robot.state.flatten()[:2]
        print(f"   Robot position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
        
        # Check each obstacle
        print(f"\n2. Obstacle Details:")
        for i, obs_obj in enumerate(env.env.obstacle_list):
            obs_state = obs_obj.state.flatten()
            obs_pos = obs_state[:2]
            
            # Get shape type (IR-SIM stores this as a string)
            shape_name = str(obs_obj.shape) if isinstance(obs_obj.shape, str) else 'unknown'
            shape_type = shape_name
            
            # Get shape parameters directly from obstacle object (not from obs.shape)
            obs_radius = getattr(obs_obj, 'radius', None)
            length = getattr(obs_obj, 'length', None)
            width = getattr(obs_obj, 'width', None)
            
            # Set size string
            if shape_name == 'circle' and obs_radius is not None:
                size = f"radius={obs_radius:.2f}"
            elif shape_name == 'rectangle' and length is not None and width is not None:
                size = f"length={length:.2f}, width={width:.2f}"
            else:
                # Use default values if we can't find the actual parameters
                if shape_name == 'circle':
                    size = "radius=0.5 (default - not found)"
                elif shape_name == 'rectangle':
                    size = "length=1.0, width=0.3 (default - not found)"
                else:
                    size = 'unknown'
            
            # Calculate distance
            dist = np.sqrt((robot_pos[0] - obs_pos[0])**2 + (robot_pos[1] - obs_pos[1])**2)
            
            print(f"   Obstacle {i+1}: {shape_type} at ({obs_pos[0]:.2f}, {obs_pos[1]:.2f}), {size}")
            print(f"                Distance from robot: {dist:.2f}m")
        
        # Test distance calculation
        print(f"\n3. Testing Distance Calculation:")
        min_dist = env._get_min_obstacle_distance(robot_pos, robot_radius=0.2)
        print(f"   Min obstacle distance (from function): {min_dist:.2f}m")
        print(f"   Min obstacle distance (from observation): {obs[5]:.2f}m")
        
        if min_dist >= 9.9:
            print(f"   ⚠️ WARNING: Distance is at maximum (10.0m), obstacles may not be detected!")
        
        # Test at different positions
        print(f"\n4. Testing at Different Robot Positions:")
        test_positions = [
            [5.0, 5.0],  # Center
            [1.0, 1.0],  # Near start
            [8.0, 8.0],  # Near goal
        ]
        
        for test_pos in test_positions:
            test_dist = env._get_min_obstacle_distance(test_pos, robot_radius=0.2)
            print(f"   Position ({test_pos[0]:.1f}, {test_pos[1]:.1f}): min_dist = {test_dist:.2f}m")
    
    else:
        print("   ⚠️ ERROR: obstacle_list attribute not found!")
    
    env.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_obstacle_detection()

