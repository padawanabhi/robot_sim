#!/usr/bin/env python3
"""Gymnasium-compatible wrapper for IR-SIM."""
import sys
import os

# Set non-interactive matplotlib backend BEFORE importing irsim
# This prevents crashes during training
if 'MPLBACKEND' not in os.environ:
    os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import irsim
import yaml
import tempfile

class DiffDriveNavEnv(gym.Env):
    """Custom Gymnasium environment for differential drive navigation."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config_path=None, render_mode=None, randomize=True, num_obstacles=8):
        super().__init__()
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'rl_world.yaml')
        
        self.config_path = config_path
        self.render_mode = render_mode
        self.randomize = randomize
        self.num_obstacles = num_obstacles
        self.env = None
        self.np_random = None
        
        # Action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [dx, dy, dtheta, v, w, dist_to_obstacle]
        # dx, dy: relative goal position
        # dtheta: heading error
        # v, w: current velocities
        # dist_to_obstacle: minimum distance to nearest obstacle
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -np.pi, -1, -1, 0]),
            high=np.array([10, 10, np.pi, 1, 1, 10]),
            dtype=np.float32
        )
        
        self.max_steps = 500
        self.current_step = 0
        self.prev_distance = None
    
    def _get_obs(self):
        """Construct observation from simulator state."""
        robot = self.env.robot_list[0]
        pose = robot.state.flatten()[:3]
        goal = robot.goal.flatten()[:3]
        
        # Relative goal position
        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        
        # Heading error to goal
        desired_theta = np.arctan2(dy, dx)
        dtheta = desired_theta - pose[2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # Current velocity (approximate from action)
        v = getattr(self, 'last_v', 0.0)
        w = getattr(self, 'last_w', 0.0)
        
        # Distance to nearest obstacle
        min_dist = 10.0
        for obs in self.env.obstacle_list:
            obs_pos = obs.state.flatten()[:2]
            dist = np.sqrt((pose[0]-obs_pos[0])**2 + (pose[1]-obs_pos[1])**2)
            obs_radius = getattr(obs.shape, 'radius', 0.5)
            dist -= obs_radius
            min_dist = min(min_dist, dist)
        
        return np.array([dx, dy, dtheta, v, w, min_dist], dtype=np.float32)
    
    def _get_info(self):
        """Additional info for debugging."""
        robot = self.env.robot_list[0]
        pose = robot.state.flatten()[:3]
        goal = robot.goal.flatten()[:3]
        distance = float(np.sqrt((goal[0]-pose[0])**2 + (goal[1]-pose[1])**2))
        return {"distance_to_goal": distance, "steps": int(self.current_step)}
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize random number generator
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()
        
        if self.env is not None:
            self.env.end()
        
        # Create environment with randomization if enabled
        # Disable plotting for training (headless mode)
        if self.randomize:
            self.env = self._create_randomized_env()
        else:
            # Try to disable plotting if supported
            try:
                self.env = irsim.make(self.config_path, disable_all_plot=True)
            except TypeError:
                # Fallback if parameter not supported
                self.env = irsim.make(self.config_path)
        
        self.current_step = 0
        self.last_v = 0.0
        self.last_w = 0.0
        
        # Compute initial distance for reward shaping
        robot = self.env.robot_list[0]
        pose = robot.state.flatten()[:3]
        goal = robot.goal.flatten()[:3]
        self.prev_distance = np.sqrt((goal[0]-pose[0])**2 + (goal[1]-pose[1])**2)
        
        return self._get_obs(), self._get_info()
    
    def _create_randomized_env(self):
        """Create environment with randomized obstacles and goal."""
        # Load base config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Randomize robot start position (but keep it away from edges)
        start_x = self.np_random.uniform(1.0, 2.5)
        start_y = self.np_random.uniform(1.0, 2.5)
        start_theta = self.np_random.uniform(-np.pi, np.pi)
        
        # Randomize goal position (far from start, away from edges)
        goal_x = self.np_random.uniform(7.0, 9.0)
        goal_y = self.np_random.uniform(7.0, 9.0)
        goal_theta = self.np_random.uniform(-np.pi, np.pi)
        
        # Ensure goal is far enough from start
        while np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2) < 5.0:
            goal_x = self.np_random.uniform(7.0, 9.0)
            goal_y = self.np_random.uniform(7.0, 9.0)
        
        config['robot'][0]['state'] = [start_x, start_y, start_theta]
        config['robot'][0]['goal'] = [goal_x, goal_y, goal_theta]
        
        # Generate random obstacles
        obstacles = []
        for i in range(self.num_obstacles):
            # Try to place obstacle
            for attempt in range(20):  # Try up to 20 times
                obs_x = self.np_random.uniform(1.5, 8.5)
                obs_y = self.np_random.uniform(1.5, 8.5)
                
                # Check distance from start
                dist_from_start = np.sqrt((obs_x - start_x)**2 + (obs_y - start_y)**2)
                # Check distance from goal
                dist_from_goal = np.sqrt((obs_x - goal_x)**2 + (obs_y - goal_y)**2)
                
                # Ensure obstacle is not too close to start or goal
                if dist_from_start > 1.5 and dist_from_goal > 1.5:
                    # Check distance from other obstacles
                    too_close = False
                    for existing_obs in obstacles:
                        ex_x, ex_y = existing_obs['state'][:2]
                        if np.sqrt((obs_x - ex_x)**2 + (obs_y - ex_y)**2) < 1.0:
                            too_close = True
                            break
                    
                    if not too_close:
                        # Random obstacle type and size
                        if self.np_random.random() < 0.7:  # 70% circles
                            radius = self.np_random.uniform(0.25, 0.5)
                            obstacles.append({
                                'shape': {'name': 'circle', 'radius': radius},
                                'state': [obs_x, obs_y],
                                'color': 'red'
                            })
                        else:  # 30% rectangles
                            length = self.np_random.uniform(1.0, 2.0)
                            width = self.np_random.uniform(0.2, 0.4)
                            angle = self.np_random.uniform(0, 2*np.pi)
                            obstacles.append({
                                'shape': {'name': 'rectangle', 'length': length, 'width': width},
                                'state': [obs_x, obs_y, angle],
                                'color': 'gray'
                            })
                        break
        
        config['obstacle'] = obstacles
        
        # Save temporary config and load it
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_config_path = f.name
            
            # Try to disable plotting if supported
            try:
                env = irsim.make(temp_config_path, disable_all_plot=True)
            except TypeError:
                env = irsim.make(temp_config_path)
            
            # Clean up temp file after a short delay to ensure it's loaded
            import time
            time.sleep(0.1)
            try:
                os.unlink(temp_config_path)
            except:
                pass  # Ignore cleanup errors
            
            return env
        except Exception as e:
            # If randomization fails, fall back to base config
            print(f"Warning: Randomization failed, using base config: {e}")
            try:
                return irsim.make(self.config_path, disable_all_plot=True)
            except TypeError:
                return irsim.make(self.config_path)
    
    def step(self, action):
        """Execute action and return new state."""
        # Store for observation
        self.last_v = action[0]
        self.last_w = action[1]
        
        # Step simulation
        self.env.step([action.tolist()])
        self.current_step += 1
        
        # Get observation
        obs = self._get_obs()
        info = self._get_info()
        
        # Compute reward
        reward = self._compute_reward(obs, info)
        
        # Check termination (ensure boolean types)
        terminated = bool(info["distance_to_goal"] < 0.3)  # goal reached
        truncated = bool(self.current_step >= self.max_steps)
        
        # Collision check
        if obs[5] < 0.1:  # too close to obstacle
            reward -= 10.0
            truncated = True
        
        # Goal bonus
        if terminated:
            reward += 100.0
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs, info):
        """Shaped reward for navigation."""
        current_distance = info["distance_to_goal"]
        
        # Progress reward
        progress = self.prev_distance - current_distance
        self.prev_distance = current_distance
        
        # Base reward components
        reward = 0.0
        reward += 5.0 * progress  # reward for getting closer
        reward -= 0.01  # small time penalty
        
        # Heading alignment bonus
        heading_error = abs(obs[2])
        if heading_error < 0.3:
            reward += 0.1
        
        # Obstacle proximity penalty
        if obs[5] < 0.5:
            reward -= 0.5 * (0.5 - obs[5])
        
        return reward
    
    def render(self):
        """Render the environment."""
        # Only render if explicitly requested and not in training mode
        if self.render_mode == "human" and self.env is not None:
            try:
                self.env.render()
            except:
                pass  # Silently fail if rendering causes issues
    
    def close(self):
        """Clean up."""
        if self.env is not None:
            self.env.end()

# Register the environment
gym.register(
    id='DiffDriveNav-v0',
    entry_point='scripts.05_rl_environment:DiffDriveNavEnv',
)

