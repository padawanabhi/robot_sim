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
        # Match robot's actual velocity limits from config (vel_min/vel_max: [-1, 1])
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [dx_norm, dy_norm, dtheta, v, w, dist_to_obstacle, distance_to_goal]
        # dx_norm, dy_norm: normalized direction to goal (unit vector)
        # dtheta: heading error
        # v, w: current velocities
        # dist_to_obstacle: minimum distance to nearest obstacle
        # distance_to_goal: absolute distance to goal
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -np.pi, -1, -1, 0, 0]),
            high=np.array([1, 1, np.pi, 1, 1, 10, 10]),
            dtype=np.float32
        )
        
        self.max_steps = 1000  # Increased to allow more time to reach goal
        self.current_step = 0
        self.prev_distance = None
    
    def _check_collision(self, robot_pos, robot_radius=0.2):
        """
        Check if robot is colliding with any obstacle.
        Returns True if collision detected, False otherwise.
        Uses accurate circle-rectangle collision detection.
        
        Note: IR-SIM stores shape parameters (radius, length, width) directly on the obstacle object,
        not on obs.shape (which is just a string like 'circle' or 'rectangle').
        """
        for obs in self.env.obstacle_list:
            obs_pos = obs.state.flatten()[:2]
            
            # IR-SIM stores shape parameters directly on the obstacle object
            # Check if it's a circle (has radius attribute)
            obs_radius = getattr(obs, 'radius', None)
            
            if obs_radius is not None:
                # Circle obstacle
                dist = np.sqrt((robot_pos[0]-obs_pos[0])**2 + (robot_pos[1]-obs_pos[1])**2)
                if dist < (robot_radius + obs_radius):
                    return True
            else:
                # Try rectangle (has length and width attributes)
                length = getattr(obs, 'length', None)
                width = getattr(obs, 'width', None)
                
                if length is not None and width is not None:
                    # Rectangle obstacle
                    angle = obs.state.flatten()[2] if len(obs.state.flatten()) > 2 else 0.0
                    
                    # Transform robot position to rectangle's local frame
                    dx = robot_pos[0] - obs_pos[0]
                    dy = robot_pos[1] - obs_pos[1]
                    
                    # Rotate to rectangle's frame (rotate by -angle to align with rectangle)
                    cos_a = np.cos(-angle)
                    sin_a = np.sin(-angle)
                    local_x = dx * cos_a - dy * sin_a
                    local_y = dx * sin_a + dy * cos_a
                    
                    # Rectangle half-dimensions
                    half_length = length / 2
                    half_width = width / 2
                    
                    # Find closest point on rectangle to robot center
                    closest_x = np.clip(local_x, -half_length, half_length)
                    closest_y = np.clip(local_y, -half_width, half_width)
                    
                    # Distance from robot center to closest point on rectangle
                    dist_to_rect = np.sqrt((local_x - closest_x)**2 + (local_y - closest_y)**2)
                    
                    # Check if robot circle overlaps with rectangle
                    if dist_to_rect < robot_radius:
                        return True
                else:
                    # Unknown shape - use default circle approximation for collision
                    default_radius = 0.5
                    dist = np.sqrt((robot_pos[0]-obs_pos[0])**2 + (robot_pos[1]-obs_pos[1])**2)
                    if dist < (robot_radius + default_radius):
                        return True
        
        return False
    
    def _get_min_obstacle_distance(self, robot_pos, robot_radius=0.2):
        """
        Get minimum distance to any obstacle, accounting for robot and obstacle sizes.
        
        Note: IR-SIM stores shape parameters (radius, length, width) directly on the obstacle object,
        not on obs.shape (which is just a string like 'circle' or 'rectangle').
        """
        min_dist = 10.0  # Initialize to world size
        
        # Check if obstacle_list exists and has obstacles
        if not hasattr(self.env, 'obstacle_list') or len(self.env.obstacle_list) == 0:
            return 10.0  # No obstacles, return max distance
        
        for obs in self.env.obstacle_list:
            obs_pos = obs.state.flatten()[:2]
            
            # IR-SIM stores shape parameters directly on the obstacle object
            # Check if it's a circle (has radius attribute)
            obs_radius = getattr(obs, 'radius', None)
            
            if obs_radius is not None:
                # Circle obstacle
                dist = np.sqrt((robot_pos[0]-obs_pos[0])**2 + (robot_pos[1]-obs_pos[1])**2)
                dist -= (robot_radius + obs_radius)  # Distance between edges
                min_dist = min(min_dist, dist)
            else:
                # Try rectangle (has length and width attributes)
                length = getattr(obs, 'length', None)
                width = getattr(obs, 'width', None)
                
                if length is not None and width is not None:
                    # Rectangle obstacle
                    angle = obs.state.flatten()[2] if len(obs.state.flatten()) > 2 else 0.0
                    
                    # Transform robot position to rectangle's local frame
                    dx = robot_pos[0] - obs_pos[0]
                    dy = robot_pos[1] - obs_pos[1]
                    
                    # Rotate to rectangle's frame
                    cos_a = np.cos(-angle)
                    sin_a = np.sin(-angle)
                    local_x = dx * cos_a - dy * sin_a
                    local_y = dx * sin_a + dy * cos_a
                    
                    # Distance from robot center to rectangle (expanded by robot radius)
                    half_length = length / 2 + robot_radius
                    half_width = width / 2 + robot_radius
                    
                    # Closest point on expanded rectangle
                    closest_x = np.clip(local_x, -half_length, half_length)
                    closest_y = np.clip(local_y, -half_width, half_width)
                    
                    # Distance to closest point
                    dist = np.sqrt((local_x - closest_x)**2 + (local_y - closest_y)**2)
                    min_dist = min(min_dist, dist)
                else:
                    # Unknown shape - use default circle approximation
                    default_radius = 0.5
                    dist = np.sqrt((robot_pos[0]-obs_pos[0])**2 + (robot_pos[1]-obs_pos[1])**2)
                    dist -= (robot_radius + default_radius)
                    min_dist = min(min_dist, dist)
        
        return max(0.0, min_dist)  # Ensure non-negative
    
    def _get_obs(self):
        """
        Construct observation from simulator state.
        
        Returns:
            observation: [dx_norm, dy_norm, dtheta, v, w, dist_to_obstacle, distance_to_goal]
        """
        robot = self.env.robot_list[0]
        pose = robot.state.flatten()[:3]
        goal = robot.goal.flatten()[:3]
        robot_pos = pose[:2]
        robot_radius = 0.2  # Robot radius from config
        
        # Relative goal position
        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        # Normalize dx, dy by distance (gives direction unit vector)
        if distance_to_goal > 0.01:
            dx_norm = dx / distance_to_goal
            dy_norm = dy / distance_to_goal
        else:
            dx_norm = 0.0
            dy_norm = 0.0
        
        # Heading error to goal
        desired_theta = np.arctan2(dy, dx)
        dtheta = desired_theta - pose[2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # Current velocity (approximate from action)
        v = getattr(self, 'last_v', 0.0)
        w = getattr(self, 'last_w', 0.0)
        
        # Distance to nearest obstacle (accounting for sizes)
        min_dist = self._get_min_obstacle_distance(robot_pos, robot_radius)
        
        # Clamp distance to goal for observation
        distance_to_goal = np.clip(distance_to_goal, 0, 10.0)
        
        return np.array([dx_norm, dy_norm, dtheta, v, w, min_dist, distance_to_goal], dtype=np.float32)
    
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
        
        # Store start position for path efficiency calculation
        self.start_pos = pose[:2].copy()
        self.initial_distance = self.prev_distance
        
        # #region agent log
        # Log episode reset for tracking
        import json
        try:
            with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "eval-run",
                    "hypothesisId": "H6",
                    "location": "05_rl_environment.py:reset:episode_start",
                    "message": "Episode reset",
                    "data": {
                        "initial_distance": float(self.initial_distance),
                        "start_pos": [float(pose[0]), float(pose[1])],
                        "goal_pos": [float(goal[0]), float(goal[1])],
                        "seed": seed
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        
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
        # Store previous position for collision recovery
        robot = self.env.robot_list[0]
        prev_pos = robot.state.flatten()[:2].copy()
        prev_theta = robot.state.flatten()[2]
        
        # Store for observation
        self.last_v = action[0]
        self.last_w = action[1]
        
        # #region agent log
        import json
        action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
        # #endregion
        
        # Step simulation
        self.env.step([action_list])
        self.current_step += 1
        
        # #region agent log
        # Smart sampling: log every 50 steps or at key events (collision, goal, first/last step)
        should_log = (self.current_step % 50 == 0 or self.current_step == 1 or self.current_step >= self.max_steps - 1)
        if should_log:
            try:
                with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                    robot = self.env.robot_list[0]
                    new_pos = robot.state.flatten()[:2]
                    moved = ((new_pos[0] - prev_pos[0])**2 + (new_pos[1] - prev_pos[1])**2)**0.5
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "eval-run",
                        "hypothesisId": "H2",
                        "location": "05_rl_environment.py:step:sampled",
                        "message": "Sampled step (every 50 or key events)",
                        "data": {
                            "step": int(self.current_step),
                            "action": [float(action[0]), float(action[1])],
                            "action_list": action_list,
                            "distance_moved": float(moved),
                            "prev_pos": [float(prev_pos[0]), float(prev_pos[1])],
                            "new_pos": [float(new_pos[0]), float(new_pos[1])]
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
        # #endregion
        
        # Get robot position for collision check
        robot = self.env.robot_list[0]
        robot_pos = robot.state.flatten()[:2]
        robot_radius = 0.2
        
        # Check for collision AFTER step (IR-SIM doesn't prevent collisions)
        collision = self._check_collision(robot_pos, robot_radius)
        
        # If collision detected, reset robot to previous position
        if collision:
            # Reset robot to previous position to prevent passing through obstacles
            robot.state[0, 0] = prev_pos[0]
            robot.state[1, 0] = prev_pos[1]
            robot.state[2, 0] = prev_theta
        
        # Get observation
        obs = self._get_obs()
        info = self._get_info()
        
        # Compute reward
        reward = self._compute_reward(obs, info)
        
        # Check termination (ensure boolean types)
        terminated = bool(info["distance_to_goal"] < 0.3)  # goal reached
        truncated = bool(self.current_step >= self.max_steps)
        
        # Collision handling - Balanced with goal bonus
        if collision:
            reward -= 200.0  # Severe penalty for collision
            truncated = True
        elif obs[5] < 0.2:  # Very close to obstacle (but not colliding)
            reward -= 10.0 * (0.2 - obs[5]) / 0.2  # Stronger penalty as getting closer
        
        # Goal bonus - Balanced with collision penalty (1:1 ratio)
        if terminated:
            reward += 200.0  # Increased to match collision penalty magnitude
        
        # #region agent log
        # CRITICAL: Log when agent is close to goal to understand why it backs away
        current_distance = info["distance_to_goal"]
        if current_distance < 3.0:  # Log every step when close
            try:
                with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "eval-run",
                        "hypothesisId": "H1",
                        "location": "05_rl_environment.py:step:close_to_goal",
                        "message": "Agent close to goal - detailed analysis",
                        "data": {
                            "step": int(self.current_step),
                            "distance_to_goal": float(current_distance),
                            "action": [float(action[0]), float(action[1])],
                            "reward": float(reward),
                            "obs": [float(x) for x in obs],
                            "collision": bool(collision),
                            "terminated": bool(terminated),
                            "prev_distance": float(getattr(self, '_prev_distance', current_distance))
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
        self._prev_distance = current_distance
        
        # Log key events: collision, goal reached, episode end
        if collision or terminated or truncated or (self.current_step % 50 == 0):
            try:
                with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "eval-run",
                        "hypothesisId": "H4",
                        "location": "05_rl_environment.py:step:reward",
                        "message": "Reward and termination status",
                        "data": {
                            "step": int(self.current_step),
                            "reward": float(reward),
                            "distance_to_goal": float(info["distance_to_goal"]),
                            "collision": bool(collision),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated)
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
        # #endregion
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs, info):
        """
        SIMPLIFIED REWARD FUNCTION - Single clear signal: distance to goal.
        
        The previous reward function was too complex with many competing signals,
        causing the agent to learn suboptimal policies. This simplified version
        provides ONE clear signal: get closer to goal = more reward.
        
        Components:
        1. Primary: Inverse distance reward (stronger when closer)
        2. Collision penalty (critical for safety)
        3. Goal bonus (for reaching goal)
        
        This ensures the agent always knows: closer = better.
        """
        current_distance = info["distance_to_goal"]
        
        # Track progress for logging
        progress = self.prev_distance - current_distance
        self.prev_distance = current_distance
        
        # Base reward starts at 0
        reward = 0.0
        
        # PRIMARY REWARD: Inverse distance to goal (simple, clear, always decreasing as agent gets closer)
        # This is the ONLY distance-based signal - no competing bonuses
        # Scale: At 10m: 0 reward, At 5m: 50 reward, At 1m: 90 reward, At 0.3m: 97 reward
        # This creates a strong, consistent gradient toward the goal
        max_distance = 10.0  # Maximum expected distance
        distance_reward = 100.0 * (1.0 - current_distance / max_distance)
        reward += distance_reward
        
        # #region agent log
        # Log reward components for debugging
        should_log = (self.current_step % 100 == 0 or current_distance < 3.0)
        if should_log:
            import json
            try:
                with open('/Users/abhi/Desktop/Github_Projects/robot_sim/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "eval-run",
                        "hypothesisId": "H1",
                        "location": "05_rl_environment.py:_compute_reward:simplified",
                        "message": "Simplified reward tracking",
                        "data": {
                            "step": int(self.current_step),
                            "current_distance": float(current_distance),
                            "distance_reward": float(distance_reward),
                            "progress": float(progress)
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
        # #endregion
        
        # Collision penalty (keep this - it's critical for safety)
        # Note: Collision is handled in step() method, but we keep obstacle proximity penalty here
        min_obstacle_dist = obs[5]
        if min_obstacle_dist < 0.2:
            reward -= 10.0  # Penalty for being very close to obstacle
        elif min_obstacle_dist < 0.4:
            reward -= 5.0  # Moderate penalty
        elif min_obstacle_dist < 0.6:
            reward -= 1.0  # Small penalty
        
        # Goal reached bonus (large bonus for success)
        if current_distance < 0.3:
            reward += 500.0  # Large bonus for reaching goal (much larger than distance reward to ensure it's attractive)
        
        # Small time penalty to encourage efficiency (but not too large to avoid discouraging exploration)
        reward -= 0.01
        
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

