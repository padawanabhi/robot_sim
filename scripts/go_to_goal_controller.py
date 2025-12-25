"""Go-to-goal controller module for reuse."""
import numpy as np

class GoToGoalController:
    def __init__(self, k_rho=0.5, k_alpha=1.5, k_beta=-0.3, obstacle_safety_dist=1.2):
        """
        Proportional controller gains with obstacle avoidance.
        k_rho: distance gain
        k_alpha: heading to goal gain  
        k_beta: final orientation gain
        obstacle_safety_dist: minimum distance to maintain from obstacles
        """
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.obstacle_safety_dist = obstacle_safety_dist
    
    def compute(self, current_pose, goal_pose, obstacles=None):
        """
        Compute velocity commands to reach goal with obstacle avoidance.
        
        Args:
            current_pose: [x, y, theta]
            goal_pose: [x_goal, y_goal, theta_goal]
            obstacles: list of obstacle objects (optional)
        
        Returns:
            [v, w]: linear and angular velocity
        """
        x, y, theta = current_pose
        x_g, y_g, theta_g = goal_pose
        
        # Distance to goal
        dx = x_g - x
        dy = y_g - y
        rho = np.sqrt(dx**2 + dy**2)
        
        # Angle to goal
        alpha = np.arctan2(dy, dx) - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize
        
        # Orientation error at goal
        beta = theta_g - theta - alpha
        beta = np.arctan2(np.sin(beta), np.cos(beta))
        
        # Check if at goal
        if rho < 0.1:
            return [0.0, 0.0]
        
        # Obstacle avoidance: adjust heading if obstacle is too close
        if obstacles is not None:
            min_dist = float('inf')
            closest_obstacle_angle = 0.0
            obstacle_in_path = False
            
            # Check if any obstacle is in the robot's path
            for obs in obstacles:
                obs_pos = obs.state.flatten()[:2]
                obs_dx = obs_pos[0] - x
                obs_dy = obs_pos[1] - y
                obs_dist = np.sqrt(obs_dx**2 + obs_dy**2)
                
                # Get obstacle radius
                obs_radius = getattr(obs.shape, 'radius', 0.5)
                safe_dist = obs_radius + self.obstacle_safety_dist
                
                # Check if obstacle is in the direction we're heading
                obs_angle = np.arctan2(obs_dy, obs_dx)
                heading_to_obs = obs_angle - theta
                heading_to_obs = np.arctan2(np.sin(heading_to_obs), np.cos(heading_to_obs))
                
                # If obstacle is in front and close
                if abs(heading_to_obs) < np.pi/3 and obs_dist < safe_dist:
                    obstacle_in_path = True
                    
                    if obs_dist < min_dist:
                        min_dist = obs_dist
                        closest_obstacle_angle = obs_angle
            
            # If obstacle is in path, prioritize avoidance
            if obstacle_in_path and min_dist < safe_dist:
                # Compute avoidance direction (turn right or left based on which is better)
                # Check which side has more clearance
                right_angle = closest_obstacle_angle + np.pi/2
                left_angle = closest_obstacle_angle - np.pi/2
                
                # Prefer turning right (positive angle)
                avoidance_heading = right_angle
                alpha = avoidance_heading - theta
                alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
                
                # Slow down significantly when avoiding
                v = self.k_rho * rho * 0.2
                
                # If very close, stop and turn
                if min_dist < self.obstacle_safety_dist * 0.7:
                    v = 0.1
            else:
                # Normal goal-seeking behavior
                v = self.k_rho * rho
        else:
            v = self.k_rho * rho
        
        # Compute angular velocity
        w = self.k_alpha * alpha + self.k_beta * beta
        
        # Slow down if turning sharply
        if abs(alpha) > 0.5:
            v *= 0.6
        
        # Clamp velocities
        v = np.clip(v, -1.0, 1.0)
        w = np.clip(w, -1.0, 1.0)
        
        return [v, w]

