"""Potential field controller module for reuse."""
import numpy as np

class PotentialFieldController:
    def __init__(self, k_att=1.0, k_rep=0.5, d0=1.5):
        """
        k_att: attractive potential gain
        k_rep: repulsive potential gain
        d0: influence distance of obstacles
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
    
    def attractive_force(self, pos, goal):
        """Compute attractive force toward goal."""
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            return np.array([0.0, 0.0])
        
        return self.k_att * np.array([dx, dy]) / dist
    
    def repulsive_force(self, pos, obstacles):
        """Compute repulsive force from obstacles."""
        force = np.array([0.0, 0.0])
        
        for obs in obstacles:
            obs_pos = obs.state.flatten()[:2]
            dx = pos[0] - obs_pos[0]
            dy = pos[1] - obs_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Obstacle radius (approximate)
            obs_radius = getattr(obs.shape, 'radius', 0.5)
            dist = max(dist - obs_radius, 0.01)
            
            if dist < self.d0:
                magnitude = self.k_rep * (1/dist - 1/self.d0) / (dist**2)
                force += magnitude * np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)
        
        return force
    
    def compute(self, pos, theta, goal, obstacles):
        """Compute velocity from potential field."""
        # Total force
        f_att = self.attractive_force(pos, goal)
        f_rep = self.repulsive_force(pos, obstacles)
        f_total = f_att + f_rep
        
        # Convert force to velocity commands
        magnitude = np.linalg.norm(f_total)
        if magnitude < 0.01:
            return [0.0, 0.0]
        
        # Desired heading
        desired_theta = np.arctan2(f_total[1], f_total[0])
        
        # Angular error
        theta_error = desired_theta - theta
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
        
        # Velocities
        v = min(magnitude, 1.0)
        w = np.clip(2.0 * theta_error, -1.0, 1.0)
        
        # Slow down if turning sharply
        if abs(theta_error) > 0.5:
            v *= 0.5
        
        return [v, w]

