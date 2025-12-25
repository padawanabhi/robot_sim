#!/usr/bin/env python3
"""Differential drive odometry with noise simulation."""
import sys
import os

# Set matplotlib backend via environment variable BEFORE any imports
os.environ['MPLBACKEND'] = 'TkAgg'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import irsim
import matplotlib.pyplot as plt

class DifferentialDriveOdometry:
    def __init__(self, wheel_base=0.3, wheel_radius=0.05):
        """
        wheel_base: distance between wheels (meters)
        wheel_radius: wheel radius (meters)
        """
        self.L = wheel_base
        self.r = wheel_radius
        
        # Estimated pose [x, y, theta]
        self.pose = np.array([0.0, 0.0, 0.0])
        
        # Noise parameters (simulate real sensor noise)
        self.noise_v = 0.02  # linear velocity noise std
        self.noise_w = 0.05  # angular velocity noise std
        
        # History for plotting
        self.estimated_trajectory = []
        self.true_trajectory = []
    
    def reset(self, initial_pose):
        """Reset odometry to initial pose."""
        self.pose = np.array(initial_pose, dtype=float)
        self.estimated_trajectory = [self.pose.copy()]
        self.true_trajectory = []
    
    def update(self, v, w, dt, true_pose=None):
        """
        Update pose estimate from velocity commands.
        
        Args:
            v: linear velocity command
            w: angular velocity command
            dt: timestep
            true_pose: actual pose for comparison (optional)
        """
        # Add noise to simulate real sensor readings
        v_noisy = v + np.random.normal(0, self.noise_v)
        w_noisy = w + np.random.normal(0, self.noise_w)
        
        # Update pose using differential drive kinematics
        theta = self.pose[2]
        
        if abs(w_noisy) < 1e-6:
            # Straight line motion
            self.pose[0] += v_noisy * np.cos(theta) * dt
            self.pose[1] += v_noisy * np.sin(theta) * dt
        else:
            # Arc motion
            self.pose[0] += (v_noisy/w_noisy) * (np.sin(theta + w_noisy*dt) - np.sin(theta))
            self.pose[1] += (v_noisy/w_noisy) * (np.cos(theta) - np.cos(theta + w_noisy*dt))
            self.pose[2] += w_noisy * dt
        
        # Normalize theta to [-pi, pi]
        self.pose[2] = np.arctan2(np.sin(self.pose[2]), np.cos(self.pose[2]))
        
        # Store for visualization
        self.estimated_trajectory.append(self.pose.copy())
        if true_pose is not None:
            self.true_trajectory.append(true_pose.copy())
        
        return self.pose
    
    def get_error(self):
        """Compute position and orientation error."""
        if len(self.true_trajectory) == 0:
            return 0, 0
        
        true = np.array(self.true_trajectory[-1])
        est = self.pose
        
        pos_error = np.sqrt((true[0]-est[0])**2 + (true[1]-est[1])**2)
        ori_error = abs(true[2] - est[2])
        
        return pos_error, ori_error
    
    def plot_comparison(self):
        """Plot estimated vs true trajectory."""
        if len(self.true_trajectory) == 0:
            print("No true trajectory data to compare")
            return
        
        est = np.array(self.estimated_trajectory)
        true = np.array(self.true_trajectory)
        
        plt.figure(figsize=(10, 8))
        plt.plot(true[:, 0], true[:, 1], 'g-', label='True Path', linewidth=2)
        plt.plot(est[:, 0], est[:, 1], 'r--', label='Estimated (Odometry)', linewidth=2)
        plt.scatter([est[0, 0]], [est[0, 1]], c='blue', s=100, marker='o', label='Start')
        plt.scatter([est[-1, 0]], [est[-1, 1]], c='red', s=100, marker='x', label='End (Est)')
        plt.scatter([true[-1, 0]], [true[-1, 1]], c='green', s=100, marker='x', label='End (True)')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Odometry Drift Visualization')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Save plot
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        plt.savefig(os.path.join(log_dir, 'odometry_comparison.png'), dpi=150)
        plt.show()

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'simple_world.yaml')
    env = irsim.make(config_path)
    
    # Enable interactive mode for live updates
    plt.ion()
    
    # Set axis limits to show the full world
    env.render()
    fig = plt.gcf()
    if fig:
        ax = fig.gca()
        if ax:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            plt.draw()
    
    # Initialize odometry
    robot = env.robot_list[0]
    initial_pose = robot.state.flatten()[:3]
    odom = DifferentialDriveOdometry()
    odom.reset(initial_pose)
    
    # Simple controller
    from scripts.go_to_goal_controller import GoToGoalController
    controller = GoToGoalController()
    
    dt = 0.1  # match simulation timestep
    
    print("Starting odometry test...")
    
    for step in range(500):
        # Get true state
        true_pose = robot.state.flatten()[:3]
        goal_pose = robot.goal.flatten()[:3]
        
        # Use odometry estimate for control (realistic scenario) with obstacle avoidance
        obstacles = env.obstacle_list
        action = controller.compute(odom.pose, goal_pose, obstacles)
        
        # Update odometry
        odom.update(action[0], action[1], dt, true_pose)
        
        # Step simulation
        env.step([action])
        env.render()
        
        # Maintain axis limits
        try:
            fig = plt.gcf()
            if fig:
                ax = fig.gca()
                if ax:
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
        except:
            pass
        
        # Force matplotlib to update
        plt.pause(0.01)
        
        # Print error periodically
        if step % 50 == 0:
            pos_err, ori_err = odom.get_error()
            print(f"Step {step}: Position error = {pos_err:.3f}m, Orientation error = {np.degrees(ori_err):.1f}°")
        
        if env.done():
            break
    
    env.end()
    print("Simulation complete. Generating odometry comparison plot...")
    odom.plot_comparison()

if __name__ == "__main__":
    main()

