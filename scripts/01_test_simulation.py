#!/usr/bin/env python3
"""Test basic IR-SIM simulation."""
import sys
import os

# Set matplotlib backend via environment variable BEFORE any imports
# This avoids conflicts with IR-SIM's internal matplotlib usage
os.environ['MPLBACKEND'] = 'TkAgg'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irsim
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    # Initialize environment from config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'simple_world.yaml')
    env = irsim.make(config_path)

    # Enable interactive mode for live updates
    plt.ion()

    # Set axis limits to show the full world (0 to 10 for both axes)
    # Do this after first render to ensure IR-SIM's figure is created
    env.render()
    fig = plt.gcf()
    if fig:
        ax = fig.gca()
        if ax:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            plt.draw()

    robot = env.robot_list[0]
    print("Starting simulation...")
    print("Close the window to exit.")
    print(f"Robot initial position: {robot.state.flatten()[:3]}")
    print(f"Robot goal position: {robot.goal.flatten()[:3]}")
    
    for step in range(500):
        # Generate velocity command [linear, angular]
        if step < 100:
            v = 0.5  # Forward velocity
            w = 0.0  # No rotation
        elif step < 200:
            v = 0.5
            w = 0.3  # Turn left
        else:
            v = np.random.uniform(0.3, 0.8)
            w = np.random.uniform(-0.5, 0.5)
        
        # Action format: [v, w] for each robot
        action = [float(v), float(w)]
        
        # Step simulation
        env.step([action])

        # Render - IR-SIM handles everything internally
        env.render()

        # Maintain axis limits to show full world (prevent auto-zoom)
        try:
            fig = plt.gcf()
            if fig:
                ax = fig.gca()
                if ax:
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
        except:
            pass

        # Force matplotlib to update the display
        plt.pause(0.01)  # Small pause to allow GUI updates

        # Print status periodically
        if step % 50 == 0:
            current_pose = robot.state.flatten()[:3]
            print(f"Step {step}: Robot at ({current_pose[0]:.2f}, {current_pose[1]:.2f}), "
                  f"theta={np.degrees(current_pose[2]):.1f}°, action=[{v:.2f}, {w:.2f}]")
        
        # Check if done
        if env.done():
            print(f"Goal reached at step {step}!")
            break
    
    env.end()
    print("Simulation complete.")

if __name__ == "__main__":
    main()
