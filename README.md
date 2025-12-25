# Robot Simulation - IR-SIM Project

A complete robot simulation environment for differential drive robot navigation, localization, and reinforcement learning using IR-SIM (2D simulation).

## Project Status

✅ **Fully Functional** - All components working:
- Basic simulation with visualization
- Navigation controllers (go-to-goal, potential field)
- Odometry with drift visualization
- RL environment with randomized obstacles
- PPO training (100k timesteps)
- Model evaluation and visualization

**Current Performance:**
- Training: Average reward ~7.81, episode length ~261 steps
- Evaluation: ~20% success rate (can improve with more training)

## Project Structure

```
robot_sim/
├── configs/              # World configuration files
│   ├── simple_world.yaml      # Basic test world
│   ├── rl_world.yaml         # RL training world
│   └── rl_world_complex.yaml # Complex static world for RL
├── scripts/              # Python scripts
│   ├── 00_test_rl_env.py          # Quick RL environment test
│   ├── 01_test_simulation.py      # Basic simulation test
│   ├── 02_go_to_goal.py           # Go-to-goal controller
│   ├── 03_potential_field.py      # Potential field navigation
│   ├── 04_odometry.py             # Odometry with drift visualization
│   ├── 05_rl_environment.py       # Gymnasium RL environment wrapper
│   ├── 06_train_rl.py             # RL training script
│   ├── 07_evaluate_rl.py          # RL evaluation script
│   ├── 08_visualize_rl.py         # RL trajectory visualization
│   └── go_to_goal_controller.py   # Reusable go-to-goal controller
├── models/               # Trained RL models (gitignored)
│   ├── ppo_nav_final.zip
│   ├── best_model.zip
│   └── checkpoints/
├── logs/                 # Training logs and visualizations (gitignored)
│   ├── tensorboard/      # TensorBoard logs
│   ├── visualizations/   # Episode trajectory PNGs
│   └── odometry_comparison.png
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md
```

## Setup Instructions

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import irsim; print('IR-SIM version:', irsim.__version__)"
```

## Usage

### Basic Simulation Test

Test the basic simulation environment:

```bash
python scripts/01_test_simulation.py
```

**Note:** On macOS, you may see warnings about accessibility. This is normal and can be ignored.

### Navigation Controllers

#### Go-to-Goal Controller

Simple proportional controller with obstacle avoidance:

```bash
python scripts/02_go_to_goal.py
```

#### Potential Field Navigation

Artificial potential field controller with obstacle avoidance:

```bash
python scripts/03_potential_field.py
```

### Odometry and Localization

Test differential drive odometry with noise simulation:

```bash
python scripts/04_odometry.py
```

This will generate a comparison plot in `logs/odometry_comparison.png`.

### Reinforcement Learning

#### Quick Environment Test

Test the RL environment before training:

```bash
python scripts/00_test_rl_env.py
```

#### Training

Train a PPO agent for navigation:

```bash
python scripts/06_train_rl.py
```

**Training Details:**
- Default: 100,000 timesteps (~15-30 minutes)
- Checkpoints saved every 10,000 steps
- Best model automatically saved
- Runs headless (no visualization) to prevent crashes

**Monitor Training:**

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

#### Evaluation

Evaluate a trained model (headless mode recommended):

```bash
# Headless mode (recommended, no crashes)
python scripts/07_evaluate_rl.py --headless

# With visualization (may cause issues on macOS)
python scripts/07_evaluate_rl.py
```

#### Visualization

Generate trajectory visualizations for trained episodes:

```bash
python scripts/08_visualize_rl.py
```

This saves PNG files to `logs/visualizations/` showing:
- Robot trajectory path
- Obstacle locations
- Goal and start positions
- Success/failure status

## Quick Start Pipeline

Run the complete pipeline in order:

```bash
# Activate virtual environment
source venv/bin/activate

# Step 1: Test basic simulation
python scripts/01_test_simulation.py

# Step 2: Test go-to-goal controller
python scripts/02_go_to_goal.py

# Step 3: Test potential field navigation
python scripts/03_potential_field.py

# Step 4: Test odometry
python scripts/04_odometry.py

# Step 5: Test RL environment
python scripts/00_test_rl_env.py

# Step 6: Train RL agent (may take 15-30 minutes)
python scripts/06_train_rl.py

# Step 7: Evaluate trained agent (headless)
python scripts/07_evaluate_rl.py --headless

# Step 8: Visualize episodes
python scripts/08_visualize_rl.py

# Monitor training (in another terminal)
tensorboard --logdir logs/tensorboard
```

## Configuration

### World Configuration

Edit world YAML files in `configs/` to customize:
- World dimensions
- Robot starting position and goal
- Obstacle positions and shapes
- Simulation timestep

**Available Configs:**
- `simple_world.yaml`: Basic test world with 2 obstacles
- `rl_world.yaml`: Training world with 4 obstacles
- `rl_world_complex.yaml`: Complex world with 9 obstacles

### RL Training Parameters

Edit `scripts/06_train_rl.py` to adjust:
- `total_timesteps`: Training duration (default: 100,000)
- `learning_rate`: PPO learning rate (default: 3e-4)
- `n_steps`: Steps per update (default: 2048)
- `batch_size`: Batch size (default: 64)

### RL Environment

Edit `scripts/05_rl_environment.py` to customize:
- Observation space
- Reward function
- Randomization settings
- Max episode length

## Tips and Best Practices

### macOS-Specific Tips

1. **Visualization Issues:**
   - IR-SIM uses Matplotlib which can cause crashes on macOS
   - Use headless mode for training: `render_mode=None`
   - For evaluation, use `--headless` flag
   - Visualization script saves PNGs instead of live display

2. **Accessibility Warnings:**
   - "This process is not trusted" warnings are normal
   - Can be safely ignored
   - Related to macOS security, not a bug

3. **Matplotlib Backend:**
   - Training uses 'Agg' backend (non-interactive)
   - Evaluation can use 'TkAgg' but may hang
   - Visualization script uses 'Agg' to save PNGs

### Training Tips

1. **Improving Performance:**
   - Train longer: Increase `total_timesteps` to 200k-500k
   - Use curriculum learning: Start with fewer obstacles
   - Tune reward function: Adjust weights in `_compute_reward()`
   - Increase randomization: More varied obstacle configurations

2. **Monitoring:**
   - Use TensorBoard to track training progress
   - Check episode rewards and lengths
   - Monitor success rate during evaluation

3. **Model Management:**
   - Best model saved automatically during training
   - Checkpoints saved every 10,000 steps
   - Final model saved at end of training

### Evaluation Tips

1. **Headless Mode:**
   - Always use `--headless` for evaluation to avoid crashes
   - Faster execution without visualization overhead
   - Results printed to terminal

2. **Visualization:**
   - Use `08_visualize_rl.py` to see trajectory plots
   - PNGs saved to `logs/visualizations/`
   - Review failure cases to improve training

## Troubleshooting

### Import Errors

If you encounter import errors:
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Install all dependencies: `pip install -r requirements.txt`
3. Run scripts from project root directory

### Python Crashes on macOS

**Symptoms:** Python crashes when running scripts with visualization

**Solutions:**
1. Use headless mode for training (already implemented)
2. Use `--headless` flag for evaluation
3. Visualization script saves PNGs instead of displaying
4. Set `MPLBACKEND=Agg` environment variable if needed

### Visualization Not Updating

**Symptoms:** Plot shows but robot doesn't move visually

**Solutions:**
1. Ensure `plt.ion()` is called (interactive mode)
2. Use `plt.pause(0.01)` after `env.render()`
3. Set axis limits explicitly: `ax.set_xlim(0, 10)`, `ax.set_ylim(0, 10)`
4. Check that `MPLBACKEND` is set before imports

### Training Performance Issues

**Low Success Rate:**
- Train for more timesteps (200k-500k)
- Adjust reward function weights
- Reduce obstacle density
- Increase training episodes

**Slow Training:**
- Reduce `n_steps` in PPO config
- Use smaller batch size
- Disable progress bar (already done)

### Display Issues (Linux)

For headless servers:

```bash
export DISPLAY=:0
# Or use virtual display
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```

### Memory Issues

If training runs out of memory:
- Reduce `n_steps` in PPO configuration
- Reduce `batch_size`
- Use fewer parallel environments (if using vectorized envs)

## Dependencies

- `ir-sim`: 2D robot simulation
- `gymnasium`: RL environment interface
- `stable-baselines3[extra]`: RL algorithms (PPO)
- `numpy`: Numerical computing
- `matplotlib`: Plotting
- `tensorboard`: Training visualization
- `pyyaml`: YAML configuration parsing
- `tqdm`, `rich`: Progress bars (optional)

## Results

### Training Results
- **Timesteps:** 100,000
- **Average Reward:** ~7.81
- **Average Episode Length:** ~261 steps
- **Model Saved:** `models/ppo_nav_final.zip`

### Evaluation Results
- **Success Rate:** ~20% (varies with randomization)
- **Average Reward:** Varies by episode difficulty
- **Visualizations:** Saved to `logs/visualizations/`

### Next Steps for Improvement
1. Train for longer (200k-500k timesteps)
2. Tune reward function hyperparameters
3. Implement curriculum learning
4. Add more diverse obstacle configurations
5. Experiment with different RL algorithms (SAC, TD3)

## License

See LICENSE file for details.
