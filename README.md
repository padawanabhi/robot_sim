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
├── configs/                    # World configuration files
│   ├── simple_world.yaml            # Basic test world
│   ├── rl_world.yaml               # RL training world
│   └── rl_world_complex.yaml       # Complex static world for RL
├── scripts/                    # Python scripts
│   ├── 00_test_rl_env.py           # Quick RL environment test
│   ├── 01_test_simulation.py       # Basic simulation test
│   ├── 02_go_to_goal.py            # Go-to-goal controller demo
│   ├── 03_potential_field.py       # Potential field navigation demo
│   ├── 04_odometry.py              # Odometry with drift visualization
│   ├── 05_rl_environment.py        # Gymnasium RL environment wrapper
│   ├── 06_train_rl.py              # RL training script (PPO/SAC/TD3)
│   ├── 07_evaluate_rl.py           # RL evaluation script
│   ├── 08_visualize_rl.py          # RL trajectory visualization
│   ├── 09_compare_controllers.py   # Compare RL vs classical controllers
│   ├── 10_train_all_algorithms.py  # Train all RL algorithms in sequence
│   ├── 11_evaluate_all_algorithms.py # Evaluate all trained algorithms
│   ├── 12_improved_training.py     # Enhanced PPO training
│   ├── 13_debug_rl.py              # Debug RL environment compatibility
│   ├── 14_diagnose_rl.py           # Detailed episode-by-episode diagnosis
│   ├── 15_test_obstacle_detection.py # Test obstacle detection accuracy
│   ├── 16_deep_diagnosis.py        # Comprehensive system diagnosis
│   ├── go_to_goal_controller.py    # Reusable go-to-goal controller
│   └── potential_field_controller.py # Reusable potential field controller
├── models/                     # Trained RL models (gitignored)
│   ├── ppo/                    # PPO models
│   │   ├── best_model.zip      # Best performing model
│   │   ├── ppo_nav_final.zip  # Final trained model
│   │   └── checkpoints/        # Training checkpoints
│   ├── sac/                    # SAC models
│   │   ├── best_model.zip
│   │   └── sac_nav_final.zip
│   └── td3/                    # TD3 models
│       ├── best_model.zip
│       └── td3_nav_final.zip
├── logs/                       # Training logs and outputs (gitignored)
│   ├── tensorboard/            # TensorBoard training logs
│   │   ├── PPO/                # PPO training runs
│   │   ├── SAC/                # SAC training runs
│   │   └── TD3/                # TD3 training runs
│   ├── visualizations/         # Episode trajectory PNGs
│   ├── comparisons/            # Controller comparison plots
│   ├── diagnostics/            # Diagnostic reports and plots
│   ├── evaluations.npz         # Evaluation metrics
│   └── odometry_comparison.png # Odometry visualization
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
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

## Scripts Overview

### Core Simulation Scripts (00-04)

**`00_test_rl_env.py`** - Quick RL environment test
- Tests Gymnasium environment compatibility
- Verifies observation/action spaces
- Validates environment reset and step functions

**`01_test_simulation.py`** - Basic simulation test
- Tests IR-SIM basic functionality
- Visualizes robot and obstacles
- Verifies simulation setup

**`02_go_to_goal.py`** - Go-to-goal controller demo
- Demonstrates proportional controller with obstacle avoidance
- Uses `go_to_goal_controller.py` module
- Real-time visualization

**`03_potential_field.py`** - Potential field navigation demo
- Demonstrates artificial potential field controller
- Uses `potential_field_controller.py` module
- Real-time visualization

**`04_odometry.py`** - Odometry with drift visualization
- Tests differential drive odometry
- Simulates sensor noise and drift
- Generates `logs/odometry_comparison.png`

### RL Environment & Training (05-08)

**`05_rl_environment.py`** - Gymnasium RL environment wrapper
- Custom `DiffDriveNavEnv` class
- Randomized obstacles and start/goal positions
- Progress-based reward function
- Action space: linear [0,1], angular [-1,1]

**`06_train_rl.py`** - RL training script
- Supports PPO, SAC, TD3 algorithms
- GPU/MPS acceleration (automatic detection)
- Checkpoints every 20,000 steps
- Saves to `models/{algorithm}/`
- Usage: `python scripts/06_train_rl.py --algorithm ppo --timesteps 500000`

**`07_evaluate_rl.py`** - RL evaluation script
- Evaluates trained models
- Reports success rate, average reward, episode length
- Headless mode recommended
- Usage: `python scripts/07_evaluate_rl.py --algorithm ppo --episodes 5`

**`08_visualize_rl.py`** - RL trajectory visualization
- Generates trajectory plots for episodes
- Saves PNGs to `logs/visualizations/`
- Shows robot path, obstacles, start/goal

### Comparison & Utility Scripts (09-12)

**`09_compare_controllers.py`** - Compare RL vs classical controllers
- Side-by-side comparison of RL, go-to-goal, potential field
- Generates comparison plots in `logs/comparisons/`
- Same scenarios for fair comparison

**`10_train_all_algorithms.py`** - Train all RL algorithms
- Trains PPO, SAC, TD3 in sequence
- Usage: `python scripts/10_train_all_algorithms.py --timesteps 500000`

**`11_evaluate_all_algorithms.py`** - Evaluate all algorithms
- Evaluates all trained models
- Compares performance across algorithms
- Generates summary report

**`12_improved_training.py`** - Enhanced PPO training
- Alternative training script with improved reward function
- Same interface as `06_train_rl.py`

### Diagnostic Scripts (13-16)

**`13_debug_rl.py`** - Debug RL environment
- Checks model-environment compatibility
- Verifies observation/action space matching
- Tests action application

**`14_diagnose_rl.py`** - Detailed episode diagnosis
- Episode-by-episode behavior analysis
- Tracks distance, rewards, actions, obstacle distances
- Generates diagnostic plots in `logs/diagnostics/`

**`15_test_obstacle_detection.py`** - Test obstacle detection
- Verifies obstacle creation and detection
- Tests distance calculations
- Identifies shape detection issues

**`16_deep_diagnosis.py`** - Comprehensive system diagnosis
- Systematic root cause analysis
- Checks model compatibility, action application, reward function
- Generates detailed report in `logs/diagnostics/deep_diagnosis_report.txt`
- Creates visualization: `logs/diagnostics/deep_diagnosis_visualization.png`

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

This generates a comparison plot in `logs/odometry_comparison.png`.

### Reinforcement Learning

#### Quick Environment Test

Test the RL environment before training:

```bash
python scripts/00_test_rl_env.py
```

#### Training

Train a PPO agent for navigation:

```bash
# Train single algorithm
python scripts/06_train_rl.py --algorithm ppo --timesteps 500000

# Train all algorithms (PPO, SAC, TD3)
python scripts/10_train_all_algorithms.py --timesteps 500000
```

**Training Details:**
- Default: 500,000 timesteps (recommended for good performance)
- Checkpoints saved every 20,000 steps to `models/{algorithm}/checkpoints/`
- Best model automatically saved to `models/{algorithm}/best_model.zip`
- Final model saved to `models/{algorithm}/{algorithm}_nav_final.zip`
- Runs headless (no visualization) to prevent crashes
- Uses GPU/MPS acceleration if available

**Monitor Training:**

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

#### Evaluation

Evaluate trained models:

```bash
# Evaluate single algorithm (headless recommended)
python scripts/07_evaluate_rl.py --algorithm ppo --episodes 5 --headless

# Evaluate all algorithms
python scripts/11_evaluate_all_algorithms.py
```

#### Visualization

Generate trajectory visualizations:

```bash
python scripts/08_visualize_rl.py
```

Saves PNG files to `logs/visualizations/` showing:
- Robot trajectory path
- Obstacle locations
- Goal and start positions
- Success/failure status

#### Comparison

Compare RL agents with classical controllers:

```bash
python scripts/09_compare_controllers.py
```

Generates comparison plots in `logs/comparisons/` showing side-by-side performance.

### Diagnostic Tools

#### Debug Environment

Check model-environment compatibility:

```bash
python scripts/13_debug_rl.py
```

#### Diagnose Episodes

Detailed episode-by-episode analysis:

```bash
python scripts/14_diagnose_rl.py
```

Generates diagnostic plots in `logs/diagnostics/`.

#### Test Obstacle Detection

Verify obstacle detection accuracy:

```bash
python scripts/15_test_obstacle_detection.py
```

#### Deep System Diagnosis

Comprehensive root cause analysis:

```bash
python scripts/16_deep_diagnosis.py
```

Generates:
- Text report: `logs/diagnostics/deep_diagnosis_report.txt`
- Visualization: `logs/diagnostics/deep_diagnosis_visualization.png`

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

## File Locations

### Models
Trained RL models are saved in `models/` directory (gitignored):
- **PPO**: `models/ppo/best_model.zip`, `models/ppo/ppo_nav_final.zip`
- **SAC**: `models/sac/best_model.zip`, `models/sac/sac_nav_final.zip`
- **TD3**: `models/td3/best_model.zip`, `models/td3/td3_nav_final.zip`
- **Checkpoints**: `models/{algorithm}/checkpoints/` (saved every 20k steps)

### Logs
All logs and outputs are saved in `logs/` directory (gitignored):
- **TensorBoard**: `logs/tensorboard/{ALGORITHM}/` - Training metrics
- **Visualizations**: `logs/visualizations/` - Episode trajectory PNGs
- **Comparisons**: `logs/comparisons/` - Controller comparison plots
- **Diagnostics**: `logs/diagnostics/` - Diagnostic reports and plots
- **Evaluations**: `logs/evaluations.npz` - Evaluation metrics
- **Odometry**: `logs/odometry_comparison.png` - Odometry visualization

## Results

### Current Performance (After Research-Based Fix)
- **Success Rate:** 5/5 (100%) ✅
- **Average Reward:** 378.35 ± 8.67
- **Average Episode Length:** 202.4 ± 24.2 steps
- **Average Progress:** 9.21 ± 0.60 meters per episode
- **Model:** `models/ppo/best_model.zip` or `models/ppo/ppo_nav_final.zip`

### Training Configuration
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Timesteps:** 500,000 (recommended)
- **Action Space:** Linear [0,1], Angular [-1,1]
- **Reward Function:** Progress-based (10.0 × progress)
- **Device:** GPU/MPS acceleration (automatic detection)

### Key Achievements
1. ✅ 100% success rate in evaluation
2. ✅ No spiral behavior (excessive rotation penalty)
3. ✅ Forward-only movement (action space constraint)
4. ✅ Clear progress-based reward signal
5. ✅ Proper obstacle avoidance learned

## License

See LICENSE file for details.
