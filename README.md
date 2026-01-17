# Reinforcement Learning Experiments

This repository contains reinforcement learning implementations for two classic environments: Taxi-v3 and Atari Breakout.

## Environments

### Taxi-v3

Q-learning implementation with multiple hyperparameter configurations:

- Optimal Baseline
- High LR + Fast Decay
- Low LR + Slow Decay
- Low Discount Factor

### Atari Breakout

Deep Q-Network (DQN) implementation based on Mnih et al. (2015):

- Baseline
- Fast Learner
- Deep Explorer

**Note:** Training Breakout requires significant computational resources. HPC (High-Performance Computing) with GPU is strongly recommended for reasonable training times (~10M timesteps).

## Installation

1. Create and activate virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For GPU training (Breakout), install PyTorch with CUDA:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Training

**Taxi (CPU, ~10 minutes):**

```bash
cd taxi
python taxi_training.py
```

**Breakout (GPU recommended, ~24-48 hours on single GPU):**

```bash
cd breakout
python dqn_training.py
```

### Generate Analysis Plots (Breakout only)

```bash
cd breakout
python generate_plots.py
```

Generates 20 plots analyzing training performance, saved in `saved_agents/plots/`.

## Project Structure

```
├── taxi/
│   ├── configs.py          # Hyperparameter configurations
│   ├── params.py           # Training parameters
│   ├── taxi_training.py    # Q-learning training script
│   └── saved_agents/       # Trained models and data
│
├── breakout/
│   ├── configs.py          # Hyperparameter configurations
│   ├── params.py           # Training parameters
│   ├── dqn_training.py     # DQN training script (HPC compatible)
│   ├── generate_plots.py   # Analysis visualization
│   └── saved_agents/       # Trained models and data
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Features

**Taxi:**

- Q-learning with epsilon-greedy exploration
- Configurable hyperparameters
- Episode-based training
- Analysis plots

**Breakout:**

- Deep Q-Network with CNN
- Experience replay
- Target network
- Mixed precision training
- Checkpoint/resume capability
- Analysis plots

## Requirements

See `requirements.txt` for full list.
