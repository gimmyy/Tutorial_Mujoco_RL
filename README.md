# MuJoCo + Reinforcement Learning Tutorial for Beginners
A practical, beginner-friendly tutorial for getting started with MuJoCo physics simulation and Reinforcement Learning (RL). The code is arranged in order of increasing difficulty, fully runnable, and takes you step by step from basic environment setup to robot arm practical applications.

## Who This Tutorial Is For
- Beginners who want to learn MuJoCo physics simulation
- RL newcomers looking for simple, runnable, and easy-to-understand code
- Students or enthusiasts in robotics, robot arms, or simulation fields
- Anyone who wants to get started quickly without being discouraged by complex environment configuration

## Project Structure
The code is organized from easy to difficult, allowing you to learn progressively. Master one file before moving to the next:

```
├── 1tutorial_ur5e                    # Import robot XML description file, simple Python control & MuJoCo simulation demo

├── 2tutorial_InvertedPendulum        # RL Introduction: Inverted pendulum in MuJoCo, customize training model

├── 3tutorial_panda_obstacle          # Panda robot: Obstacle avoidance & path planning with PPO, customize RL environment and reward function

├── 4tutorial_orca_cube_orientation   # ORCA Hand: Cube orientation, increase robot DOF and complexity, customize RL framework

└── README.md                         # This documentation 
```

## Quick Start

### 1. Create Conda Environment

 `conda create -n orca python=3.11`
### 2. Install Dependencies 

Run this command in your terminal to install all required packages:
`pip install mujoco gymnasium stable-baselines3[extra]`
### 3. Run the Simplest Code First

Start with the basic MuJoCo environment to understand core concepts, cd into directory 1tutorial_ur5e and run:
`python main.py`
### 4. Train Your First RL Agent

Move on to training a reinforcement learning agent, cd into directory 2tutorial_InvertedPendulum and run:
`python mujoco_reinforce_disp.py`

## Learning Path Suggestion

1. Run the environment first: Understand what observation, action, and reward are.
2. Train a simple model: Experience the process of RL agents learning from random actions to completing tasks.
3. Customize the reward function: Learn how rewards affect the robot's behavior.
4. Build a custom environment: Try creating your own robot simulation environment.

## Features

- Simple code with detailed comments, no redundant content.
- Based on MuJoCo + Stable-Baselines3 — a commonly used combination in industry and research.
- Gradually transition from toy examples to robot arm/robot practical applications.

## Supplementary Note
All code has been run and tested on Windows 10. If you use other operating systems, please check the system compatibility and modify the code accordingly.
