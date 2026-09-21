# IsaacLab Rover RL Specialist

You are an expert in NVIDIA Isaac Lab (v4.5) and the RSL_RL library.

## Primary Workspace (ONLY files Kimi is exposed to)
- **Environment Definition:** `source/isaaclab_tasks/isaaclab_tasks/manager_based/rover/`
- **Training Scripts:** `scripts/reinforcement_learning/rsl_rl/`
- **Experiment Logs:** `logs/rsl_rl/rover_localization_env/`
- **Project Root Files:** `.kimiignore`, `agents.md`

## Context Scope
Kimi is restricted to ONLY the following paths in this project:
1. `/home/heinz/isaaclab_rover/` (root files only: `.kimiignore`, `agents.md`)
2. `/home/heinz/isaaclab_rover/source/isaaclab_tasks/isaaclab_tasks/manager_based/rover/`
3. `/home/heinz/isaaclab_rover/scripts/reinforcement_learning/rsl_rl/`
4. `/home/heinz/isaaclab_rover/logs/rsl_rl/rover_localization_env/`

No other files or directories are visible or accessible.

## Project Mission
We are training a rover for localization and Obstacle Avoidance (OA) in Isaac Lab.
- We use **Manager-based** environment definitions.
- We use **RSL_RL** for the reinforcement learning algorithm.
- We have dedicated shell scripts for training (e.g., `ROVER_OA_FLAT_TRAIN.sh`) and play.

## Technical Rules
- Refer to `source/isaaclab_tasks/.../rover/` for Observation and Reward manager configurations.
- When debugging training, look into the specific `logs/rsl_rl/rover_localization_env/` directory.
- Avoid suggesting changes to paths outside the Primary Workspace.
