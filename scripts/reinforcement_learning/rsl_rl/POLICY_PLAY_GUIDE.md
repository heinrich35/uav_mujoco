# Policy Play Scripts Guide

This guide explains how to use the policy play scripts with keyboard control and policy switching capabilities.

## Scripts Overview

1. **play_with_keyboard.py** - Run a policy with keyboard input to stop (press 'X')
2. **play_with_policy_switch.py** - Run a policy with the ability to switch between policies at runtime
3. **switch_policy.sh** - Helper script to switch policies while play_with_policy_switch.py is running

## 1. Running a Policy with Keyboard Control

### Basic Usage

Run a policy and stop it by pressing 'X':

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_keyboard.py \
  --task Isaac-G1-Standing-Revised-Direct-v0 \
  --num_envs 1 \
  --checkpoint /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-06_16-20-28/model_700.pt
```

### Arguments

- `--task`: The task name (e.g., `Isaac-G1-Standing-Revised-Direct-v0`)
- `--num_envs`: Number of environments (typically 1 for inference)
- `--checkpoint`: Path to the policy checkpoint file (.pt)
- `--real-time`: (Optional) Run in real-time mode
- `--video`: (Optional) Record video during playback
- `--video_length`: (Optional) Length of video to record (default: 200 steps)

### Keyboard Controls

- **X**: Stop the simulation and exit

**Note**: Make sure the Isaac Sim window has focus for keyboard input to work properly.

## 2. Running with Policy Switching

### Starting the Play Script

Run the play script with policy switching enabled:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_policy_switch.py \
  --task Isaac-G1-Standing-Revised-Direct-v0 \
  --num_envs 1 \
  --checkpoint /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-06_16-20-28/model_700.pt
```

### Switching Policies

While the script is running, open a **new terminal** and use the switch script:

```bash
./scripts/reinforcement_learning/rsl_rl/switch_policy.sh \
  /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-05_23-23-15/final_model.pt
```

The policy will be switched automatically without stopping the simulation.

### Keyboard Controls

- **X**: Stop the simulation and exit

## 3. Examples

### Example 1: Run a single policy and stop with 'X'

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_keyboard.py \
  --task Isaac-G1-Standing-Revised-Direct-v0 \
  --num_envs 1 \
  --checkpoint /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-06_16-20-28/model_700.pt
```

### Example 2: Run with policy switching

**Terminal 1** (start the play script):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_policy_switch.py \
  --task Isaac-G1-Standing-Revised-Direct-v0 \
  --num_envs 1 \
  --checkpoint /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-06_16-20-28/model_700.pt
```

**Terminal 2** (switch to a different policy):
```bash
./scripts/reinforcement_learning/rsl_rl/switch_policy.sh \
  /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-05_23-23-15/final_model.pt
```

**Terminal 2** (switch to yet another policy):
```bash
./scripts/reinforcement_learning/rsl_rl/switch_policy.sh \
  /home/heinz/IsaacLab/logs/rsl_rl/improved_g1_standing/2026-01-05_18-04-33/model_6300.pt
```

## 4. Troubleshooting

### Keyboard input not working

- Make sure the Isaac Sim window has focus (click on it)
- The keyboard listener only works when the window is active

### Policy switch not working

- Ensure `play_with_policy_switch.py` is running
- Check that the policy file path is correct and the file exists
- Verify the policy checkpoint is compatible with the task

### File not found errors

- Use absolute paths for checkpoints
- Ensure the checkpoint file exists before running
- Check file permissions

## 5. Technical Details

### How Policy Switching Works

The `play_with_policy_switch.py` script:
1. Watches a file (`/tmp/isaac_policy_switch`) for policy switch requests
2. When a new policy path is written to this file, it loads the new policy
3. The switch happens between simulation steps, so it's seamless

The `switch_policy.sh` script:
1. Validates the policy file exists
2. Converts the path to absolute
3. Writes the path to `/tmp/isaac_policy_switch`

### Thread Safety

Policy switching uses a lock to ensure thread-safe policy loading and switching.

## 6. Comparison with Original play.py

| Feature | play.py | play_with_keyboard.py | play_with_policy_switch.py |
|---------|---------|----------------------|---------------------------|
| Keyboard stop | No | Yes (X key) | Yes (X key) |
| Policy switching | No | No | Yes |
| Video recording | Yes | Yes | Yes |
| Real-time mode | Yes | Yes | Yes |

