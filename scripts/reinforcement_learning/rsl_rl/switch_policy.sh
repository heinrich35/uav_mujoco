#!/bin/bash

# Script to switch policies at runtime for play_with_policy_switch.py
# Usage: ./switch_policy.sh <path_to_policy_checkpoint>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_policy_checkpoint>"
    echo "Example: $0 /home/heinz/IsaacLab/logs/rsl_rl/g1_standing_revised/2026-01-06_16-20-28/model_700.pt"
    exit 1
fi

POLICY_PATH="$1"
SWITCH_FILE="/tmp/isaac_policy_switch"

# Check if policy file exists
if [ ! -f "$POLICY_PATH" ]; then
    echo "Error: Policy file not found: $POLICY_PATH"
    exit 1
fi

# Convert to absolute path
POLICY_PATH=$(readlink -f "$POLICY_PATH" 2>/dev/null || realpath "$POLICY_PATH" 2>/dev/null || echo "$POLICY_PATH")

# Write the policy path to the switch file
echo "$POLICY_PATH" > "$SWITCH_FILE"

if [ $? -eq 0 ]; then
    echo "[INFO] Policy switch request sent: $POLICY_PATH"
else
    echo "[ERROR] Failed to write to switch file: $SWITCH_FILE"
    exit 1
fi

