#!/bin/bash
# =============================================================================
# Wall-X Franka deploy - single config file. Edit here to adjust all scripts.
#
# Before first run: set ROBOT_IP, POLICY_SERVER_HOST, camera serials, and paths.
# =============================================================================

export DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Paths (use defaults or set for your machine)
# -----------------------------------------------------------------------------
export VENV_PATH="${VENV_PATH:-$DEPLOY_DIR/.venv}"
export FRANKY_SOURCE="${FRANKY_SOURCE:-$HOME/franky}"
export LOG_DIR="$DEPLOY_DIR/logs"
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Control node (franka_control_node.py)
# -----------------------------------------------------------------------------
export ROBOT_IP="${ROBOT_IP:-192.168.1.2}"

export DYNAMICS_FACTOR="${DYNAMICS_FACTOR:-0.12}"
export DYNAMICS_VELOCITY="${DYNAMICS_VELOCITY:-0}"
export DYNAMICS_ACCELERATION="${DYNAMICS_ACCELERATION:-0}"
export DYNAMICS_JERK="${DYNAMICS_JERK:-0}"
export WAIT_FOR_MOTION="${WAIT_FOR_MOTION:-false}"
export CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-3.0}"
export MIN_Z="${MIN_Z:-0.03}"
export QUEUE_LOW_THRESHOLD="${QUEUE_LOW_THRESHOLD:-0}"
export EXECUTE_STRIDE="${EXECUTE_STRIDE:-2}"
export SETTLE_DELAY="${SETTLE_DELAY:-0}"
export GRIPPER_SLEEP_AFTER_GRASP="${GRIPPER_SLEEP_AFTER_GRASP:-0.5}"

# -----------------------------------------------------------------------------
# Inference node (wall_inference_node.py) - Policy server
# -----------------------------------------------------------------------------
export POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-localhost}"
export POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8000}"

export TASK="${TASK:-stack the bowls}"
export DATASET_NAME="${DATASET_NAME:-stack_bowls_rc}"

export MAX_ACTIONS_TO_PUBLISH="${MAX_ACTIONS_TO_PUBLISH:-16}"
export USE_LAST_ACTIONS="${USE_LAST_ACTIONS:-false}"
export INFERENCE_FREQUENCY="${INFERENCE_FREQUENCY:-1.0}"
export ACTION_PUBLISH_INTERVAL="${ACTION_PUBLISH_INTERVAL:-0}"

export DELAY_STEPS_MANUAL="${DELAY_STEPS_MANUAL:-0}"
export REVERSE_ACTIONS="${REVERSE_ACTIONS:-false}"
export USE_LAST_ACTION_AS_STATE="${USE_LAST_ACTION_AS_STATE:-false}"
export ACTION_SMOOTHING_ALPHA="${ACTION_SMOOTHING_ALPHA:-0.45}"
export SAVE_ACTION_PLOTS="${SAVE_ACTION_PLOTS:-true}"
export SAVE_CONCATENATED_PLOT="${SAVE_CONCATENATED_PLOT:-true}"

# -----------------------------------------------------------------------------
# Cameras (RealSense serials - replace with your device serials)
# Mapping: face_view=cam_wrist, left_wrist_view=cam_side, right_wrist_view=cam_high
# -----------------------------------------------------------------------------
export CAM_HIGH_SERIAL="${CAM_HIGH_SERIAL:-}"
export CAM_WRIST_SERIAL="${CAM_WRIST_SERIAL:-}"
export CAM_SIDE_SERIAL="${CAM_SIDE_SERIAL:-}"

export CAMERA0_SERIAL="${CAMERA0_SERIAL:-$CAM_WRIST_SERIAL}"
export CAMERA1_SERIAL="${CAMERA1_SERIAL:-$CAM_HIGH_SERIAL}"
export CAMERA2_SERIAL="${CAMERA2_SERIAL:-$CAM_SIDE_SERIAL}"
export ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

# -----------------------------------------------------------------------------
# ROS2 topics (no need to change usually)
# -----------------------------------------------------------------------------
export EE_STATES_TOPIC="${EE_STATES_TOPIC:-/franka/ee_states}"
export ACTION_TOPIC="${ACTION_TOPIC:-/franka/action_command}"
