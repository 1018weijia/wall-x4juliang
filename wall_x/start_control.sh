#!/bin/bash
# Start Franka control node (run in a separate terminal).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=========================================="
echo "  Franka control node"
echo "=========================================="
echo "  Robot IP:        $ROBOT_IP"
echo "  Dynamics factor:  $DYNAMICS_FACTOR"
echo "  Wait for motion:  $WAIT_FOR_MOTION"
echo "  Control freq:     $CONTROL_FREQUENCY Hz"
echo "  Min Z:            ${MIN_Z}m"
echo "  Queue low thresh:  $QUEUE_LOW_THRESHOLD"
echo "  Execute stride:   $EXECUTE_STRIDE"
echo "  Settle delay:     ${SETTLE_DELAY}s"
echo "  Gripper sleep:    ${GRIPPER_SLEEP_AFTER_GRASP}s"
echo "=========================================="

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using venv: $VENV_PATH"
else
    echo "Venv not found: $VENV_PATH (using current env)"
fi
source /opt/ros/humble/setup.bash
cd "$SCRIPT_DIR"

[[ "${WAIT_FOR_MOTION}" =~ ^(true|1|yes)$ ]] && WAIT_FLAG="--wait-for-motion" || WAIT_FLAG="--no-wait-for-motion"

python franka_control_node.py \
    --robot-ip="$ROBOT_IP" \
    --dynamics-factor="${DYNAMICS_FACTOR:-0.05}" \
    --dynamics-velocity="${DYNAMICS_VELOCITY:-0}" \
    --dynamics-acceleration="${DYNAMICS_ACCELERATION:-0}" \
    --dynamics-jerk="${DYNAMICS_JERK:-0}" \
    --control-frequency="${CONTROL_FREQUENCY:-2.0}" \
    $WAIT_FLAG \
    --min-z="${MIN_Z:-0.03}" \
    --queue-low-threshold="${QUEUE_LOW_THRESHOLD:-0}" \
    --execute-stride="${EXECUTE_STRIDE:-1}" \
    --settle-delay="${SETTLE_DELAY:-0.25}" \
    --gripper-sleep-after-grasp="${GRIPPER_SLEEP_AFTER_GRASP:-0.5}"
