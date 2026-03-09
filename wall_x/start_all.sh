#!/bin/bash
# Wall-X Franka deploy - quick start guide. Run control and inference in separate terminals.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo ""
echo "=========================================="
echo "  Wall-X Franka deploy - quick start"
echo "=========================================="
echo ""
echo "Current config:"
echo "  Robot IP:        $ROBOT_IP"
echo "  Policy server:   $POLICY_SERVER_HOST:$POLICY_SERVER_PORT"
echo "  Task:            $TASK"
echo "  Dataset:         $DATASET_NAME"
echo ""

echo "=========================================="
echo "  Step 1: Cameras (optional)"
echo "=========================================="
echo "Inference node uses pyrealsense2; no separate ROS2 camera launch needed."
echo "If you use ROS2 camera nodes, start them first."
echo ""

echo "=========================================="
echo "  Step 2: Start control and inference"
echo "=========================================="
echo ""
echo "Terminal 1 - control node:"
echo "  cd $SCRIPT_DIR"
echo "  bash start_control.sh"
echo ""
echo "Terminal 2 - inference node:"
echo "  cd $SCRIPT_DIR"
echo "  bash start_inference.sh"
echo ""
echo "Stop: Ctrl+C in each terminal."
echo ""
