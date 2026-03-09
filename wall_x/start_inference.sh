#!/bin/bash
# Start Wall-X inference node (run in a separate terminal).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CHUNK_RANGE="first"
[[ "${USE_LAST_ACTIONS}" =~ ^(true|1)$ ]] && CHUNK_RANGE="last"

echo "=========================================="
echo "  Wall-X inference node"
echo "=========================================="
echo "  Policy server:  $POLICY_SERVER_HOST:$POLICY_SERVER_PORT"
echo "  Task:           $TASK"
echo "  Dataset:        $DATASET_NAME"
echo "  Chunk:          ${MAX_ACTIONS_TO_PUBLISH} actions ($CHUNK_RANGE N)"
echo "  Infer freq:     $INFERENCE_FREQUENCY Hz"
echo "  State:          robot (no chunk blend)"
echo "  Smoothing:      alpha=${ACTION_SMOOTHING_ALPHA:-0.45}"
echo "=========================================="

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using venv: $VENV_PATH"
else
    echo "Venv not found: $VENV_PATH (using current env)"
fi
source /opt/ros/humble/setup.bash
cd "$SCRIPT_DIR"

USE_LAST_FLAG="--use-last-actions"
[[ "${USE_LAST_ACTIONS}" =~ ^(false|0)$ ]] && USE_LAST_FLAG="--no-use-last-actions"

REVERSE_FLAG=""
[[ "${REVERSE_ACTIONS}" =~ ^(true|1)$ ]] && REVERSE_FLAG="--reverse-actions"

USE_LAST_ACTION_FLAG="--use-last-action-as-state"
[[ "${USE_LAST_ACTION_AS_STATE}" =~ ^(false|0)$ ]] && USE_LAST_ACTION_FLAG="--no-use-last-action-as-state"

SAVE_PLOTS_FLAG="--save-action-plots"
[[ "${SAVE_ACTION_PLOTS}" =~ ^(false|0)$ ]] && SAVE_PLOTS_FLAG="--no-save-action-plots"

SAVE_CONCAT_FLAG="--save-concatenated-plot"
[[ "${SAVE_CONCATENATED_PLOT}" =~ ^(false|0)$ ]] && SAVE_CONCAT_FLAG="--no-save-concatenated-plot"

python wall_inference_node.py \
    --policy-server-host="$POLICY_SERVER_HOST" \
    --policy-server-port="$POLICY_SERVER_PORT" \
    --cam-high-serial="$CAM_HIGH_SERIAL" \
    --cam-wrist-serial="$CAM_WRIST_SERIAL" \
    --cam-side-serial="$CAM_SIDE_SERIAL" \
    --task="$TASK" \
    --dataset-name="$DATASET_NAME" \
    --inference-frequency="${INFERENCE_FREQUENCY:-1.0}" \
    --max-actions-to-publish="${MAX_ACTIONS_TO_PUBLISH:-32}" \
    --delay-steps-manual="${DELAY_STEPS_MANUAL:-0}" \
    --action-smoothing-alpha="${ACTION_SMOOTHING_ALPHA:-0.45}" \
    --action-publish-interval="${ACTION_PUBLISH_INTERVAL:-0.01}" \
    $USE_LAST_FLAG \
    $USE_LAST_ACTION_FLAG \
    $SAVE_PLOTS_FLAG \
    $SAVE_CONCAT_FLAG \
    $REVERSE_FLAG
