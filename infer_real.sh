#!/bin/bash
set -euo pipefail

# Real-robot serving entrypoint for Wall-X.
#
# This script starts a websocket server that serves `WallXPolicy`:
#   wall_x/serving/policy/wall_x_policy.py
#
# It is meant to match the real-robot training config:
#   workspace/lerobot_example/franka_real.yml

# ================= Environment Setup =================
CONDA_BASE="${CONDA_BASE:-/mnt/data/lfwj/miniconda3}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

WALLX_ROOT="${WALLX_ROOT:-/mnt/data2/lfwj/wall-x4juliang}"
cd "${WALLX_ROOT}"
export PYTHONPATH="${WALLX_ROOT}:${PYTHONPATH:-}"

# ================= Config (override via env vars) =================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

ENV_MODE="${ENV_MODE:-aloha}"  # {aloha,libero} (only affects default presets/metadata)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-32157}"

CKPT_ROOT="${CKPT_ROOT:-${WALLX_ROOT}/ckpt/franka}"
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "[infer_real.sh] ERROR: CKPT_ROOT does not exist: ${CKPT_ROOT}" >&2
    exit 1
  fi
  latest="$(ls -1 "${CKPT_ROOT}" 2>/dev/null | rg -n \"^[0-9]+$\" | sort -n | tail -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "[infer_real.sh] ERROR: no numeric checkpoints found under: ${CKPT_ROOT}" >&2
    exit 1
  fi
  MODEL_PATH="${CKPT_ROOT}/${latest}"
fi

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-${MODEL_PATH}/config.yml}"
# For this repo's checkpoints, the action tokenizer artifacts (tokenizer.json, preprocessor_config.json, ...)
# are saved alongside model.safetensors, so using the same folder is usually correct.
ACTION_TOKENIZER_PATH="${ACTION_TOKENIZER_PATH:-${MODEL_PATH}}"

PRED_HORIZON="${PRED_HORIZON:-32}"  # franka_real.yml:data.action_horizon
ACTION_DIM="${ACTION_DIM:-7}"       # stack_bowls_rc single-arm: 3+3+1
STATE_DIM="${STATE_DIM:-7}"         # customized_agent_pos_config: panda_state_eef_with_gripper: 7
DTYPE="${DTYPE:-bfloat16}"
PREDICT_MODE="${PREDICT_MODE:-fast}"  # {fast,diffusion}

# Camera keys must match what your *client* sends at runtime.
# franka_real.yml has resolution constraints for these keys.
CAMERA_KEYS=(${CAMERA_KEYS:-"face_view left_wrist_view right_wrist_view"})

echo "[infer_real.sh] CONDA_ENV=${CONDA_ENV:-<already-active>}"
echo "[infer_real.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[infer_real.sh] ENV_MODE=${ENV_MODE}"
echo "[infer_real.sh] HOST=${HOST}"
echo "[infer_real.sh] PORT=${PORT}"
echo "[infer_real.sh] MODEL_PATH=${MODEL_PATH}"
echo "[infer_real.sh] TRAIN_CONFIG_PATH=${TRAIN_CONFIG_PATH}"
echo "[infer_real.sh] ACTION_TOKENIZER_PATH=${ACTION_TOKENIZER_PATH}"
echo "[infer_real.sh] PRED_HORIZON=${PRED_HORIZON} ACTION_DIM=${ACTION_DIM} STATE_DIM=${STATE_DIM}"
echo "[infer_real.sh] DTYPE=${DTYPE} PREDICT_MODE=${PREDICT_MODE}"
echo "[infer_real.sh] CAMERA_KEYS=${CAMERA_KEYS[*]}"

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "[infer_real.sh] ERROR: missing train config: ${TRAIN_CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/model.safetensors" ]]; then
  echo "[infer_real.sh] ERROR: missing model weights: ${MODEL_PATH}/model.safetensors" >&2
  exit 1
fi

# ================= Dependency Preflight =================
python -c 'import websockets; import yaml' >/dev/null
python -c 'import msgpack; import msgpack_numpy' >/dev/null 2>&1 || {
  cat >&2 <<EOF
[infer_real.sh] ERROR: missing msgpack deps in current env.

Serving uses msgpack + msgpack-numpy for sending/receiving numpy arrays over websockets:
  wall_x/serving/websocket_policy_server.py

Install (inside your conda env):
  python -m pip install -U msgpack msgpack-numpy
EOF
  exit 2
}

# ================= Start Serving =================
exec python -m wall_x.serving.launch_serving \
  --env "${ENV_MODE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-config.model-path "${MODEL_PATH}" \
  --model-config.action-tokenizer-path "${ACTION_TOKENIZER_PATH}" \
  --model-config.train-config-path "${TRAIN_CONFIG_PATH}" \
  --model-config.action-dim "${ACTION_DIM}" \
  --model-config.state-dim "${STATE_DIM}" \
  --model-config.pred-horizon "${PRED_HORIZON}" \
  --model-config.device "cuda" \
  --model-config.dtype "${DTYPE}" \
  --model-config.predict-mode "${PREDICT_MODE}" \
  --model-config.camera-key "${CAMERA_KEYS[@]}"
