#!/bin/bash
set -euo pipefail

# ================= Environment Setup =================
# You need ONE conda env that contains BOTH:
# - Wall-X runtime deps (torch/transformers/...)
# - LIBERO sim deps (libero/robosuite/bddl/mujoco/...)
#
# Option A (recommended): export CONDA_ENV=<your_env> then run: bash infer.sh
# Option B: activate your env manually before running this script.
CONDA_BASE="${CONDA_BASE:-/mnt/data/lfwj/miniconda3}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# Point PYTHONPATH to your local Wall-X repo and local LIBERO repo (if not installed via pip).
WALLX_ROOT="${WALLX_ROOT:-/mnt/data2/lfwj/wall-x4juliang}"
LIBERO_REPO="${LIBERO_REPO:-${WALLX_ROOT}/LIBERO}"
export PYTHONPATH="${LIBERO_REPO}:${WALLX_ROOT}:${PYTHONPATH:-}"

# Numba cache: robosuite uses numba with cache=True; if the conda env isn't writable,
# set a writable cache dir to avoid "no locator available" errors.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "${NUMBA_CACHE_DIR}"

# Libero reads ${LIBERO_CONFIG_PATH}/config.yaml; if missing it may prompt for input and hang.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${WALLX_ROOT}/.libero}"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${LIBERO_REPO}/libero/libero
bddl_files: ${LIBERO_REPO}/libero/libero/bddl_files
init_states: ${LIBERO_REPO}/libero/libero/init_files
datasets: ${LIBERO_REPO}/datasets
assets: ${LIBERO_REPO}/libero/libero/assets
EOF
  echo "[infer.sh] Wrote default LIBERO config to: ${LIBERO_CONFIG_PATH}/config.yaml"
fi

# Quick preflight check: fail fast with a readable error instead of empty logs.
python - <<'PY'
import importlib, sys
mods = ["libero", "robosuite", "bddl", "mujoco"]
failed = []
print("[infer.sh] Python:", sys.executable)
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[infer.sh] import {m}: OK")
    except Exception as e:
        print(f"[infer.sh] import {m}: FAIL ({type(e).__name__}: {e})")
        failed.append(m)

# Gym is optional if gymnasium is installed (LIBERO venv can use either).
try:
    importlib.import_module("gym")
    print("[infer.sh] import gym: OK")
except Exception:
    try:
        importlib.import_module("gymnasium")
        print("[infer.sh] import gym: MISSING (gymnasium OK)")
    except Exception as e:
        print(f"[infer.sh] import gym/gymnasium: FAIL ({type(e).__name__}: {e})")
        failed.append("gym")

# bddl depends on `future` (pure python); check explicitly for clearer guidance.
try:
    importlib.import_module("future")
    print("[infer.sh] import future: OK")
except Exception as e:
    print(f"[infer.sh] import future: FAIL ({type(e).__name__}: {e})")
    failed.append("future")
if failed:
    print(
        "\n[infer.sh] Missing sim dependencies in the current conda env. "
        "Please install them into the SAME env you use to run infer.sh.\n"
        "Suggested (in your env):\n"
        "  python -m pip install -e /mnt/data2/lfwj/wall-x4juliang/LIBERO\n"
        "  python -m pip install robosuite==1.4.0 bddl==1.0.1 future mujoco gym\n"
        "\nNote: do NOT pip install -r LIBERO/requirements.txt into the Wall-X env, "
        "it pins old transformers/numpy and may break Wall-X.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY

# ================= Run Config =================
# ================= 配置区域 =================
checkpoint_path="/mnt/data2/lfwj/wall-x4juliang/ckpt/libero/juliang-plan_libero/11"
mode="flow"
norm_key="libero_all"

initial_states_path="DEFAULT"
num_trials_per_task=10
seed=3407
rollout_dir="./rollouts/eval_${mode}_${seed}_$(date +%Y%m%d_%H%M%S)"
cam_names=("face_view" "right_wrist_view")

# 日志保存目录
log_dir="./logs/eval_${mode}_${seed}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# ================= 并行任务定义 =================

# 定义 4 个子任务套件名称
task_suites=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# export MUJOCO_GL="egl"
# export MUJOCO_EGL_DEVICE_ID=0
# export MUJOCO_GL=osmesa
# Define GPU IDs to use.
# If you provide fewer GPU IDs than task_suites, we will reuse them with modulo indexing.
gpu_ids=(4 5)

# ================= 开始执行 =================

echo "开始并行评测..."
echo "日志将保存至: $log_dir"

# 存放 PID 的数组
pids=()

len=${#task_suites[@]}
num_gpus=${#gpu_ids[@]}
if (( num_gpus == 0 )); then
  echo "[infer.sh] ERROR: gpu_ids is empty." >&2
  exit 1
fi
if (( num_gpus < len )); then
  echo "[infer.sh] WARNING: gpu_ids has ${num_gpus} entries but task_suites has ${len}; will reuse GPUs with modulo indexing."
fi
for (( i=0; i<$len; i++ )); do
    suite_name=${task_suites[$i]}
    cuda_id=${gpu_ids[$(( i % num_gpus ))]}
    log_file="$log_dir/${suite_name}_gpu${cuda_id}.log"

    echo "正在启动任务: $suite_name on GPU $cuda_id (Log: $log_file)"

    CUDA_VISIBLE_DEVICES=$cuda_id python scripts/infer_libero.py \
        --seed "$seed" \
        --mode "$mode" \
        --checkpoint_path "$checkpoint_path" \
        --norm_key "$norm_key" \
        --cam_names "${cam_names[@]}" \
        --task_suite_name "$suite_name" \
        --initial_states_path "$initial_states_path" \
        --num_trials_per_task "$num_trials_per_task" \
        --rollout_dir "$rollout_dir" \
        > "$log_file" 2>&1 &

    # 记录 PID
    pids+=($!)
done

# ================= 等待结束 =================

echo "所有任务已在后台启动。"
echo "正在等待所有任务完成..."
echo "可以使用 'tail -f $log_dir/*.log' 查看实时进度。"

# 打印 kill 命令提示
echo "如需终止所有进程，请执行："
echo "kill -9 ${pids[*]}"

wait

echo "所有评测任务已完成！"
