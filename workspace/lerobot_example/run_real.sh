#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
cd /mnt/data2/lfwj/wall-x4juliang
# print current time
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

code_dir="/mnt/data2/lfwj/wall-x4juliang"
config_path="/mnt/data2/lfwj/wall-x4juliang/workspace/lerobot_example/franka_real.yml"

# Pick a valid (integer) port for accelerate's rendezvous.
# If you only run 1 GPU/process, you can also omit --main_process_port entirely.
# PORT="${PORT:-29500}"
# SEED="${SEED:-10239}"

export PORT=21432

MASTER_PORT=10239 # use 5 digits ports
export LAUNCHER="accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT"

export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"

$LAUNCHER $SCRIPT $SCRIPT_ARGS

