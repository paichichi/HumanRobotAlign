#!/usr/bin/env bash
#SBATCH --job-name=hralign_2a100
#SBATCH --account=uoa04758
#SBATCH --partition=milan
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/nesi/project/uoa04758/xzha593/logs/%x-%j.out
#SBATCH --error=/nesi/project/uoa04758/xzha593/logs/%x-%j.err

set -euo pipefail

mkdir -p /nesi/project/uoa04758/xzha593/logs

source /nesi/project/uoa04758/xzha593/envs/activate_hralign.sh

cd /nesi/project/uoa04758/xzha593/GitHub/HumanRobotAlign/rvt

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Python: $(which python)"
nvidia-smi

python train.py \
  --exp_cfg_path configs/unadaptedR3M.yaml \
  --device 0,1 \
# python train.py \
#   --exp_cfg_path configs/adaptedR3M.yaml \
#   --device 0 \
#   --refresh_replay