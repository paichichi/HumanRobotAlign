#!/usr/bin/env bash
#SBATCH --job-name=hralign_2a100_genoa
#SBATCH --account=uoa04758
#SBATCH --partition=genoa,milan
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=36:00:00
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

echo "============================================================"
echo "Job info"
echo "============================================================"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Python: $(which python)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
nvidia-smi

echo "============================================================"
echo "Sanity check"
echo "============================================================"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))

import pyrep
print("pyrep OK")

import rlbench
print("rlbench OK")

import point_renderer
import point_renderer._C
print("point_renderer._C OK")

import point_renderer.ops
print("point_renderer.ops OK")

import pytorch3d.ops
print("pytorch3d.ops OK")

import bitsandbytes as bnb
print("bitsandbytes:", getattr(bnb, "__version__", "unknown"))
PY

echo "============================================================"
echo "Start training"
echo "============================================================"

python train.py \
  --exp_cfg_path configs/unadaptedR3M_3.yaml \
  --mvt_cfg_path mvt/configs/rvt2.yaml \
  --device 0,1