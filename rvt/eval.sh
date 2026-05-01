#!/usr/bin/env bash
set -euo pipefail

cd /home/xli990/paichichi/GitHub/HumanRobotAlign/rvt

export COPPELIASIM_ROOT=/home/xli990/software/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:$CONDA_PREFIX/lib:/usr/lib/nvidia
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=xcb

export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#   --eval-datafolder "/home/xli990/paichichi/GitHub/X-ICM/data/unseen_tasks/test" \
xvfb-run -a\
 -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
 python eval.py \
  --model-folder "runs/UnadaptedR3M2RLBench" \
  --eval-datafolder "/home/xli990/paichichi/GitHub/HumanRobotAlign/rvt/runs/data/test" \
  --tasks "push_buttons" \
  --eval-episodes 25 \
  --episode-length 25 \
  --log-name "debug_original_push_buttons" \
  --device 0 \
  --headless \
  --model-name "model_4.pth"