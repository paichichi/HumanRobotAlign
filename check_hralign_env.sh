#!/usr/bin/env bash

set +e

# ============================================================
# HR-Align / RVT Environment Checker
# Usage:
#   bash tools/check_hralign_env.sh
#   bash tools/check_hralign_env.sh ~/projects/HumanRobotAlign
#   bash tools/check_hralign_env.sh ~/projects/HumanRobotAlign --launch
#
# Notes:
#   --launch will try to launch PyRep / RLBench through xvfb-run.
# ============================================================

PROJECT_ROOT="${1:-$(pwd)}"
LAUNCH_TEST=0

if [[ "$PROJECT_ROOT" == "--launch" ]]; then
  PROJECT_ROOT="$(pwd)"
  LAUNCH_TEST=1
fi

if [[ "${2:-}" == "--launch" ]]; then
  LAUNCH_TEST=1
fi

RVT_ROOT="$PROJECT_ROOT/rvt"

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[1;34m"
NC="\033[0m"

ok() {
  echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

fail() {
  echo -e "${RED}[FAIL]${NC} $1"
}

info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_cmd() {
  echo
  echo "\$ $*"
  eval "$@"
  local code=$?
  if [[ $code -eq 0 ]]; then
    ok "Command succeeded"
  else
    fail "Command failed with exit code $code"
  fi
  return $code
}

exists_file() {
  if [[ -f "$1" ]]; then
    ok "$1 exists"
  else
    fail "$1 not found"
  fi
}

exists_dir() {
  if [[ -d "$1" ]]; then
    ok "$1 exists"
  else
    fail "$1 not found"
  fi
}

section "0. Basic Project Path"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RVT_ROOT=$RVT_ROOT"

exists_dir "$PROJECT_ROOT"
exists_dir "$RVT_ROOT"
exists_dir "$RVT_ROOT/libs"

echo
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "Kernel: $(uname -a)"

if grep -qi microsoft /proc/version 2>/dev/null; then
  warn "You are running inside WSL/WSL2"
else
  info "Not detected as WSL"
fi

section "1. Conda / Python / pip"

echo "CONDA_PREFIX=${CONDA_PREFIX:-<empty>}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<empty>}"

run_cmd "which python"
run_cmd "python --version"
run_cmd "which pip"
run_cmd "python -m pip --version"

python - <<'PY'
import sys
import site
import platform
from pathlib import Path

print("sys.executable:", sys.executable)
print("sys.version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("site.getsitepackages():")
try:
    for p in site.getsitepackages():
        print("  -", p)
except Exception as e:
    print("  ERROR:", repr(e))

print("sys.path first 10:")
for p in sys.path[:10]:
    print("  -", p)
PY

section "2. CUDA Driver / GPU / nvcc"

run_cmd "which nvidia-smi"
run_cmd "nvidia-smi"

run_cmd "which nvcc"
run_cmd "nvcc --version"

echo
echo "CUDA_HOME=${CUDA_HOME:-<empty>}"
echo "PATH first 20:"
echo "$PATH" | tr ':' '\n' | head -20

echo
echo "LD_LIBRARY_PATH entries containing cuda/CUDA/CoppeliaSim:"
echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -Ei "cuda|coppelia|conda" || true

section "3. Torch / CUDA Runtime / cuDNN"

python - <<'PY'
import os
import sys

print("Python executable:", sys.executable)

try:
    import torch
    print("[OK] import torch")
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda:{i} name:", torch.cuda.get_device_name(i))
            print(f"cuda:{i} capability:", torch.cuda.get_device_capability(i))
            props = torch.cuda.get_device_properties(i)
            print(f"cuda:{i} total_memory_GB:", round(props.total_memory / 1024**3, 2))

    from torch.utils.cpp_extension import CUDA_HOME
    print("torch.utils.cpp_extension.CUDA_HOME:", CUDA_HOME)

except Exception as e:
    print("[FAIL] torch check failed:", repr(e))
PY

section "4. CUDA Headers / Libraries Required by point_renderer"

echo "Checking curand.h, cuda_runtime.h, libcudart..."

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  find "$CONDA_PREFIX" -name curand.h 2>/dev/null | head -20
  find "$CONDA_PREFIX" -name cuda_runtime.h 2>/dev/null | head -20
  find "$CONDA_PREFIX" -name "libcudart*" 2>/dev/null | head -20
else
  warn "CONDA_PREFIX is empty; cannot search conda env"
fi

echo
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if find "$CONDA_PREFIX" -name curand.h 2>/dev/null | grep -q curand.h; then
    ok "curand.h found"
  else
    fail "curand.h not found. You may need: conda install -c nvidia cuda-toolkit=12.8 -y"
  fi

  if find "$CONDA_PREFIX" -name cuda_runtime.h 2>/dev/null | grep -q cuda_runtime.h; then
    ok "cuda_runtime.h found"
  else
    fail "cuda_runtime.h not found. You may need CUDA toolkit dev headers."
  fi
fi

section "5. C / C++ Compiler"

run_cmd "which gcc"
run_cmd "gcc --version | head -5"

run_cmd "which g++"
run_cmd "g++ --version | head -5"

echo
echo "CC=${CC:-<empty>}"
echo "CXX=${CXX:-<empty>}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX:-<empty>}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  exists_file "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  exists_file "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi

section "6. PyTorch3D Check"

python - <<'PY'
import sys
import importlib.util
import subprocess

print("Checking PyTorch3D...")

spec = importlib.util.find_spec("pytorch3d")
if spec is None:
    print("[FAIL] pytorch3d module not found")
else:
    print("[OK] pytorch3d module found")
    print("pytorch3d origin:", spec.origin)

try:
    import pytorch3d
    print("pytorch3d.__version__:", getattr(pytorch3d, "__version__", "<no __version__>"))
except Exception as e:
    print("[FAIL] import pytorch3d failed:", repr(e))

try:
    import pytorch3d.ops
    print("[OK] import pytorch3d.ops")
except Exception as e:
    print("[WARN] import pytorch3d.ops failed:", repr(e))

print()
print("pip show pytorch3d:")
subprocess.run([sys.executable, "-m", "pip", "show", "pytorch3d"])
PY

section "7. Editable Package Paths"

declare -A LOCAL_PATHS
LOCAL_PATHS["PyRep"]="$RVT_ROOT/libs/PyRep"
LOCAL_PATHS["RLBench"]="$RVT_ROOT/libs/RLBench"
LOCAL_PATHS["YARR"]="$RVT_ROOT/libs/YARR"
LOCAL_PATHS["peract_colab"]="$RVT_ROOT/libs/peract_colab"
LOCAL_PATHS["point-renderer"]="$RVT_ROOT/libs/point-renderer"

for name in "PyRep" "RLBench" "YARR" "peract_colab" "point-renderer"; do
  echo
  echo "---- $name ----"
  path="${LOCAL_PATHS[$name]}"
  exists_dir "$path"
  if [[ -f "$path/setup.py" ]]; then
    ok "$path/setup.py exists"
  else
    warn "$path/setup.py not found"
  fi
  if [[ -f "$path/pyproject.toml" ]]; then
    ok "$path/pyproject.toml exists"
  else
    info "$path/pyproject.toml not found"
  fi
done

section "8. pip show for the 5 editable dependencies"

python - <<'PY'
import sys
import subprocess

pip_names = [
    "pyrep",
    "rlbench",
    "yarr",
    "peract_colab",
    "point_renderer",
]

for name in pip_names:
    print("\n" + "-" * 60)
    print("pip show", name)
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", name],
        text=True
    )
    if result.returncode == 0:
        print(f"[OK] pip package found: {name}")
    else:
        print(f"[WARN] pip package not found by name: {name}")
PY

section "9. Python Import Check for Key Packages"

python - <<'PY'
import importlib
import importlib.util
import traceback
import sys
from pathlib import Path

targets = {
    "torch": ["torch"],
    "pytorch3d": ["pytorch3d", "pytorch3d.ops"],
    "pyrep": ["pyrep"],
    "rlbench": ["rlbench"],
    "yarr": ["yarr"],
    "point_renderer": ["point_renderer", "point_renderer._C"],

    # peract_colab 的 top-level module 可能因 repo 版本不同而不同。
    # 所以这里检测常见模块，失败不一定代表 pip install -e 没装。
    "peract_colab_possible_modules": [
        "peract_colab",
        "agents",
        "helpers",
        "launch_utils",
    ],
}

for group, modules in targets.items():
    print("\n" + "=" * 60)
    print(group)
    print("=" * 60)

    for mod in modules:
        print(f"Importing {mod} ...")
        try:
            m = importlib.import_module(mod)
            origin = getattr(m, "__file__", "<built-in or namespace>")
            print(f"[OK] {mod}")
            print(f"     origin: {origin}")
        except Exception as e:
            print(f"[FAIL/WARN] {mod}: {repr(e)}")
PY

section "10. Detailed Distribution Metadata"

python - <<'PY'
import json
import sys
from pathlib import Path
import importlib.metadata as md

dist_names = [
    "pyrep",
    "rlbench",
    "yarr",
    "peract_colab",
    "point_renderer",
    "pytorch3d",
    "torch",
]

for name in dist_names:
    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    try:
        dist = md.distribution(name)
        print("[OK] distribution found")
        print("version:", dist.version)
        print("location:", dist.locate_file(""))
        print("metadata name:", dist.metadata.get("Name"))

        direct_url = None
        for f in dist.files or []:
            if str(f).endswith("direct_url.json"):
                direct_url = dist.locate_file(f)
                break

        if direct_url and Path(direct_url).exists():
            print("direct_url.json:", direct_url)
            try:
                print(Path(direct_url).read_text())
            except Exception as e:
                print("cannot read direct_url.json:", repr(e))
        else:
            print("direct_url.json: <not found>")

    except Exception as e:
        print("[WARN] distribution not found or unreadable:", repr(e))
PY

section "11. CoppeliaSim Environment Check"

echo "COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-<empty>}"
echo "QT_QPA_PLATFORM_PLUGIN_PATH=${QT_QPA_PLATFORM_PLUGIN_PATH:-<empty>}"
echo "QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-<empty>}"

if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
  fail "COPPELIASIM_ROOT is not set"
  echo "Example:"
  echo "  export COPPELIASIM_ROOT=/home/paichichi/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"
else
  exists_dir "$COPPELIASIM_ROOT"
  exists_file "$COPPELIASIM_ROOT/coppeliaSim.sh"
  exists_file "$COPPELIASIM_ROOT/libcoppeliaSim.so"

  echo
  echo "CoppeliaSim files:"
  ls "$COPPELIASIM_ROOT" | head -30
fi

echo
echo "LD_LIBRARY_PATH entries:"
echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | nl | head -50

if [[ "${QT_QPA_PLATFORM:-}" == "offscreen" ]]; then
  warn "QT_QPA_PLATFORM=offscreen detected. For RLBench/PyRep this may cause OpenGL context errors. Usually use: unset QT_QPA_PLATFORM"
fi

section "12. Xvfb Check"

run_cmd "which xvfb-run"
run_cmd "which Xvfb"

if command -v xvfb-run >/dev/null 2>&1; then
  run_cmd "xvfb-run -a -e /tmp/hralign_xvfb_basic.log -s \"-screen 0 1024x768x24 +extension GLX +render -noreset\" bash -lc 'echo DISPLAY=\$DISPLAY; sleep 1'"
else
  warn "xvfb-run not found. If no sudo, try: conda install -c conda-forge xorg-xserver-xvfb -y"
fi

section "13. Optional Launch Test"

if [[ "$LAUNCH_TEST" -eq 0 ]]; then
  info "Launch test skipped."
  echo "To run launch test:"
  echo "  bash tools/check_hralign_env.sh $PROJECT_ROOT --launch"
else
  info "Running PyRep / RLBench launch test through xvfb-run..."

  if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
    fail "Cannot launch: COPPELIASIM_ROOT is empty"
  elif ! command -v xvfb-run >/dev/null 2>&1; then
    fail "Cannot launch: xvfb-run not found"
  else
    echo
    echo "---- PyRep launch test ----"
    xvfb-run -a -e /tmp/hralign_pyrep_xvfb.log -s "-screen 0 1024x768x24 +extension GLX +render -noreset" python - <<'PY'
from pyrep import PyRep

pr = PyRep()
pr.launch('', headless=True)
print("[OK] PyRep launched CoppeliaSim")
pr.shutdown()
print("[OK] PyRep shutdown")
PY
    PYREP_CODE=$?
    if [[ $PYREP_CODE -eq 0 ]]; then
      ok "PyRep launch test passed"
    else
      fail "PyRep launch test failed. Check /tmp/hralign_pyrep_xvfb.log"
      cat /tmp/hralign_pyrep_xvfb.log 2>/dev/null | tail -80
    fi

    echo
    echo "---- RLBench launch test ----"
    xvfb-run -a -e /tmp/hralign_rlbench_xvfb.log -s "-screen 0 1024x768x24 +extension GLX +render -noreset" python - <<'PY'
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)

env = Environment(
    action_mode=action_mode,
    obs_config=None,
    headless=True
)

env.launch()
print("[OK] RLBench launched")
env.shutdown()
print("[OK] RLBench shutdown")
PY
    RLBENCH_CODE=$?
    if [[ $RLBENCH_CODE -eq 0 ]]; then
      ok "RLBench launch test passed"
    else
      fail "RLBench launch test failed. Check /tmp/hralign_rlbench_xvfb.log"
      cat /tmp/hralign_rlbench_xvfb.log 2>/dev/null | tail -120
    fi
  fi
fi

section "14. Final Summary"

echo "If all key items are OK, your environment should satisfy:"
echo "  1. Python / conda env active"
echo "  2. torch + CUDA runtime OK"
echo "  3. nvcc + CUDA headers OK"
echo "  4. PyTorch3D import OK"
echo "  5. PyRep / RLBench / YARR / peract_colab / point_renderer installed"
echo "  6. CoppeliaSim variables configured"
echo "  7. xvfb-run available for headless simulation"
echo
echo "Important expected env vars:"
echo "  CUDA_HOME=$CONDA_PREFIX"
echo "  COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"
echo "  LD_LIBRARY_PATH should include \$CONDA_PREFIX/lib and \$COPPELIASIM_ROOT"
echo "  QT_QPA_PLATFORM_PLUGIN_PATH should usually be \$COPPELIASIM_ROOT"
echo "  QT_QPA_PLATFORM should usually be unset"
echo
echo "Done."