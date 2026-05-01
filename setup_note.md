python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

for 
Python 3.11
PyTorch 2.10.0+cu128
CUDA runtime 12.8

could run test 
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("cuda test:", y.mean().item())
PY