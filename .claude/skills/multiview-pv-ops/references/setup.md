# Environment Setup

Complete setup for a fresh machine. Follow the steps in order.

---

## Step 1 — Detect Hardware

```bash
# Check if GPU is available and get CUDA version
nvidia-smi
```

Note the **CUDA Version** shown in the top-right corner (e.g., `12.4`).
If `nvidia-smi` is not found → CPU-only install (torch without CUDA).

Also note Python version:
```bash
python --version
# e.g., Python 3.11.x → cp311
```

---

## Step 2 — Create & Activate Conda Environment

```bash
conda create -n multiview-pv python=3.11 -y
conda activate multiview-pv
```

---

## Step 3 — Install GDAL

GDAL must be installed via conda-forge (pip builds are unreliable):

```bash
conda install -c conda-forge gdal -y
```

Verify:
```bash
python -c "from osgeo import gdal, ogr; print('GDAL OK', gdal.__version__)"
```

---

## Step 4 — Install PyTorch (GPU or CPU)

Select the command based on your CUDA version from Step 1.

**CUDA 11.8**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.6**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**No GPU (CPU only)**
```bash
pip install torch torchvision
```

Verify:
```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

## Step 5 — Install FlashAttention

FlashAttention requires a pre-built wheel matched to your exact environment. Do NOT `pip install flash-attn` directly — it triggers a long slow build.

### 5a. Collect version strings

```bash
# Python version tag (e.g., cp311 for Python 3.11)
python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"

# PyTorch version (e.g., 2.5 → torch2.5)
python -c "import torch; v=torch.__version__.split('.')[:2]; print('torch'+ '.'.join(v))"

# CUDA major version (e.g., CUDA 12.x → cu12)
nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1 | xargs -I{} echo "cu{}"
```

### 5b. Find the matching wheel

Go to: **https://github.com/Dao-AILab/flash-attention/releases**

Use the `WebFetch` tool to read the releases page, then look for a `.whl` filename that matches all of:
- `cu{CUDA_MAJOR}` — e.g., `cu12`
- `torch{TORCH_MAJOR.MINOR}` — e.g., `torch2.5`
- `cp{PYVER}-cp{PYVER}` — e.g., `cp311-cp311`
- `linux_x86_64`
- `cxx11abiTRUE` (standard; use `FALSE` only if your environment explicitly uses the old ABI)

Typical filename pattern:
```
flash_attn-{fa_ver}+cu{cuda}torch{torch}cxx11abi{TRUE/FALSE}-cp{py}-cp{py}-linux_x86_64.whl
```

Example:
```
flash_attn-2.7.4+cu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

### 5c. Install from the release URL

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v{FA_VER}/{FILENAME}
```

If no pre-built wheel matches your exact environment, fall back to source build (slow, ~10 min):
```bash
pip install flash-attn --no-build-isolation
```

---

## Step 6 — Install Other Python Dependencies

```bash
pip install \
  omegaconf \
  tqdm \
  opencv-python \
  shapely \
  rtree \
  networkx \
  pillow \
  scipy \
  modelscope
```

---

## Step 7 — Clone SAM3 Submodule

The SAM3 source lives in `third_party/sam3/`. Clone it now (independently of the main repo clone):

```bash
# From the project root
git submodule update --init third_party/sam3
```

Then install it as an editable package:

```bash
pip install -e third_party/sam3
```

Verify:
```bash
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 import OK')"
```

---

## Step 8 — Download SAM3 Weights

Weights are downloaded from ModelScope and placed in the `weights/` folder.

```bash
# Download to weights/sam3/
python -c "
from modelscope import snapshot_download
snapshot_download('facebook/sam3', cache_dir='weights/sam3_cache')
"
```

After download, locate the checkpoint file (typically `*.pt`) and BPE vocab file (`bpe_simple_vocab_16e6.txt.gz`):

```bash
find weights/ -name "*.pt" -o -name "*.txt.gz" 2>/dev/null
```

> **Note**: The BPE vocab file may also be found inside the submodule at
> `third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`

Set the exact paths in your station config (see `new-station.md`):
```yaml
model:
  checkpoint_path: weights/sam3_cache/facebook/sam3/sam3.pt   # adjust to actual filename
  bpe_path: third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

---

## Verification Checklist

```bash
# 1. GDAL
python -c "from osgeo import gdal, ogr, osr; print('GDAL:', gdal.__version__)"

# 2. PyTorch + CUDA
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# 3. FlashAttention
python -c "import flash_attn; print('FlashAttn:', flash_attn.__version__)"

# 4. SAM3
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"

# 5. Other core libs
python -c "import cv2, shapely, rtree, networkx, omegaconf; print('All libs OK')"
```

All checks passing → environment is ready.
