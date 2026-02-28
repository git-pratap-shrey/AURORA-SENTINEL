# AURORA SENTINEL — Dev Replication Guide

Step-by-step instructions to set up AURORA SENTINEL on a new machine from scratch.

---

## Step 1 — Install System Prerequisites

Install these **before** cloning. None of these come from `pip` or `npm`.

| Tool | Minimum Version | Install |
|---|---|---|
| **Python** | 3.10+ | https://python.org → check "Add to PATH" during install |
| **Node.js** | 18 LTS | https://nodejs.org |
| **Git** | Latest | https://git-scm.com |
| **FFmpeg** | Latest | `winget install ffmpeg` (Windows) or https://ffmpeg.org/download.html — must be in PATH |
| **Ollama** | Latest | https://ollama.com → only if using `VLM_PROVIDER=ollama` |

Verify installs:
```powershell
python --version    # should be 3.10+
node --version      # should be 18+
npm --version
git --version
ffmpeg -version
ollama --version    # only if using Ollama
```

---

## Step 2 — Clone the Repository

```bash
git clone https://github.com/git-pratap-shrey/AURORA-SENTINEL.git
cd AURORA-SENTINEL
git checkout finalreplication
```

---

## Step 3 — Create Python Virtual Environment

```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 4 — Install Python Dependencies

### Option A — CPU only (no GPU)
```bash
pip install -r requirements.txt
```

### Option B — NVIDIA GPU (CUDA 11.8) — Recommended for performance
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
pip install -r requirements.txt
```

> **Note:** CUDA 11.8 requires an NVIDIA GPU with drivers ≥ 450.80.02.  
> Check your CUDA version: `nvidia-smi`

---

## Step 5 — Configure Environment Variables

```powershell
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Now open `.env` and fill in your values:

```env
VLM_PROVIDER=ollama           # or: gemini

# Only needed if VLM_PROVIDER=gemini:
GEMINI_API_KEY=your_key_here  # get from https://aistudio.google.com/app/apikey

# Only needed if VLM_PROVIDER=ollama:
OLLAMA_MODEL=llava            # must match what you pulled in Step 6
```

---

## Step 6 — Pull Ollama Model (Ollama users only)

If `VLM_PROVIDER=ollama`, download a vision-capable model:

```bash
# Recommended (4.7 GB)
ollama pull llava

# Higher quality option (8 GB RAM required)
ollama pull llava:13b
```

Make sure Ollama is running before starting the backend:
```bash
ollama serve   # runs in background automatically on Windows after install
```

---

## Step 7 — Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Step 8 — Extract Large Assets (Shared RAR)

Certain files are too large for GitHub and are shared separately as **`aurora_large_assets.rar`**.
Request this file from the project owner.

### What's inside the RAR

| File / Folder | Size | Purpose |
|---|---|---|
| `wepon.pt` | ~250 MB | Custom weapon detection model |
| `yolov8n.pt` | ~6 MB | YOLOv8 Nano — general detection |
| `yolov8s.pt` | ~22 MB | YOLOv8 Small — general detection |
| `yolov8n-pose.pt` | ~6 MB | YOLOv8 pose estimation |
| `data/` | ~950 MB | Sample surveillance videos for testing |

### How to extract

Extract the RAR **directly into the project root** (same folder as `requirements.txt`):

```
AURORA-SENTINEL/        ← extract here (overwrite if prompted)
├── wepon.pt
├── yolov8n.pt
├── yolov8s.pt
├── yolov8n-pose.pt
└── data/
```

> **Windows:** Right-click `aurora_large_assets.rar` → *Extract Here* (WinRAR / 7-Zip)  
> **macOS/Linux:** `unrar x aurora_large_assets.rar /path/to/AURORA-SENTINEL/`

---

## Step 9 — Run the Application

### Option A — One-click (Windows)
```powershell
.\start.ps1
```

### Option B — Manual (any OS)

**Terminal 1 — Backend:**
```bash
.\venv\Scripts\activate         # Windows
# source venv/bin/activate      # macOS/Linux

python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm start
```

---

## Access Points

| Service | URL |
|---|---|
| Frontend Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |

---

## Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | venv not activated — run `.\venv\Scripts\activate` first |
| `ffmpeg not found` | FFmpeg binary not in PATH — reinstall and check PATH |
| `ollama: connection refused` | Ollama not running — open Ollama app or run `ollama serve` |
| Port 8000 in use | `netstat -ano \| findstr :8000` → `taskkill /PID <pid> /F` |
| `wepon.pt not found` | Extract `aurora_large_assets.rar` into the project root (Step 8) |
| Torch GPU not detected | Install CUDA-compatible torch (Step 4 Option B) |
| `npm: command not found` | Install Node.js from nodejs.org |
