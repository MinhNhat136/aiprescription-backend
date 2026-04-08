# Offline Whisper Model Setup

This document describes how to pre-download the Whisper model for offline deployment on air-gapped Windows machines.

## Overview

The VoiceFillPrescription application uses `faster-whisper` with the `large-v3-turbo` model for Vietnamese speech-to-text. On first run, the model (~3GB) is automatically downloaded from HuggingFace. For **offline/air-gapped deployment**, pre-download the model before first startup.

## Prerequisites

- Python 3.12+ installed
- `faster-whisper` installed: `pip install faster-whisper`
- ~5GB free disk space
- Internet connection (for download phase only)

## Quick Start

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Download the Model

Run the download script with your desired model size:

```bash
# Default: large-v3-turbo (recommended for Vietnamese medical transcription)
python scripts/download_model.py --output D:\Models\whisper

# Or use environment variable for model size
set WHISPER_MODEL=large-v3-turbo
python scripts/download_model.py --output D:\Models\whisper
```

**First run download time:** 10-30 minutes depending on internet speed (~3GB).

### Step 3: Configure the Application

Edit `backend/.env` to point to the downloaded model:

```env
# Point to your local model directory
WHISPER_MODEL_PATH=D:\Models\whisper

# Optional: HuggingFace cache directory
# HF_HOME=D:\Models\huggingface
```

### Step 4: Verify Offline Operation

```bash
# Verify the model works without internet
python scripts/download_model.py --verify --model-path D:\Models\whisper --model large-v3-turbo
```

### Step 5: Start the Application

```bash
# Backend
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev
```

## Model Options

### Model Sizes

| Model | Parameters | Size | VRAM (GPU) | CPU Speed | Vietnamese Quality |
|-------|------------|------|------------|-----------|-------------------|
| `tiny` | 39M | ~75MB | ~1GB | Fastest | Basic |
| `base` | 74M | ~150MB | ~1GB | Fast | Good |
| `small` | 244M | ~500MB | ~2GB | Fast | Better |
| `medium` | 769M | ~1.5GB | ~4GB | Medium | Best |
| `large-v3` | 1.5B | ~3GB | ~6GB | Slow | Excellent |
| `large-v3-turbo` | 809M | ~1.6GB | ~6GB | Medium | Excellent (recommended) |

**Recommendation:** `large-v3-turbo` offers excellent Vietnamese transcription quality with ~8x faster inference than `large-v3` and only 809M parameters.

### Vietnamese Fine-tuned Model

For potentially better Vietnamese medical terminology accuracy, consider the community fine-tuned model:

```bash
python scripts/download_model.py --model suzii/vi-whisper-large-v3-turbo-v1 --output D:\Models\whisper
```

> **Note:** This model is fine-tuned on 240 hours of Vietnamese medical audio. Test both and compare transcription accuracy for your use case.

## Troubleshooting

### Model Download Fails

1. Check internet connectivity
2. Increase download timeout:
   ```env
   HF_HUB_DOWNLOAD_TIMEOUT=1200
   ```
3. Use a VPN if behind a restricted network
4. Try downloading via HuggingFace CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli download Systran/faster-whisper-large-v3-turbo
   ```

### Model Not Found After Download

The model is cached by HuggingFace. Verify the path:

```bash
# Default HuggingFace cache location
dir %USERPROFILE%\.cache\huggingface\hub\

# Check for model directory
dir %USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3-turbo/
```

### CUDA/GPU Errors

If running on CPU but CUDA errors occur:

1. Set device to CPU in `.env`:
   ```env
   DEVICE=cpu
   ```

2. Verify GPU drivers are installed if using CUDA:
   ```bash
   nvidia-smi
   ```

### Model Loads Slowly

- First load compiles CTranslate2 kernels (~30 seconds, one-time)
- Subsequent loads use cached compiled kernels
- Consider using `--compute-type int8` for faster CPU inference

## Model Storage Locations

| Configuration | Location |
|--------------|----------|
| Default (HF cache) | `%USERPROFILE%\.cache\huggingface\hub\` |
| Custom `HF_HOME` | `$HF_HOME\hub\` |
| Custom `WHISPER_MODEL_PATH` | `$WHISPER_MODEL_PATH\` |

## Air-Gapped Deployment Checklist

Before deploying to an offline machine:

- [ ] Download model on internet-connected machine
- [ ] Copy model files to deployment machine
- [ ] Set `WHISPER_MODEL_PATH` to local directory
- [ ] Set `DEVICE=cpu` if no GPU on deployment machine
- [ ] Test startup: `python -c "from faster_whisper import WhisperModel; m = WhisperModel('large-v3-turbo', device='cpu')"`
- [ ] Verify transcription: `python scripts/download_model.py --verify`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Model size to download/use |
| `WHISPER_MODEL_PATH` | (none) | Local directory with model files |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `600` | Download timeout in seconds |
| `DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
