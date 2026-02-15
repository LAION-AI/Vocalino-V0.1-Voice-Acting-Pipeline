#!/usr/bin/env python3
"""
============================================================================
 SETUP SCRIPT — Qwen3-TTS + Chatterbox Voice Pipeline
============================================================================

This script installs all dependencies and downloads model weights for:

  1. Qwen3-TTS Voice Design (1.7B) — Alibaba's text-to-speech model that
     generates expressive speech from text + a natural-language style
     instruction.  Produces audio in a random voice identity.

  2. Chatterbox VC — Resemble AI's voice conversion model that re-synthesises
     audio in a target speaker's voice using S3 speech tokens and a
     flow-matching decoder conditioned on a speaker embedding.

Dependency order matters:
  - PyTorch >= 2.6 first (required by transformers >= 4.52 for CVE fix)
  - transformers >= 4.52 next (needed by qwen-tts for auto_docstring)
  - qwen-tts last (pulls in its own deps without conflicts)

Usage:
    python setup.py

After setup, start the server:
    python server.py

Or run standalone scripts directly:
    python generate_samples.py --help
    python convert_voice.py --help
============================================================================
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60 + "\n")


def run(cmd, error_msg="Command failed", allow_fail=False):
    """Run a shell command verbosely.  Exits on failure unless allow_fail."""
    print(f"  $ {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"\n  [ERROR] {error_msg}  (exit code {ret})")
        if not allow_fail:
            sys.exit(1)
    return ret


def install_dependencies():
    banner("Step 1/4 — Python dependencies")

    # --- PyTorch 2.6+ with CUDA 12.4 ---
    # Required because transformers >= 4.52 refuses torch.load on torch < 2.6
    # (CVE-2025-32434).  The NVIDIA driver must be >= 525 for CUDA 12.4.
    print("[1a] PyTorch 2.6 + CUDA 12.4")
    run(
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "Failed to install PyTorch",
    )

    # --- Core Python packages ---
    print("\n[1b] Core packages")
    run(
        "pip install --upgrade "
        "'transformers>=4.52' accelerate soundfile librosa numpy scipy "
        "fastapi uvicorn python-multipart pydub "
        "huggingface_hub pyyaml",
        "Failed to install core packages",
    )

    # --- Qwen3-TTS ---
    # Must come after transformers so it doesn't downgrade it.
    print("\n[1c] Qwen3-TTS")
    run("pip install --upgrade qwen-tts", "Failed to install qwen-tts")

    # --- Chatterbox TTS (includes VC) ---
    print("\n[1d] Chatterbox TTS + VC")
    run("pip install --upgrade chatterbox-tts", "Failed to install chatterbox-tts")

    # --- Optional: flash-attention (speeds up Qwen inference) ---
    print("\n[1e] flash-attn (optional)")
    run(
        "pip install flash-attn --no-build-isolation",
        "flash-attn skipped (Qwen will use slower manual attention)",
        allow_fail=True,
    )


def download_models():
    banner("Step 2/4 — Downloading model weights")
    MODELS_DIR.mkdir(exist_ok=True)

    # --- Qwen3-TTS Voice Design ---
    qwen_dir = MODELS_DIR / "qwen_voice_design"
    if qwen_dir.exists() and (qwen_dir / "model.safetensors").exists():
        print(f"[SKIP] Qwen model already at {qwen_dir}")
    else:
        print(f"Downloading {QWEN_MODEL_ID} …")
        run(
            f"huggingface-cli download {QWEN_MODEL_ID} --local-dir {qwen_dir}",
            "Failed to download Qwen model",
        )

    # --- Chatterbox VC weights ---
    # These are auto-downloaded on first use by ChatterboxVC.from_pretrained(),
    # but we trigger it here so the user doesn't wait on first request.
    print("\nPre-downloading Chatterbox VC weights …")
    run(
        'python3 -c "from chatterbox.vc import ChatterboxVC; '
        "ChatterboxVC.from_pretrained(device='cpu')\"",
        "Chatterbox weight download failed (will retry at runtime)",
        allow_fail=True,
    )


def verify():
    banner("Step 3/4 — Verification")
    errors = []

    try:
        import torch
        print(f"  torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    except ImportError:
        errors.append("torch not importable")

    try:
        from qwen_tts import Qwen3TTSModel
        print("  qwen-tts OK")
    except Exception as e:
        errors.append(f"qwen-tts: {e}")

    try:
        from chatterbox.vc import ChatterboxVC
        print("  chatterbox VC OK")
    except Exception as e:
        errors.append(f"chatterbox: {e}")

    qwen_safetensors = MODELS_DIR / "qwen_voice_design" / "model.safetensors"
    if qwen_safetensors.exists():
        mb = qwen_safetensors.stat().st_size / 1024 / 1024
        print(f"  Qwen weights: {mb:.0f} MB")
    else:
        errors.append(f"Qwen weights missing: {qwen_safetensors}")

    if errors:
        print("\n[WARN] Issues:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n[OK] Everything looks good!")


def print_next_steps():
    banner("Step 4/4 — Ready!")
    print("Start the server:")
    print(f"  cd {ROOT_DIR}")
    print("  python server.py")
    print()
    print("Or run standalone scripts:")
    print("  python generate_samples.py --ref chris-ref.mp3")
    print("  python convert_voice.py --source input.wav --target ref.wav")


if __name__ == "__main__":
    banner("Qwen3-TTS + Chatterbox Voice Pipeline — Setup")
    install_dependencies()
    download_models()
    verify()
    print_next_steps()
