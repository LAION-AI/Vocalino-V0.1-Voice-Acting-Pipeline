#!/usr/bin/env python3
"""
============================================================================
 FastAPI SERVER — Qwen3-TTS Voice Design + Chatterbox Voice Conversion
============================================================================

HIGH-LEVEL OVERVIEW
-------------------
This server exposes a two-stage AI voice pipeline over HTTP:

  Stage 1 — Text-to-Speech (Qwen3-TTS Voice Design, 1.7B parameters)
      Takes plain text + a natural-language style instruction (e.g.
      "speak with trembling fear, medium-pitched male voice") and generates
      expressive speech audio.  The voice *identity* is random each time —
      only the emotion, pace, and style are controlled.

  Stage 2 — Voice Conversion (Chatterbox VC by Resemble AI)
      Takes any audio + a short reference clip of a target speaker and
      re-synthesises the audio in that speaker's voice.  Internally it:
        a) Tokenises the source audio into discrete S3 speech tokens
           (capturing phonetic content and prosody)
        b) Extracts a speaker embedding from the reference clip
        c) Decodes new audio through a flow-matching model conditioned
           on both the S3 tokens and the speaker embedding
      The result sounds like the target person saying the source content
      with the source emotion/style.

  Combined Pipeline — TTS then VC in one call:
      Send text + style instruction + reference audio and get back audio
      that sounds like the reference speaker saying the text with the
      described emotion.

ENDPOINTS
---------
  POST /tts/generate-voice-design   — Stage 1 only (TTS)
  POST /vc/convert                  — Stage 2 only (voice conversion)
  POST /pipeline/tts-then-vc        — Both stages combined
  GET  /health                      — Check which models are loaded

All audio is sent/received as base64-encoded WAV or MP3 strings in JSON.

STARTUP
-------
  python server.py                     # default: 0.0.0.0:8000
  python server.py --port 9000         # custom port

Both models are loaded into GPU memory on startup (~5 GB total).
============================================================================
"""

import sys
import os
import argparse
import traceback
import tempfile
import uuid
import gc

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
import soundfile as sf
import base64
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Qwen3-TTS import
# ---------------------------------------------------------------------------
from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Chatterbox VC import
# ---------------------------------------------------------------------------
from chatterbox.vc import ChatterboxVC

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

# ---------------------------------------------------------------------------
# Global model handles (populated during startup)
# ---------------------------------------------------------------------------
_qwen: Optional[Qwen3TTSModel] = None   # TTS model
_vc: Optional[ChatterboxVC] = None       # Voice conversion model
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# MODEL LOADING
# ===================================================================

def _load_qwen():
    """Load Qwen3-TTS Voice Design model.

    Tries the local models/ directory first; falls back to downloading
    from HuggingFace if not found.
    """
    global _qwen
    local_path = MODELS_DIR / "qwen_voice_design"

    if local_path.exists() and (local_path / "model.safetensors").exists():
        source = str(local_path)
    else:
        source = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    print(f"[INFO] Loading Qwen3-TTS from {source} …")
    _qwen = Qwen3TTSModel.from_pretrained(
        source,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    print("[INFO] Qwen3-TTS ready.")


def _load_chatterbox():
    """Load Chatterbox VC model.

    Weights are auto-downloaded from HuggingFace on first use and cached
    in the default huggingface_hub cache directory.
    """
    global _vc
    print("[INFO] Loading Chatterbox VC …")
    _vc = ChatterboxVC.from_pretrained(device=str(_device))
    print(f"[INFO] Chatterbox VC ready (sr={_vc.sr}).")


# ===================================================================
# LIFESPAN — load models at startup, release at shutdown
# ===================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  Loading models …")
    print("=" * 60)

    # Load TTS
    try:
        _load_qwen()
    except Exception:
        print("[ERROR] Qwen3-TTS failed to load:")
        traceback.print_exc()

    # Load VC
    try:
        _load_chatterbox()
    except Exception:
        print("[ERROR] Chatterbox VC failed to load:")
        traceback.print_exc()

    print("=" * 60)
    print("  Server ready.")
    print("=" * 60)

    yield  # ── server runs here ──

    # Cleanup
    print("[INFO] Shutting down …")
    global _qwen, _vc
    _qwen = None
    _vc = None
    gc.collect()
    torch.cuda.empty_cache()


# ===================================================================
# FASTAPI APP
# ===================================================================

app = FastAPI(
    title="Qwen3-TTS + Chatterbox Voice Pipeline",
    description="Generate styled speech and convert it to any voice.",
    lifespan=lifespan,
)


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def audio_to_b64(audio: np.ndarray, sr: int) -> str:
    """Encode a float32 numpy waveform as a base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def b64_to_wav_file(b64: str, tmp_dir: str, name: str = "audio") -> str:
    """Decode a base64 audio blob (WAV or MP3) and write it as a WAV file.

    Returns the path to the temporary WAV file.  The caller is responsible
    for cleaning up tmp_dir after use.
    """
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)

    # Try reading directly (works for WAV, FLAC, OGG)
    try:
        data, sr = sf.read(buf)
    except Exception:
        # Fallback: might be MP3 — decode with pydub
        from pydub import AudioSegment
        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        data = samples / (2 ** (seg.sample_width * 8 - 1))

    path = os.path.join(tmp_dir, f"{name}_{uuid.uuid4().hex[:8]}.wav")
    sf.write(path, data, sr)
    return path


# ===================================================================
# REQUEST / RESPONSE SCHEMAS
# ===================================================================

class TTSRequest(BaseModel):
    """Generate speech with Qwen3-TTS Voice Design."""
    text: str = Field(..., description="Text to synthesize.")
    style_prompt: str = Field(..., description=(
        "Natural-language description of desired voice style, emotion, "
        "pitch, and pacing.  Example: 'Speak with warm affection, "
        "medium-pitched male voice, slow contemplative pace.'"
    ))
    language: str = Field("Auto", description="Language: Auto, English, Chinese, Japanese, …")
    max_new_tokens: Optional[int] = Field(None, description="Limit generated codec tokens.")
    top_p: Optional[float] = Field(None)
    temperature: Optional[float] = Field(None)
    repetition_penalty: Optional[float] = Field(None)


class VCRequest(BaseModel):
    """Voice conversion with Chatterbox VC."""
    source_audio_base64: str = Field(..., description="Base64-encoded source audio (WAV/MP3).")
    target_audio_base64: str = Field(..., description="Base64-encoded reference audio for target speaker.")


class PipelineRequest(BaseModel):
    """Full pipeline: TTS Voice Design → Chatterbox VC."""
    text: str = Field(..., description="Text to synthesize.")
    style_prompt: str = Field(..., description="Voice style instruction for TTS.")
    target_audio_base64: str = Field(..., description="Reference audio for target speaker identity.")
    language: str = Field("Auto")
    # Optional TTS generation kwargs
    tts_max_new_tokens: Optional[int] = None
    tts_top_p: Optional[float] = None
    tts_temperature: Optional[float] = None
    tts_repetition_penalty: Optional[float] = None
    # Whether to also return the intermediate TTS audio (before VC)
    return_intermediate: bool = Field(False)


# ===================================================================
# ENDPOINTS
# ===================================================================

@app.post("/tts/generate-voice-design")
async def tts_generate(req: TTSRequest):
    """Generate speech with Qwen3-TTS Voice Design.

    Returns base64-encoded WAV audio of the generated speech.
    The voice identity is random — only style/emotion are controlled.
    """
    if _qwen is None:
        raise HTTPException(500, "Qwen3-TTS model not loaded.")

    # Forward optional generation kwargs
    kwargs = {}
    for attr in ("max_new_tokens", "top_p", "temperature", "repetition_penalty"):
        val = getattr(req, attr)
        if val is not None:
            kwargs[attr] = val

    try:
        wavs, sr = _qwen.generate_voice_design(
            text=req.text,
            instruct=req.style_prompt,
            language=req.language,
            **kwargs,
        )
        return {
            "status": "success",
            "sample_rate": sr,
            "audio_base64": audio_to_b64(wavs[0], sr),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"TTS failed: {e}")


@app.post("/vc/convert")
async def vc_convert(req: VCRequest):
    """Convert source audio to the target speaker's voice with Chatterbox VC.

    Both source and target audio are base64-encoded strings.
    Returns base64-encoded WAV of the converted audio.
    """
    if _vc is None:
        raise HTTPException(500, "Chatterbox VC model not loaded.")

    tmp_dir = tempfile.mkdtemp(prefix="vc_")
    try:
        src_path = b64_to_wav_file(req.source_audio_base64, tmp_dir, "source")
        tgt_path = b64_to_wav_file(req.target_audio_base64, tmp_dir, "target")

        # Run Chatterbox voice conversion
        result = _vc.generate(audio=src_path, target_voice_path=tgt_path)

        # result is a (1, samples) tensor
        wav = result.squeeze(0).cpu().numpy().astype(np.float32)
        return {
            "status": "success",
            "sample_rate": _vc.sr,
            "audio_base64": audio_to_b64(wav, _vc.sr),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Voice conversion failed: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/pipeline/tts-then-vc")
async def pipeline(req: PipelineRequest):
    """Full pipeline: generate styled speech, then convert to target voice.

    1. Qwen3-TTS generates audio from text + style_prompt.
    2. Chatterbox VC converts it to match target_audio_base64.
    3. Returns the final audio (and optionally the intermediate TTS audio).
    """
    if _qwen is None:
        raise HTTPException(500, "Qwen3-TTS not loaded.")
    if _vc is None:
        raise HTTPException(500, "Chatterbox VC not loaded.")

    # ── Stage 1: TTS ──────────────────────────────────────────────
    kwargs = {}
    for attr, key in [("tts_max_new_tokens", "max_new_tokens"),
                      ("tts_top_p", "top_p"),
                      ("tts_temperature", "temperature"),
                      ("tts_repetition_penalty", "repetition_penalty")]:
        val = getattr(req, attr)
        if val is not None:
            kwargs[key] = val

    try:
        wavs, tts_sr = _qwen.generate_voice_design(
            text=req.text,
            instruct=req.style_prompt,
            language=req.language,
            **kwargs,
        )
        tts_audio = wavs[0]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"TTS step failed: {e}")

    intermediate_b64 = audio_to_b64(tts_audio, tts_sr) if req.return_intermediate else None

    # ── Stage 2: Voice Conversion ─────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="pipeline_")
    try:
        # Write TTS output to disk (Chatterbox expects file paths)
        src_path = os.path.join(tmp_dir, "tts.wav")
        sf.write(src_path, tts_audio, tts_sr)

        # Write target reference to disk
        tgt_path = b64_to_wav_file(req.target_audio_base64, tmp_dir, "target")

        # Convert
        result = _vc.generate(audio=src_path, target_voice_path=tgt_path)
        wav = result.squeeze(0).cpu().numpy().astype(np.float32)

        resp = {
            "status": "success",
            "sample_rate": _vc.sr,
            "audio_base64": audio_to_b64(wav, _vc.sr),
        }
        if intermediate_b64:
            resp["intermediate_tts_audio_base64"] = intermediate_b64
            resp["intermediate_tts_sample_rate"] = tts_sr
        return resp

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"VC step failed: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/health")
async def health():
    """Check which models are loaded and whether CUDA is available."""
    return {
        "qwen_loaded": _qwen is not None,
        "chatterbox_loaded": _vc is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(_device),
    }


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Pipeline Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
