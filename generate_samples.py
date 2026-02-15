#!/usr/bin/env python3
"""
============================================================================
 STANDALONE SCRIPT — Generate Emotionally Styled Speech in a Target Voice
============================================================================

This script generates multiple audio samples by combining:

  1. Qwen3-TTS Voice Design — generates expressive speech from text +
     a style/emotion instruction.  Outputs a random voice identity each
     time, but the emotion, pace, and energy match the instruction.

  2. Chatterbox VC — converts that TTS output into the target speaker's
     voice.  Only the timbre/identity changes; the emotion and prosody
     from Stage 1 are preserved.

Both models are loaded directly on the local GPU (~5 GB total VRAM).
No server required.

The script produces:
  - One WAV file per emotion
  - A self-contained HTML report with embedded audio players

USAGE
-----
  # Use the built-in example emotions:
  python generate_samples.py --ref chris-ref.mp3

  # Custom output directory:
  python generate_samples.py --ref chris-ref.mp3 --output my_output/

  # Specify speaker name (shown in the report):
  python generate_samples.py --ref chris-ref.mp3 --name "Chris"

CUSTOMISATION
-------------
Edit the SAMPLES list below to change emotions, texts, or style prompts.
The "voice_hint" prefix is prepended to every style_prompt to steer the
TTS toward the right gender/pitch range (important when converting to a
specific speaker).
============================================================================
"""

import argparse
import base64
import io
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf
import torch
from datetime import datetime
from pathlib import Path

# =====================================================================
# ██  CONFIGURATION — edit these to customise  ██
# =====================================================================

# Prepended to every style_prompt so the TTS generates a voice that
# roughly matches the target speaker's gender and pitch range.
# Change this if your reference speaker is female, very deep, etc.
VOICE_HINT = "Medium-pitched adult male voice. "

# Each sample: a label (for the report), a style_prompt (emotion/delivery
# instruction for Qwen), and the text to speak.
SAMPLES = [
    {
        "label": "Warm & Grateful",
        "color": "#2d6a4f",
        "icon": "🌿",
        "style_prompt": (
            "Speak with deep warmth, gratitude, and gentle affection. "
            "A soft, heartfelt voice with a slow, contemplative pace — "
            "savoring every word with genuine love and tenderness."
        ),
        "text": (
            "You know what I love most? Those long walks through the forest "
            "after the rain, when the air smells like pine and wet earth, "
            "and the sunlight breaks through the canopy in golden beams. "
            "It fills my heart with so much gratitude just to be alive "
            "in moments like these."
        ),
    },
    {
        "label": "Terrified & Uneasy",
        "color": "#4a235a",
        "icon": "👻",
        "style_prompt": (
            "Speak with genuine fear and creeping dread. The voice should tremble, "
            "whispering at times, with an unsteady rhythm. Nervous, shaky breathing "
            "between phrases — as if every small sound could be something terrible."
        ),
        "text": (
            "Okay... I'm in the basement now. The lights keep flickering "
            "and I swear I just heard footsteps behind me, but when I turned "
            "around — nothing. Just this long, dark hallway. My skin is crawling. "
            "Every creak in this old building sounds like something is about to "
            "grab me. I need to get out of here."
        ),
    },
    {
        "label": "Furious & Ranting",
        "color": "#922b21",
        "icon": "🤬",
        "style_prompt": (
            "Speak with intense, boiling anger and frustrated impatience. "
            "Loud, sharp, and exasperated — someone who has completely lost "
            "their patience. Fast pace, rising intonation, seething pauses."
        ),
        "text": (
            "Are you kidding me?! This stupid computer just crashed AGAIN! "
            "I've been debugging this cursed machine for five hours straight "
            "and it just ate my entire session — gone, all of it! Every single "
            "time I trust this thing it finds a new way to betray me. "
            "I am SO done with this garbage!"
        ),
    },
    {
        "label": "Absolute Elation",
        "color": "#b7950b",
        "icon": "🎉",
        "style_prompt": (
            "Speak with pure, overflowing joy and astonished amazement. "
            "Bright, breathless, almost laughing — someone who just received "
            "the best news of their life. High energy, fast, disbelieving happiness."
        ),
        "text": (
            "Oh my God — wait, are you serious?! I got in?! I actually got "
            "the scholarship?! I can't — I literally cannot believe this "
            "is happening right now! This is the single greatest day of my "
            "entire life — I need to call my mom, I need to tell everyone! "
            "I'm shaking, I'm actually shaking!"
        ),
    },
    {
        "label": "Vulnerable & Heartbroken",
        "color": "#5b6e8a",
        "icon": "💔",
        "style_prompt": (
            "Speak with deep, quiet vulnerability and fragile sadness. "
            "A voice on the edge of breaking — raw, genuine, intimate. "
            "Slow, with long pauses, as if each word costs real effort."
        ),
        "text": (
            "I just... I really thought you were different. I told you "
            "things I've never told anyone, and you just... you used them. "
            "I keep replaying it in my head, trying to understand how "
            "someone I trusted so completely could just walk away like "
            "none of it mattered. Maybe it never did. Maybe I was the only "
            "one who thought we had something real."
        ),
    },
]


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_qwen(device="cuda"):
    """Load Qwen3-TTS Voice Design model.

    Checks for a local copy in models/qwen_voice_design first,
    otherwise downloads from HuggingFace.
    """
    from qwen_tts import Qwen3TTSModel

    local = Path(__file__).resolve().parent / "models" / "qwen_voice_design"
    source = str(local) if (local / "model.safetensors").exists() \
        else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    print(f"[INFO] Loading Qwen3-TTS from {source} …")
    model = Qwen3TTSModel.from_pretrained(source, device_map=device, dtype=torch.bfloat16)
    print("[INFO] Qwen3-TTS ready.")
    return model


def load_chatterbox(device="cuda"):
    """Load Chatterbox VC model.

    Weights are cached by huggingface_hub after the first download.
    """
    from chatterbox.vc import ChatterboxVC

    print(f"[INFO] Loading Chatterbox VC on {device} …")
    vc = ChatterboxVC.from_pretrained(device=device)
    print(f"[INFO] Chatterbox VC ready (sr={vc.sr}).")
    return vc


# =====================================================================
# GENERATION
# =====================================================================

def generate_one(qwen, vc, text, style_prompt, ref_path, language="English"):
    """Run the full TTS → VC pipeline for a single sample.

    Args:
        qwen:         Loaded Qwen3TTSModel
        vc:           Loaded ChatterboxVC
        text:         Text to speak
        style_prompt: Emotion/style instruction (VOICE_HINT is prepended)
        ref_path:     Path to the target speaker's reference audio
        language:     Language code for Qwen

    Returns:
        (vc_audio_np, vc_sr, tts_time, vc_time)
    """
    full_prompt = VOICE_HINT + style_prompt

    # ── Stage 1: TTS ──────────────────────────────────────────────
    t0 = time.time()
    wavs, tts_sr = qwen.generate_voice_design(
        text=text, instruct=full_prompt, language=language,
    )
    tts_audio = wavs[0]
    tts_time = time.time() - t0

    # Write TTS to a temp file (Chatterbox expects a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, tts_audio, tts_sr)
        tts_path = f.name

    # ── Stage 2: Voice Conversion ─────────────────────────────────
    t1 = time.time()
    result = vc.generate(audio=tts_path, target_voice_path=str(ref_path))
    vc_time = time.time() - t1

    os.unlink(tts_path)

    vc_audio = result.squeeze(0).cpu().numpy().astype(np.float32)
    return vc_audio, vc.sr, tts_time, vc_time


# =====================================================================
# HTML BUILDER
# =====================================================================

def audio_to_b64(arr, sr):
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def audio_mime(path):
    ext = Path(path).suffix.lower().lstrip(".")
    return {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(ext, "audio/wav")


def build_html(speaker_name, ref_path, results):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ref_mime = audio_mime(ref_path)
    ref_b64 = file_to_b64(ref_path)

    cards = ""
    for r in results:
        cards += f"""
        <div class="card" style="border-left:4px solid {r['color']}">
          <div class="hdr"><span class="icon">{r['icon']}</span><h3>{r['label']}</h3></div>
          <blockquote>{r['text']}</blockquote>
          <p class="note"><b>Style:</b> {VOICE_HINT}{r['style_prompt']}</p>
          <audio controls preload="none">
            <source src="data:audio/wav;base64,{r['b64']}" type="audio/wav">
          </audio>
          <p class="meta">{r['dur']:.1f}s &middot; TTS {r['tts_t']:.1f}s + VC {r['vc_t']:.1f}s</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Samples — {speaker_name}</title>
<style>
:root{{--bg:#0c0c10;--s:#16161e;--c:#1c1c28;--t:#d8d8e0;--m:#777;--a:#e07b4a;--b:#2a2a38}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--t);
line-height:1.65;max-width:720px;margin:0 auto;padding:2.5rem 1.5rem}}
h1{{font-size:1.6rem;color:#fff}} h2{{font-size:1.2rem;color:var(--a);margin:2rem 0 .6rem}}
h3{{font-size:1.05rem;margin:0}} .sub{{color:var(--m);font-size:.85rem;margin:.3rem 0 2rem}}
.intro{{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:2.5rem}}
.intro p{{margin-bottom:.7rem;font-size:.92rem}} .intro p:last-child{{margin-bottom:0}}
.intro strong{{color:var(--a)}}
.ref{{background:var(--s);border:1px solid var(--b);border-radius:10px;padding:1rem 1.3rem;
margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}}
.ref .lbl{{font-size:.8rem;color:var(--m);text-transform:uppercase;letter-spacing:.6px}}
.ref audio{{flex:1;min-width:200px;height:36px}}
.card{{background:var(--c);border-radius:10px;padding:1.3rem 1.5rem;margin-bottom:1.2rem}}
.hdr{{display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem}}
.icon{{font-size:1.3rem}}
blockquote{{font-style:italic;color:#bbb;background:#111118;padding:.9rem 1.1rem;
border-radius:8px;margin-bottom:.7rem}}
.note{{font-size:.82rem;color:var(--m);margin-bottom:.8rem}}
audio{{width:100%;height:40px;border-radius:8px}}
.meta{{font-size:.72rem;color:#555;margin-top:.5rem;text-align:right}}
footer{{text-align:center;color:#444;font-size:.72rem;margin-top:2.5rem;
border-top:1px solid #1a1a1a;padding-top:1rem}}
</style></head><body>
<h1>Voice Samples — {speaker_name}</h1>
<p class="sub">Generated {ts}</p>
<div class="intro">
<p>This report was generated using a <strong>two-stage AI voice pipeline</strong>.</p>
<p><strong>Stage 1 — Qwen3-TTS Voice Design</strong> generates expressive speech from text
and a style instruction. The emotion, pace, and energy are controlled, but the voice
identity is random.</p>
<p><strong>Stage 2 — Chatterbox VC</strong> converts that audio to the target speaker's
voice using S3 speech tokens and a flow-matching decoder conditioned on a speaker
embedding from a short reference clip.</p>
</div>
<div class="ref">
<span class="lbl">Reference clip</span>
<audio controls preload="none"><source src="data:{ref_mime};base64,{ref_b64}" type="{ref_mime}"></audio>
</div>
{cards}
<footer>Qwen3-TTS 1.7B VoiceDesign + Chatterbox VC &middot; {ts}</footer>
</body></html>"""


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate emotionally styled speech samples in a target voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ref", required=True, help="Path to target speaker reference audio.")
    parser.add_argument("--name", default=None, help="Speaker name (used in report title).")
    parser.add_argument("--output", default="output", help="Output directory.")
    parser.add_argument("--language", default="English", help="Language code for TTS.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    args = parser.parse_args()

    ref_path = Path(args.ref).resolve()
    if not ref_path.exists():
        print(f"[ERROR] Reference audio not found: {ref_path}")
        sys.exit(1)

    speaker_name = args.name or ref_path.stem
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────
    qwen = load_qwen(args.device)
    vc = load_chatterbox(args.device)

    # ── Generate samples ──────────────────────────────────────────
    results = []
    total_t0 = time.time()

    for i, s in enumerate(SAMPLES, 1):
        print(f"\n[{i}/{len(SAMPLES)}] {s['icon']}  {s['label']}")
        try:
            audio, sr, tts_t, vc_t = generate_one(
                qwen, vc, s["text"], s["style_prompt"], ref_path, args.language,
            )
            dur = len(audio) / sr
            print(f"  {dur:.1f}s audio  (TTS {tts_t:.1f}s + VC {vc_t:.1f}s)")

            # Save wav
            safe = s["label"].lower().replace(" ", "_").replace("&", "and")
            wav_path = out_dir / f"{speaker_name.lower()}_{safe}.wav"
            sf.write(str(wav_path), audio, sr)

            results.append({
                **s,
                "b64": audio_to_b64(audio, sr),
                "dur": dur,
                "tts_t": tts_t,
                "vc_t": vc_t,
            })
        except Exception:
            print(f"  [ERROR]")
            traceback.print_exc()

    print(f"\nAll done in {time.time() - total_t0:.0f}s.")

    # ── Build HTML ────────────────────────────────────────────────
    html = build_html(speaker_name, str(ref_path), results)
    report = out_dir / "report.html"
    report.write_text(html, encoding="utf-8")
    print(f"Report: {report}")
    print(f"WAVs:   {out_dir}/")


if __name__ == "__main__":
    main()
