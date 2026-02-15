#!/usr/bin/env python3
"""
============================================================================
 STANDALONE SCRIPT — Voice Conversion with Chatterbox VC
============================================================================

Converts a source audio file to sound like a target speaker, using
Chatterbox VC by Resemble AI.

HOW IT WORKS
------------
Chatterbox VC is a zero-shot voice conversion model:

  1. The source audio is encoded into discrete S3 speech tokens by a
     neural speech tokenizer.  These tokens capture the phonetic content
     and prosody (rhythm, intonation) but discard the speaker identity.

  2. A speaker embedding is extracted from a short reference clip of the
     target speaker (up to 10 seconds used).

  3. A flow-matching decoder takes the S3 tokens + speaker embedding and
     generates a new waveform at 24 kHz that sounds like the target
     speaker saying the source content with the source prosody.

No fine-tuning or training is needed — just a few seconds of reference
audio from the target speaker.

USAGE
-----
  # Basic: convert source to target voice, save WAV
  python convert_voice.py --source input.wav --target ref.wav

  # Also generate an HTML comparison page
  python convert_voice.py --source input.wav --target ref.wav --html

  # Custom output path
  python convert_voice.py --source input.wav --target ref.wav -o converted.wav

  # Batch: convert multiple source files to the same target voice
  python convert_voice.py --source a.wav b.wav c.wav --target ref.wav --html
============================================================================
"""

import argparse
import base64
import io
import os
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch
from datetime import datetime
from pathlib import Path


# =====================================================================
# CHATTERBOX VC LOADER
# =====================================================================

def load_chatterbox(device="cuda"):
    """Load Chatterbox VC model.

    On first call the weights are downloaded from HuggingFace and cached
    locally.  Subsequent calls load from cache instantly.

    Args:
        device: "cuda", "cpu", or "mps"

    Returns:
        ChatterboxVC instance with .generate() and .sr attributes.
    """
    from chatterbox.vc import ChatterboxVC

    print(f"[INFO] Loading Chatterbox VC on {device} …")
    vc = ChatterboxVC.from_pretrained(device=device)
    print(f"[INFO] Ready (output sample rate: {vc.sr} Hz).")
    return vc


# =====================================================================
# CONVERSION
# =====================================================================

def convert(vc, source_path: str, target_path: str):
    """Convert source audio to the target speaker's voice.

    Args:
        vc:           Loaded ChatterboxVC instance
        source_path:  Path to the audio whose content we want to keep
        target_path:  Path to the reference audio of the target speaker

    Returns:
        (numpy_array, sample_rate) — the converted waveform
    """
    result = vc.generate(audio=source_path, target_voice_path=target_path)
    # result shape: (1, num_samples)
    wav = result.squeeze(0).cpu().numpy().astype(np.float32)
    return wav, vc.sr


# =====================================================================
# HTML REPORT
# =====================================================================

def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def audio_to_b64(arr, sr):
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def audio_mime(path):
    ext = Path(path).suffix.lower().lstrip(".")
    return {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(ext, "audio/wav")


def build_html(target_path, conversions):
    """Build an HTML page comparing source → converted for each file.

    Args:
        target_path:  Path to the target/reference audio
        conversions:  List of dicts with keys:
            source_path, output_path, source_b64, source_mime,
            result_b64, duration, elapsed
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    tgt_mime = audio_mime(target_path)
    tgt_b64 = file_to_b64(target_path)
    tgt_name = Path(target_path).name

    rows = ""
    for c in conversions:
        src_name = Path(c["source_path"]).name
        out_name = Path(c["output_path"]).name
        rows += f"""
        <div class="card">
          <h3>{src_name} &rarr; {out_name}</h3>
          <div class="row">
            <div class="col">
              <span class="lbl">Source (original voice)</span>
              <audio controls preload="none">
                <source src="data:{c['source_mime']};base64,{c['source_b64']}" type="{c['source_mime']}">
              </audio>
            </div>
            <div class="col">
              <span class="lbl">Converted (target voice)</span>
              <audio controls preload="none">
                <source src="data:audio/wav;base64,{c['result_b64']}" type="audio/wav">
              </audio>
            </div>
          </div>
          <p class="meta">{c['duration']:.1f}s &middot; converted in {c['elapsed']:.1f}s</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Conversion — Chatterbox VC</title>
<style>
:root{{--bg:#0c0c10;--s:#16161e;--c:#1c1c28;--t:#d8d8e0;--m:#777;--a:#e07b4a;--b:#2a2a38}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--t);
line-height:1.65;max-width:720px;margin:0 auto;padding:2.5rem 1.5rem}}
h1{{font-size:1.6rem;color:#fff}} h2{{font-size:1.2rem;color:var(--a);margin:2rem 0 .6rem}}
h3{{font-size:1rem;margin:0 0 .8rem;color:#ccc}}
.sub{{color:var(--m);font-size:.85rem;margin:.3rem 0 2rem}}
.intro{{background:var(--s);border:1px solid var(--b);border-radius:12px;
padding:1.4rem 1.6rem;margin-bottom:2rem}}
.intro p{{margin-bottom:.7rem;font-size:.92rem}} .intro p:last-child{{margin-bottom:0}}
.intro strong{{color:var(--a)}}
.ref{{background:var(--s);border:1px solid var(--b);border-radius:10px;
padding:1rem 1.3rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}}
.ref .lbl{{font-size:.8rem;color:var(--m);text-transform:uppercase;letter-spacing:.6px}}
.ref audio{{flex:1;min-width:200px;height:36px}}
.card{{background:var(--c);border-radius:10px;padding:1.3rem 1.5rem;margin-bottom:1.2rem}}
.row{{display:flex;gap:1.5rem;flex-wrap:wrap}}
.col{{flex:1;min-width:240px}}
.lbl{{display:block;font-size:.8rem;color:var(--m);text-transform:uppercase;
letter-spacing:.5px;margin-bottom:.3rem}}
audio{{width:100%;height:40px;border-radius:8px}}
.meta{{font-size:.72rem;color:#555;margin-top:.5rem;text-align:right}}
footer{{text-align:center;color:#444;font-size:.72rem;margin-top:2.5rem;
border-top:1px solid #1a1a1a;padding-top:1rem}}
</style></head><body>
<h1>Voice Conversion — Chatterbox VC</h1>
<p class="sub">Generated {ts}</p>
<div class="intro">
<p><strong>Chatterbox VC</strong> by Resemble AI is a zero-shot voice conversion model.</p>
<p>It encodes the source audio into discrete <strong>S3 speech tokens</strong> (capturing
content and prosody), extracts a <strong>speaker embedding</strong> from a short reference
clip, then decodes a new waveform via a <strong>flow-matching model</strong> that sounds
like the target speaker saying the source content.</p>
<p>No training or fine-tuning needed — just a few seconds of reference audio.</p>
</div>
<div class="ref">
<span class="lbl">Target voice ({tgt_name})</span>
<audio controls preload="none"><source src="data:{tgt_mime};base64,{tgt_b64}" type="{tgt_mime}"></audio>
</div>
{rows}
<footer>Chatterbox VC (s3gen) &middot; 24 kHz &middot; {ts}</footer>
</body></html>"""


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert audio to a target speaker's voice using Chatterbox VC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", nargs="+", required=True,
                        help="Source audio file(s) to convert.")
    parser.add_argument("--target", required=True,
                        help="Reference audio of the target speaker.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (WAV). For multiple sources, this is treated as a directory.")
    parser.add_argument("--html", action="store_true",
                        help="Generate an HTML comparison report.")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (cuda, cpu, mps).")
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    if not target_path.exists():
        print(f"[ERROR] Target file not found: {target_path}")
        sys.exit(1)

    source_paths = [Path(s).resolve() for s in args.source]
    for sp in source_paths:
        if not sp.exists():
            print(f"[ERROR] Source file not found: {sp}")
            sys.exit(1)

    # Determine output directory
    if args.output and len(source_paths) > 1:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    elif args.output and len(source_paths) == 1:
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("output_vc")
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    vc = load_chatterbox(args.device)

    # ── Convert each source ───────────────────────────────────────
    conversions = []

    for i, src in enumerate(source_paths, 1):
        print(f"\n[{i}/{len(source_paths)}] {src.name} → {target_path.name}")

        t0 = time.time()
        try:
            wav, sr = convert(vc, str(src), str(target_path))
        except Exception:
            print("  [ERROR]")
            traceback.print_exc()
            continue
        elapsed = time.time() - t0
        duration = len(wav) / sr
        print(f"  {duration:.1f}s audio in {elapsed:.1f}s")

        # Determine output filename
        if args.output and len(source_paths) == 1:
            out_path = Path(args.output)
        else:
            out_path = out_dir / f"{src.stem}_as_{target_path.stem}.wav"

        sf.write(str(out_path), wav, sr)
        print(f"  Saved: {out_path}")

        conversions.append({
            "source_path": str(src),
            "output_path": str(out_path),
            "source_b64": file_to_b64(str(src)),
            "source_mime": audio_mime(str(src)),
            "result_b64": audio_to_b64(wav, sr),
            "duration": duration,
            "elapsed": elapsed,
        })

    # ── HTML report ───────────────────────────────────────────────
    if args.html and conversions:
        html = build_html(str(target_path), conversions)
        html_path = out_dir / "vc_report.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"\nHTML report: {html_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
