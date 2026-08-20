"""Synthesise Kabyle speech using Matoub-TTS.

This tool runs inference over any trained Matoub Stage 1 or Stage 2 checkpoint
(such as Epoch 4 or the final Epoch 6 model) and outputs a 24 kHz WAV file.

Usage:
    # Synthesise a sentence with the Epoch 4 checkpoint:
    python3 -m tools.infer_matoub --text "Azul fell-awen, amek i telliḍ taṣebḥit-a?"

    # Custom text and output path:
    python3 -m tools.infer_matoub --text "Aɣbalu d anegraw ameqqran." --out sample.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from agbalu.tts import kokoro
from agbalu.tts.g2p import phonemize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="Azul fell-awen, amek i telliḍ taṣebḥit-a?",
        help="Kabyle text to synthesise into speech",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage2",
        choices=["stage1", "stage2"],
        help="Stage of the checkpoint (default: stage2)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="/data/tts/matoub/restored/logs/kab_male/epoch_2nd_00003.pth",
        help="Remote path to checkpoint on Modal volume (default: epoch_2nd_00003.pth)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Diffusion sampler steps (default: 10)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Acoustic style blend: 0.0=100% reference timbre (default: 0.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Prosodic style blend: 0.0=100% reference prosody (default: 0.0)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default="kab_male",
        choices=["kab_male", "kab_female"],
        help="Voice to synthesise: kab_male or kab_female (default: kab_male)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("artifacts/matoub/matoub_sample.wav"),
        help="Local output path for the generated WAV file",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        default=True,
        help="Play audio automatically on macOS after downloading (default: True)",
    )
    args = parser.parse_args(argv)

    ipa = kokoro.fold(phonemize(args.text))
    print(f'text:       "{args.text}"')
    print(f"ipa:        /{ipa}/")
    print(f"checkpoint: {args.checkpoint}")
    print(f"voice:      {args.voice}")
    print(
        f"stage:      {args.stage} "
        f"(steps={args.steps}, scale={args.scale}, alpha={args.alpha}, beta={args.beta})"
    )

    filename = args.out.name
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("synthesising on a Modal GPU...")
    infer_cmd = [
        "modal",
        "run",
        "-m",
        "modal_app.matoub::matoub_infer",
        "--stage",
        args.stage,
        "--voice",
        args.voice,
        "--checkpoint",
        args.checkpoint,
        "--text",
        args.text,
        "--diffusion-steps",
        str(args.steps),
        "--embedding-scale",
        str(args.scale),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--output-filename",
        filename,
    ]
    res = subprocess.run(infer_cmd, check=False)  # noqa: S603
    if res.returncode != 0:
        print("synthesis failed; see the logs above", file=sys.stderr)
        return res.returncode

    print(f"downloading to {args.out}...")
    remote_wav = f"tts/matoub/restored/samples/{filename}"
    get_cmd = ["modal", "volume", "get", "--force", "agbalu-data", remote_wav, str(out_dir)]
    res = subprocess.run(get_cmd, check=False)  # noqa: S603
    if res.returncode != 0:
        fallback_wav = f"tts/matoub/samples/{filename}"
        get_cmd = ["modal", "volume", "get", "--force", "agbalu-data", fallback_wav, str(out_dir)]
        res = subprocess.run(get_cmd, check=False)  # noqa: S603
        if res.returncode != 0:
            print(f"could not download {remote_wav} from the volume", file=sys.stderr)
            return res.returncode

    print(f"saved {args.out}")

    if args.play and sys.platform == "darwin":
        subprocess.run(["/usr/bin/afplay", str(args.out)], check=False)  # noqa: S603

    return 0


if __name__ == "__main__":
    sys.exit(main())
