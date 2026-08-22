"""Command-line interface for Feraoun-Pro document OCR."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agbalu.ocr.adlis_gold import harvest_real_line_crops
from agbalu.ocr.backgrounds_pro import BackgroundPool
from agbalu.ocr.model_pro import HybridCTCRecognizer
from agbalu.ocr.release_pro import RELEASE_CONFIG, write_config
from agbalu.ocr.synthetic_pro import LineRenderer
from agbalu.ocr.vocabulary_pro import decode_ctc

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("agbalu.ocr.cli_pro")

DEFAULT_CORPUS = Path("data/processed/text/agbalu-text-v1.jsonl")
DEFAULT_CHECKPOINT = Path("artifacts/runs/feraoun-pro/best.pt")


def handle_generate(args: argparse.Namespace) -> None:
    """Render variable-width synthetic line images, degraded, for inspecting the corpus."""
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    if not input_path.exists():
        message = f"no corpus at {input_path}"
        raise FileNotFoundError(message)

    lines: list[str] = []
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                    text = str(obj.get("text", "")).strip()
                    if text:
                        lines.append(text)
                except json.JSONDecodeError:
                    continue
            else:
                lines.append(s)
            if args.lines and len(lines) >= args.lines:
                break

    renderer = LineRenderer()
    log.info("Rendering %d Feraoun-Pro synthetic lines into %s...", len(lines), out_dir)

    for idx, text_line in enumerate(lines):
        img, final_text = renderer.render_line(text_line, apply_augmentation=True)
        img_path = out_dir / f"line_{idx:06d}.png"
        img.save(img_path)
        (out_dir / f"line_{idx:06d}.txt").write_text(final_text, encoding="utf-8")

    log.info("Finished generating %d lines.", len(lines))


def handle_harvest_backgrounds(args: argparse.Namespace) -> None:
    """Harvest blank margin tiles from Adlis scan images."""
    scan_dir = Path(args.scan_dir)
    pool = BackgroundPool.harvest_from_directory(scan_dir=scan_dir, max_tiles=args.max_tiles)
    log.info("Harvested %d real paper background tiles.", len(pool.tiles))


def handle_harvest_gold(args: argparse.Namespace) -> None:
    """Sample real line crops from Adlis scans for gold benchmark annotation."""
    scan_dir = Path(args.scan_dir)
    out_dir = Path(args.output)
    crops = harvest_real_line_crops(scan_dir=scan_dir, output_dir=out_dir, num_samples=args.samples)
    log.info("Extracted %d line crops to %s for gold annotation.", len(crops), out_dir)


def prepare_image(img_path: str | Path) -> torch.Tensor:
    """Load image and prepare normalized tensor of height 64 preserving aspect ratio."""
    img = Image.open(img_path).convert("RGB")
    h_orig, w_orig = img.height, img.width
    scale = 64.0 / max(1, h_orig)
    new_w = max(16, round(w_orig * scale))
    if new_w % 4 != 0:
        new_w += 4 - (new_w % 4)
    new_w = min(1024, new_w)

    resized = img.resize((new_w, 64), Image.Resampling.BILINEAR)
    arr = (np.array(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def handle_infer(args: argparse.Namespace) -> None:
    """Run inference over a single line crop."""
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        message = f"no checkpoint at {ckpt_path}; pull it with `make modal-ocr-pro-pull`"
        raise FileNotFoundError(message)

    img_path = Path(args.image)
    if not img_path.is_file():
        message = f"no image at {img_path}"
        raise FileNotFoundError(message)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Loading Feraoun-Pro checkpoint from %s on %s...", ckpt_path, device)

    from agbalu.model.checkpoint import BEST, load

    model = HybridCTCRecognizer()
    load(ckpt_path.parent, model, name=BEST if ckpt_path.name == "best.pt" else ckpt_path.stem)
    model.to(device)
    model.eval()

    tensor = prepare_image(img_path).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor)
        pred_ids = torch.argmax(out.ctc_logits, dim=-1)[0].cpu().tolist()
        pred_text = decode_ctc(pred_ids)

    log.info("================ TRANSCRIPTION ================")
    log.info("PRED: %s", pred_text)
    log.info("===============================================")
    print(pred_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Render synthetic line images")
    gen_parser.add_argument("--input", default=str(DEFAULT_CORPUS), help="Input text JSONL")
    gen_parser.add_argument(
        "--output", default="artifacts/ocr_pro_samples", help="Output directory"
    )
    gen_parser.add_argument("--lines", type=int, default=100, help="Number of lines to render")

    # Harvest backgrounds
    bg_parser = subparsers.add_parser(
        "harvest-backgrounds", help="Harvest blank paper tiles from scans"
    )
    bg_parser.add_argument(
        "--scan-dir", default="data/raw/hf.boffire.adlis-pdfs-ocr-kab", help="Scan directory"
    )
    bg_parser.add_argument("--max-tiles", type=int, default=200, help="Max tiles to extract")

    # Harvest gold lines
    gold_parser = subparsers.add_parser(
        "harvest-gold", help="Harvest real line crops for gold benchmark"
    )
    gold_parser.add_argument(
        "--scan-dir", default="data/raw/hf.boffire.adlis-pdfs-ocr-kab", help="Scan directory"
    )
    gold_parser.add_argument(
        "--output", default="data/processed/ocr/adlis_real_lines", help="Output directory"
    )
    gold_parser.add_argument("--samples", type=int, default=500, help="Number of crops to sample")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Transcribe a line crop image")
    infer_parser.add_argument("--image", required=True, help="Path to line crop image")
    infer_parser.add_argument(
        "--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to best.pt"
    )
    infer_parser.add_argument("--device", default="", help="Device override (cpu/cuda)")

    release_parser = subparsers.add_parser(
        "release", help="write config.json for the published repo"
    )
    release_parser.add_argument("--out", type=Path, default=RELEASE_CONFIG)

    args = parser.parse_args(argv)

    if args.command == "generate":
        handle_generate(args)
    elif args.command == "harvest-backgrounds":
        handle_harvest_backgrounds(args)
    elif args.command == "harvest-gold":
        handle_harvest_gold(args)
    elif args.command == "infer":
        handle_infer(args)
    elif args.command == "release":
        print(write_config(args.out))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
