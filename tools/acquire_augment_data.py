#!/usr/bin/env python3
"""AƔBALU: Download and extract MUSAN and BUT ReverbDB for MCT speech augmentation.

Runs on the HPC pod:
- Downloads MUSAN noise dataset (~11 GB) from OpenSLR / mirrors.
- Downloads BUT ReverbDB RIR dataset (~8.7 GB) from Brno UT:
  http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz
- Extracts both into data/raw/speech/augment/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

MIN_ARCHIVE_BYTES: Final[int] = 1000

MUSAN_URLS: Sequence[str] = (
    "http://www.openslr.org/resources/17/musan.tar.gz",
    "https://huggingface.co/datasets/huseinzol05/musan-mirror/resolve/main/musan.tar.gz",
    "https://www.openslr.org/resources/17/musan.tar.gz",
)
REVERB_URLS: Sequence[str] = (
    "http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
)
DEFAULT_AUGMENT_DIR = Path("data/raw/speech/augment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_AUGMENT_DIR,
        help="Target directory for augmentation data",
    )
    parser.add_argument(
        "--skip-musan",
        action="store_true",
        help="Skip MUSAN noise download",
    )
    parser.add_argument(
        "--skip-reverb",
        action="store_true",
        help="Skip BUT ReverbDB download",
    )
    return parser.parse_args()


def download_and_extract(
    urls: Sequence[str],
    dest_dir: Path,
    archive_name: str,
    *,
    extracted_check: Path | None = None,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / archive_name

    if extracted_check and extracted_check.exists():
        sys.stdout.write(f"--> Assets already extracted at {extracted_check}, skipping.\n")
        return

    if archive_path.is_file() and archive_path.stat().st_size == 0:
        archive_path.unlink()

    if not archive_path.is_file() or archive_path.stat().st_size < MIN_ARCHIVE_BYTES:
        download_succeeded = False
        for url in urls:
            sys.stdout.write(f"--> Attempting download of {archive_name} from {url}...\n")
            cmd_curl = [
                "curl",
                "-L",
                "-C",
                "-",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "--retry-connrefused",
                "--connect-timeout",
                "30",
                "-o",
                str(archive_path),
                url,
            ]
            res = subprocess.run(cmd_curl, check=False)
            if (
                res.returncode == 0
                and archive_path.is_file()
                and archive_path.stat().st_size > MIN_ARCHIVE_BYTES
            ):
                download_succeeded = True
                break
            sys.stderr.write(f"Warning: download failed from {url} (exit code {res.returncode})\n")
            if archive_path.is_file() and archive_path.stat().st_size < MIN_ARCHIVE_BYTES:
                archive_path.unlink()

        if not download_succeeded:
            msg = f"Failed to download {archive_name} from all available mirror URLs."
            raise RuntimeError(msg)
    else:
        sys.stdout.write(
            f"--> Archive already present ({archive_path.stat().st_size:,} bytes): {archive_path}\n"
        )

    sys.stdout.write(f"--> Extracting {archive_name} into {dest_dir}...\n")
    cmd_tar = ["tar", "-xzf", str(archive_path), "-C", str(dest_dir)]
    subprocess.run(cmd_tar, check=True)
    sys.stdout.write(f"--> Extraction complete for {archive_name}.\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_musan:
        musan_dir = args.out_dir / "musan"
        download_and_extract(
            MUSAN_URLS,
            musan_dir,
            "musan.tar.gz",
            extracted_check=musan_dir / "musan",
        )

    if not args.skip_reverb:
        reverb_dir = args.out_dir / "reverb"
        download_and_extract(
            REVERB_URLS,
            reverb_dir,
            "BUT_ReverbDB_RIR.tgz",
            extracted_check=reverb_dir / "RIR_19_06",
        )

    sys.stdout.write("\n--> Multi-condition augmentation assets ready.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
