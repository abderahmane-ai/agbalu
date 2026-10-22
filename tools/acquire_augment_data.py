#!/usr/bin/env python3
"""AƔBALU: High-speed multi-connection download and extraction of MUSAN and BUT ReverbDB.

Accelerates download speeds from OpenSLR and Brno UT by opening multiple parallel
HTTP streams (via aria2c or native multi-threaded Range requests with os.pwrite),
bypassing single-stream bandwidth throttling.

Runs on the HPC pod:
- Downloads MUSAN noise dataset (~11 GB) from OpenSLR
- Downloads BUT ReverbDB RIR dataset (~8.7 GB) from Brno UT
- Extracts both into data/raw/speech/augment/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

MUSAN_EXPECTED_BYTES: Final[int] = 11086114085
REVERB_EXPECTED_BYTES: Final[int] = 9308593693
DEFAULT_CHUNK_SIZE: Final[int] = 8 * 1024 * 1024  # 8 MB per chunk
DEFAULT_WORKERS: Final[int] = 16
PROGRESS_INTERVAL_SECS: Final[float] = 2.0
MAX_CHUNK_RETRIES: Final[int] = 5

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
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent connection streams (default: 16)",
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


def is_musan_extracted(musan_dir: Path) -> bool:
    """Check if all three MUSAN categories are already extracted."""
    target = musan_dir / "musan"
    return (
        target.is_dir()
        and (target / "noise").is_dir()
        and (target / "speech").is_dir()
        and (target / "music").is_dir()
    )


def is_reverb_extracted(reverb_dir: Path) -> bool:
    """Check if BUT ReverbDB RIR wav files are already extracted."""
    target = reverb_dir / "RIR_19_06"
    return target.is_dir() and any(target.rglob("*.wav"))


def is_archive_complete(archive_path: Path, expected_size: int) -> bool:
    """Check if archive file exists on disk and matches expected exact byte count."""
    return archive_path.is_file() and archive_path.stat().st_size == expected_size


def get_remote_file_size(url: str) -> int:
    """Determine the file size in bytes using Range request."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            return int(content_range.split("/")[-1])
        cl = resp.headers.get("Content-Length")
        if cl:
            return int(cl)
    return 0


def _fetch_chunk_bytes(url: str, start: int, end: int) -> bytes:
    """Fetch a single byte-range slice with retry backoff."""
    expected = end - start + 1
    last_err: Exception | None = None
    for attempt in range(MAX_CHUNK_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=25) as resp:  # noqa: S310
                data = bytes(resp.read())
            if len(data) == expected:
                return data
            last_err = OSError(f"Size mismatch: got {len(data)}, expected {expected}")
        except Exception as exc:
            last_err = exc
        time.sleep(1.0 + attempt)
    err_msg = f"Failed chunk {start}-{end} after retries: {last_err}"
    raise RuntimeError(err_msg)


def _log_download_progress(
    completed: set[int],
    total_size: int,
    chunk_size: int,
    session_bytes: int,
    t_start: float,
) -> None:
    elapsed = max(0.1, time.time() - t_start)
    speed_mb = (session_bytes / elapsed) / (1024 * 1024)
    done_bytes = sum(min(chunk_size, total_size - i * chunk_size) for i in completed)
    pct = (done_bytes / total_size) * 100.0
    rem = max(0, total_size - done_bytes)
    eta_s = int(rem / max(1.0, speed_mb * 1024 * 1024))
    eta_str = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"
    cur_gb = done_bytes / (1024**3)
    tot_gb = total_size / (1024**3)
    sys.stdout.write(
        f"  [Multi-Stream] {cur_gb:.2f} / {tot_gb:.2f} GB ({pct:5.1f}%) "
        f"- {speed_mb:.2f} MB/s - ETA: {eta_str}\n"
    )
    sys.stdout.flush()


def native_multi_stream_download(
    url: str,
    output_path: Path,
    *,
    num_workers: int = DEFAULT_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Download a file using multiple concurrent HTTP Range streams with os.pwrite."""
    total_size = get_remote_file_size(url)
    if total_size <= 0:
        msg = f"Could not determine total size for {url}"
        raise RuntimeError(msg)

    progress_file = output_path.with_suffix(output_path.suffix + ".progress.json")
    completed: set[int] = set()

    fd = os.open(str(output_path), os.O_RDWR | os.O_CREAT)
    if output_path.stat().st_size < total_size:
        os.ftruncate(fd, total_size)

    if progress_file.is_file():
        try:
            completed = set(json.loads(progress_file.read_text(encoding="utf-8")))
        except Exception:
            completed = set()

    num_chunks = math.ceil(total_size / chunk_size)
    remaining = [i for i in range(num_chunks) if i not in completed]

    mb_each = chunk_size // (1024 * 1024)
    sys.stdout.write(
        f"--> Native multi-stream downloader ({num_workers} streams):\n"
        f"    Target: {output_path.name} ({total_size / (1024**3):.2f} GB)\n"
        f"    Chunks: {len(completed)} done, {len(remaining)} left ({mb_each} MB each)\n"
    )

    lock = threading.Lock()
    session_bytes = 0
    t_start = time.time()
    last_print = t_start

    def download_worker(idx: int) -> None:
        nonlocal session_bytes, last_print
        start = idx * chunk_size
        end = min(total_size - 1, (idx + 1) * chunk_size - 1)
        data = _fetch_chunk_bytes(url, start, end)
        os.pwrite(fd, data, start)

        with lock:
            completed.add(idx)
            session_bytes += len(data)
            now = time.time()
            if now - last_print >= PROGRESS_INTERVAL_SECS:
                last_print = now
                _log_download_progress(completed, total_size, chunk_size, session_bytes, t_start)
                progress_file.write_text(json.dumps(list(completed)), encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(download_worker, remaining))

    os.close(fd)
    if progress_file.is_file():
        progress_file.unlink()
    size_str = f"{output_path.stat().st_size:,}"
    sys.stdout.write(f"--> Multi-stream download complete: {output_path} ({size_str} bytes)\n")


def _try_aria2c_download(
    url: str, dest_dir: Path, archive_name: str, workers: int, expected_size: int
) -> bool:
    """Attempt download using aria2c if available on system."""
    cmd = shutil.which("aria2c")
    if not cmd:
        return False
    sys.stdout.write(f"--> Accelerating download with aria2c ({workers} streams) from {url}...\n")
    cmd_args = [
        cmd,
        "-x",
        str(workers),
        "-s",
        str(workers),
        "-j",
        str(workers),
        "-k",
        "1M",
        "--file-allocation=none",
        "--continue=true",
        "--dir",
        str(dest_dir),
        "--out",
        archive_name,
        url,
    ]
    res = subprocess.run(cmd_args, check=False)
    archive_path = dest_dir / archive_name
    return res.returncode == 0 and is_archive_complete(archive_path, expected_size)


def download_and_extract(
    urls: Sequence[str],
    archive_path: Path,
    *,
    expected_size: int,
    is_extracted_fn: Callable[[Path], bool] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> None:
    dest_dir = archive_path.parent
    archive_name = archive_path.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_extracted_fn and is_extracted_fn(dest_dir):
        sys.stdout.write(f"--> Assets already fully extracted in {dest_dir}, skipping.\n")
        return

    if not is_archive_complete(archive_path, expected_size):
        if archive_path.is_file():
            sys.stdout.write(
                f"--> Existing archive is incomplete "
                f"({archive_path.stat().st_size:,} / {expected_size:,} bytes). Re-downloading...\n"
            )
            archive_path.unlink()

        download_succeeded = False
        for url in urls:
            try:
                if _try_aria2c_download(url, dest_dir, archive_name, workers, expected_size):
                    download_succeeded = True
                    break
                # Fallback to high-speed native multi-stream downloader
                sys.stdout.write(f"--> Launching native multi-stream downloader from {url}...\n")
                native_multi_stream_download(url, archive_path, num_workers=workers)
                if is_archive_complete(archive_path, expected_size):
                    download_succeeded = True
                    break
            except Exception as exc:
                sys.stderr.write(f"Warning: download attempt failed from {url}: {exc}\n")

        if not download_succeeded or not is_archive_complete(archive_path, expected_size):
            msg = f"Failed to download complete {archive_name} ({expected_size:,} bytes expected)."
            raise RuntimeError(msg)
    else:
        size_str = f"{archive_path.stat().st_size:,}"
        sys.stdout.write(
            f"--> Complete archive already present ({size_str} bytes): {archive_path}\n"
        )

    sys.stdout.write(f"--> Extracting {archive_name} into {dest_dir}...\n")
    cmd_tar = ["tar", "-xzf", str(archive_path), "-C", str(dest_dir)]
    subprocess.run(cmd_tar, check=True)
    sys.stdout.write(f"--> Extraction complete for {archive_name}.\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_musan:
        musan_path = args.out_dir / "musan" / "musan.tar.gz"
        download_and_extract(
            MUSAN_URLS,
            musan_path,
            expected_size=MUSAN_EXPECTED_BYTES,
            is_extracted_fn=is_musan_extracted,
            workers=args.workers,
        )

    if not args.skip_reverb:
        reverb_path = args.out_dir / "reverb" / "BUT_ReverbDB_RIR.tgz"
        download_and_extract(
            REVERB_URLS,
            reverb_path,
            expected_size=REVERB_EXPECTED_BYTES,
            is_extracted_fn=is_reverb_extracted,
            workers=args.workers,
        )

    sys.stdout.write("\n--> Multi-condition augmentation assets ready.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
