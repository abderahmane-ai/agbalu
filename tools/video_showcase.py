"""Render four synchronized speech players as a finished social video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from agbalu.codec_tts.artifacts import atomic_binary, digest_file, provenance, write_json
from tools.speech_showcase import Showcase, Voice, Word

WIDTH, HEIGHT, FPS, RATE_HZ = 1080, 1350, 30, 24000
INTRO_S, GAP_S, OUTRO_S = 1.2, 0.35, 1.4
BACKGROUND = (246, 244, 238)
INK = (27, 39, 40)
MUTED = (115, 125, 123)
ACCENT = (33, 106, 91)
CARD = (237, 237, 230)
MAX_CAPTION_WORDS = 8
CAPTION_WIDTH = 900


@dataclass(frozen=True)
class Clip:
    voice: Voice
    waveform: NDArray[np.float32]
    bars: NDArray[np.float32]
    words: tuple[Word, ...]
    start_s: float
    end_s: float


def schedule(durations_s: list[float]) -> list[tuple[float, float]]:
    if not durations_s or any(not math.isfinite(d) or d <= 0 for d in durations_s):
        message = f"invalid clip durations_s={durations_s}"
        raise ValueError(message)
    cursor = INTRO_S
    intervals: list[tuple[float, float]] = []
    for duration in durations_s:
        intervals.append((cursor, cursor + duration))
        cursor += duration + GAP_S
    return intervals


def waveform_bars(waveform: NDArray[np.float32], count: int = 100) -> NDArray[np.float32]:
    if waveform.ndim != 1 or waveform.size < count or not np.isfinite(waveform).all():
        message = f"invalid waveform shape={waveform.shape}, count={count}"
        raise ValueError(message)
    values = np.asarray(
        [
            math.sqrt(float(np.mean(chunk.astype(np.float64) ** 2)))
            for chunk in np.array_split(waveform, count)
        ],
        dtype=np.float32,
    )
    maximum = float(values.max())
    if maximum == 0:
        message = "cannot display a silent speech clip"
        raise ValueError(message)
    return np.asarray((values / maximum) ** 0.65, dtype=np.float32)


def ease(value: float) -> float:
    bounded = max(0.0, min(1.0, value))
    return bounded * bounded * (3 - 2 * bounded)


def caption_groups(words: tuple[Word, ...]) -> tuple[tuple[Word, ...], ...]:
    groups: list[tuple[Word, ...]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        boundary = word.text.endswith((".", "!", "?", ";", ":"))
        is_link = word.text.lower() in {"d", "n", "i", "s", "ad"}
        if boundary or (len(current) >= MAX_CAPTION_WORDS and not is_link):
            groups.append(tuple(current))
            current = []
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def mix(
    first: tuple[int, int, int], second: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    red = round(first[0] + (second[0] - first[0]) * amount)
    green = round(first[1] + (second[1] - first[1]) * amount)
    blue = round(first[2] + (second[2] - first[2]) * amount)
    return red, green, blue


def parse_loudness(log: str) -> dict[str, str]:
    opening = log.rfind("{")
    if opening < 0:
        message = "FFmpeg returned no loudness report"
        raise ValueError(message)
    payload, _ = json.JSONDecoder().raw_decode(log[opening:])
    if not isinstance(payload, dict):
        message = "FFmpeg returned a non-object loudness report"
        raise TypeError(message)
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            message = f"invalid loudness field={key!r}, value={value!r}"
            raise TypeError(message)
        result[key] = value
    return result


def master_audio(source: Path, output: Path, ffmpeg: Path) -> None:
    analysis = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-i",
            str(source),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    measurements = parse_loudness(analysis.stderr)
    for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        if not math.isfinite(float(measurements[key])):
            message = f"invalid loudness {key}={measurements[key]!r}: {source}"
            raise ValueError(message)
    audio_filter = (
        "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:"
        f"measured_I={measurements['input_i']}:measured_TP={measurements['input_tp']}:"
        f"measured_LRA={measurements['input_lra']}:measured_thresh={measurements['input_thresh']}:"
        f"offset={measurements['target_offset']}:print_format=json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-i",
            str(source),
            "-af",
            audio_filter,
            "-ar",
            str(RATE_HZ),
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    write_json(
        output.with_suffix(".stats.json"),
        {
            "source_sha256": digest_file(source),
            "sha256": digest_file(output),
            "loudness": parse_loudness(result.stderr),
        },
    )


class PlayerFilm:
    def __init__(self, spec: Showcase, clips: list[Clip]) -> None:
        self.spec, self.clips = spec, clips
        self.duration_s = clips[-1].end_s + OUTRO_S
        self.fonts: dict[tuple[int, str], ImageFont.FreeTypeFont] = {}

    def font(self, size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
        key = size, weight
        if key not in self.fonts:
            font = ImageFont.truetype(str(self.spec.font), size)
            font.set_variation_by_name(weight)
            self.fonts[key] = font
        return self.fonts[key]

    def active_position(self, seconds: float) -> float:
        position = 0.0
        for index, clip in enumerate(self.clips[:-1]):
            if seconds >= clip.end_s:
                position = index + ease((seconds - clip.end_s) / GAP_S)
        return position

    def frame(self, seconds: float) -> Image.Image:
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text((72, 50), "AƔBALU", font=self.font(25, "SemiBold"), fill=INK)
        draw.text(
            (1008, 53),
            "Matoub–4B",
            font=self.font(22, "Medium"),
            fill=MUTED,
            anchor="ra",
        )
        draw.text((68, 117), "One language.", font=self.font(77, "SemiBold"), fill=INK)
        draw.text((68, 209), "Four voices.", font=self.font(77, "SemiBold"), fill=INK)
        draw.text(
            (74, 315),
            "KABYLE TTS, WITH VOICE CLONING",
            font=self.font(17, "Medium"),
            fill=MUTED,
        )
        position = self.active_position(seconds)
        for index, clip in enumerate(self.clips):
            top = 365 + index * 178
            focus = max(0.0, 1 - abs(position - index))
            progress = min(1.0, max(0.0, (seconds - clip.start_s) / (clip.end_s - clip.start_s)))
            draw.rounded_rectangle(
                (72, top, 1008, top + 160), radius=20, fill=mix(CARD, (255, 255, 252), focus)
            )
            draw.rounded_rectangle(
                (72, top, 1008, top + 160),
                radius=20,
                outline=mix(CARD, (199, 218, 207), focus),
                width=2,
            )
            center = 122, top + 80
            draw.ellipse((94, top + 52, 150, top + 108), fill=mix((222, 226, 217), ACCENT, focus))
            color = mix((128, 139, 131), (255, 255, 252), focus)
            is_playing = clip.start_s <= seconds < clip.end_s
            if progress >= 1:
                draw.line([(111, top + 80), (119, top + 88), (134, top + 72)], fill=color, width=3)
            elif is_playing:
                for pause_x in (115, 126):
                    draw.rounded_rectangle(
                        (pause_x, top + 69, pause_x + 5, top + 91), radius=2, fill=color
                    )
            else:
                draw.polygon(
                    [
                        (center[0] - 5, center[1] - 10),
                        (center[0] - 5, center[1] + 10),
                        (center[0] + 10, center[1]),
                    ],
                    fill=color,
                )
            draw.text((180, top + 19), clip.voice.name, font=self.font(30, "SemiBold"), fill=INK)
            label = "Synthetic · " if clip.voice.identity == "celebrity" else "Demo name · "
            draw.text(
                (181, top + 62),
                label + clip.voice.route.replace(" → ", " to "),
                font=self.font(18),
                fill=MUTED,
            )
            duration = max(1, round(clip.end_s - clip.start_s))
            elapsed = (
                duration if progress >= 1 else min(duration, int(max(0, seconds - clip.start_s)))
            )
            timer = f"{elapsed // 60}:{elapsed % 60:02d} / {duration // 60}:{duration % 60:02d}"
            draw.text((968, top + 27), timer, font=self.font(20), fill=MUTED, anchor="ra")
            x0, x1 = 182, 965
            cursor = x0 + progress * (x1 - x0)
            for bar, value in enumerate(clip.bars):
                x = x0 + bar * (x1 - x0) / (len(clip.bars) - 1)
                height = max(4.0, float(value) * 38)
                bar_color = ACCENT if x <= cursor and progress > 0 else (207, 215, 205)
                draw.rounded_rectangle(
                    (x, top + 121 - height / 2, x + 4, top + 121 + height / 2),
                    radius=2,
                    fill=bar_color,
                )
            if is_playing:
                draw.line((cursor, top + 97, cursor, top + 145), fill=ACCENT, width=2)
        highlight_y = 365 + position * 178
        draw.rounded_rectangle((73, highlight_y + 33, 77, highlight_y + 127), radius=2, fill=ACCENT)
        active = next((c for c in self.clips if c.start_s <= seconds < c.end_s), None)
        if active is not None:
            self.caption(draw, active, seconds - active.start_s)
        elif seconds >= self.clips[-1].end_s + 0.2:
            draw.text(
                (540, 1136),
                "Taqbaylit, s taɣect-nwen.",
                font=self.font(37, "Medium"),
                fill=INK,
                anchor="ma",
            )
        draw.line((72, 1257, 1008, 1257), fill=(213, 219, 210), width=1)
        draw.text((72, 1280), "agbalu/Matoub-4B", font=self.font(20, "Medium"), fill=INK)
        return image

    def caption(self, draw: ImageDraw.ImageDraw, clip: Clip, seconds: float) -> None:
        groups = caption_groups(clip.words)
        words = next((group for group in groups if seconds < group[-1].end_s), groups[-1])
        font = self.font(36, "Medium")
        lines: list[list[Word]] = [[]]
        width = 0.0
        for word in words:
            length = float(font.getlength(word.text + " "))
            if width + length > CAPTION_WIDTH and lines[-1]:
                lines.append([])
                width = 0.0
            lines[-1].append(word)
            width += length
        for row, line in enumerate(lines):
            width = sum(float(font.getlength(w.text + " ")) for w in line)
            x = (WIDTH - width) / 2
            for word in line:
                color = ACCENT if word.start_s <= seconds < word.end_s else INK
                draw.text((x, 1110 + row * 51), word.text, font=font, fill=color)
                x += float(font.getlength(word.text + " "))


def load_film(spec: Showcase, ffmpeg: Path) -> PlayerFilm:
    audio_dir = spec.output / "audio"
    waves: list[NDArray[np.float32]] = []
    for voice in spec.voices:
        source = spec.output / "selected" / f"{voice.key}.wav"
        mastered = audio_dir / source.name
        mastering_report = mastered.with_suffix(".stats.json")
        has_current_master = False
        if mastered.is_file() and mastering_report.is_file():
            identity = json.loads(mastering_report.read_bytes())
            has_current_master = identity["source_sha256"] == digest_file(source) and identity[
                "sha256"
            ] == digest_file(mastered)
        if not has_current_master:
            master_audio(source, mastered, ffmpeg)
        data, rate = sf.read(mastered, dtype="float32")
        if rate != RATE_HZ or data.ndim != 1 or not np.isfinite(data).all():
            message = f"invalid mastered audio: {mastered}, rate={rate}, shape={data.shape}"
            raise ValueError(message)
        waves.append(np.asarray(data, dtype=np.float32))
    intervals = schedule([w.size / RATE_HZ for w in waves])
    clips: list[Clip] = []
    for voice, waveform, (start, end) in zip(spec.voices, waves, intervals, strict=True):
        data = json.loads((spec.output / f"{voice.key}.words.json").read_bytes())
        if data["audio_sha256"] != digest_file(spec.output / "selected" / f"{voice.key}.wav"):
            message = f"subtitle audio changed for {voice.key}"
            raise ValueError(message)
        words = tuple(Word.model_validate(w) for w in data["words"])
        if not words or " ".join(w.text for w in words) != " ".join(voice.text.split()):
            message = f"subtitle text conservation failed for {voice.key}"
            raise ValueError(message)
        clips.append(Clip(voice, waveform, waveform_bars(waveform), words, start, end))
    return PlayerFilm(spec, clips)


def render(film: PlayerFilm, ffmpeg: Path) -> Path:
    output = film.spec.output
    master = np.zeros(math.ceil(film.duration_s * RATE_HZ), dtype=np.float32)
    for clip in film.clips:
        offset = round(clip.start_s * RATE_HZ)
        master[offset : offset + clip.waveform.size] = clip.waveform
    with atomic_binary(output / "master.wav") as stream:
        sf.write(stream, master, RATE_HZ, format="WAV", subtype="FLOAT")
    target = output / "Matoub-4B-four-voices.mp4"
    temporary = output / ".Matoub-4B-four-voices.partial.mp4"
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-i",
        str(output / "master.wav"),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-t",
        str(film.duration_s),
        str(temporary),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            message = "FFmpeg did not open its frame input"
            raise RuntimeError(message)
        for frame in range(math.ceil(film.duration_s * FPS)):
            process.stdin.write(film.frame(frame / FPS).tobytes())
            if frame % (FPS * 5) == 0:
                print(
                    json.dumps({"rendered_s": frame / FPS, "total_s": film.duration_s}), flush=True
                )
        process.stdin.close()
        code = process.wait(timeout=120)
        if code:
            raise subprocess.CalledProcessError(code, command)
    temporary.replace(target)
    write_json(
        target.with_suffix(".stats.json"),
        {
            **provenance(hashlib.sha256(film.spec.model_dump_json().encode()).hexdigest()),
            "sha256": digest_file(target),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_s": film.duration_s,
            "audio_samples": master.size,
            "clips": [
                {
                    "key": c.voice.key,
                    "start_s": c.start_s,
                    "end_s": c.end_s,
                    "text": c.voice.text,
                    "audio_sha256": digest_file(output / "audio" / f"{c.voice.key}.wav"),
                }
                for c in film.clips
            ],
        },
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/runs/matoub-linkedin/storyboard.json")
    )
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--task", choices=["preview", "render"], default="preview")
    parser.add_argument("--time", type=float, default=4.0)
    args = parser.parse_args()
    film = load_film(Showcase.model_validate_json(args.manifest.read_bytes()), args.ffmpeg)
    if args.task == "preview":
        film.frame(args.time).save(film.spec.output / "preview.png")
    else:
        print(render(film, args.ffmpeg), flush=True)


if __name__ == "__main__":
    main()
