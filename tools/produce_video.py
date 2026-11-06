"""Compose consent-gated speech launch videos from audited HPC recordings."""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import subprocess
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from agbalu.codec_tts.artifacts import atomic_binary, digest_file, write_json
from agbalu.llm.provenance import code_stamp

OUTPUT = Path("artifacts/matoub-4b-launch")
FPS = 30
RATE_HZ = 24000
DURATION_S = 55
DURATION_TOLERANCE_S = 0.1
BACKGROUND = "#15191b"
WHITE = "#f3efe6"
MUTED = "#a8b3b4"
ACCENT = "#62d3c5"


class Specification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sample(Specification):
    speaker_id: str = Field(min_length=1)
    reference_audio: Path
    reference_text: str = Field(min_length=1)
    reference_language: Literal["French", "English"]
    target_text: str = Field(min_length=1)
    target_source: Path
    generated_audio: Path
    consent_document: Path
    review_document: Path
    generation_audit: Path


class Consent(Specification):
    speaker_id: str
    signed_by: str = Field(min_length=1)
    signed_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    reference_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    voice_cloning: Literal[True]
    public_demonstration: Literal[True]
    distribute_audio: Literal[True]


class Review(Specification):
    reviewer: str = Field(min_length=1)
    generated_sha256: str
    pronunciation: Literal[True]
    naturalness: Literal[True]
    speaker_identity: Literal[True]
    complete_utterance: Literal[True]
    exact_transcripts_verified: Literal[True]
    kabyle_text_verified: Literal[True]


class Generation(Specification):
    repository: Literal["agbalu/Matoub-4B"]
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    execution_target: Literal["hpc"]
    precision: Literal["bf16", "nf4"]
    reference_sha256: str
    generated_sha256: str
    reference_text: str
    target_text: str
    settings_file: Path
    report_file: Path


class Production(Specification):
    version: Literal[1]
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    samples: tuple[Sample, ...] = Field(max_length=2)
    publication_permission: Path | None


class Permission(Specification):
    signed_permission_file: Path
    public_video_without_upstream_attribution: Literal[True]
    public_audio_distribution: Literal[True]


@dataclass(frozen=True)
class Scene:
    start_s: float
    end_s: float
    label: str
    title: str
    body: str
    detail: str
    audio: Path | None = None


def require_equal(actual: object, expected: object, location: str) -> None:
    if actual != expected:
        message = f"{location}: expected {expected!r}, received {actual!r}"
        raise ValueError(message)


def read_production(path: Path) -> Production:
    production = Production.model_validate_json(path.read_text(encoding="utf-8"))
    identities = [sample.speaker_id for sample in production.samples]
    if len(set(identities)) != len(identities):
        message = f"{path}: duplicate speaker identities {identities!r}"
        raise ValueError(message)
    for sample in production.samples:
        verify_sample(sample, production.model_revision)
    if production.publication_permission is not None:
        permission = Permission.model_validate_json(
            production.publication_permission.read_text(encoding="utf-8")
        )
        permission.signed_permission_file.read_bytes()
    return production


def verify_sample(sample: Sample, revision: str) -> None:
    consent = Consent.model_validate_json(sample.consent_document.read_text(encoding="utf-8"))
    review = Review.model_validate_json(sample.review_document.read_text(encoding="utf-8"))
    generation = Generation.model_validate_json(sample.generation_audit.read_text(encoding="utf-8"))
    reference_hash = digest_file(sample.reference_audio)
    generated_hash = digest_file(sample.generated_audio)
    checks = (
        (consent.speaker_id, sample.speaker_id, "consent speaker"),
        (consent.reference_sha256, reference_hash, "consent reference"),
        (generation.reference_sha256, reference_hash, "generation reference"),
        (generation.generated_sha256, generated_hash, "generation output"),
        (generation.revision, revision, "model revision"),
        (generation.reference_text, sample.reference_text, "reference transcript"),
        (generation.target_text, sample.target_text, "target transcript"),
        (review.generated_sha256, generated_hash, "review output"),
    )
    for actual, expected, location in checks:
        require_equal(actual, expected, f"{sample.generation_audit}: {location}")
    if sample.target_text not in sample.target_source.read_text(encoding="utf-8"):
        message = f"{sample.target_source}: target text absent: {sample.target_text!r}"
        raise ValueError(message)
    generation.settings_file.read_bytes()
    generation.report_file.read_bytes()


def build_scenes(samples: Sequence[Sample]) -> tuple[Scene, ...]:
    first = samples[0] if samples else None
    second = samples[1] if len(samples) > 1 else None
    return (
        Scene(
            0,
            4,
            "AƔBALU / SPEECH RESEARCH",
            "Matoub-4B",
            "Kabyle Speech Synthesis",
            "& Voice Cloning",
        ),
        Scene(
            4,
            12,
            "01 / REFERENCE VOICE",
            first.reference_language if first else "A voice begins here.",
            first.reference_text if first else "Consented reference recording required.",
            "Original recording" if first else "No voice recording is included in this preview.",
            first.reference_audio if first else None,
        ),
        Scene(
            12,
            17,
            "02 / TARGET TEXT",
            "Taqbaylit",
            first.target_text if first else "Verified Kabyle text pending.",
            "Exact synthesis input"
            if first
            else "The final words will come from a verified source.",
        ),
        Scene(
            17,
            34,
            "03 / MATOUB-4B GENERATED SPEECH",
            "A voice. In Kabyle.",
            first.target_text if first else "HPC-generated speech pending.",
            "Synthetic speech" if first else "No synthetic speech is included in this preview.",
            first.generated_audio if first else None,
        ),
        Scene(
            34,
            38,
            "04 / A DIFFERENT SPEAKER",
            second.reference_language if second else "Another voice.",
            second.reference_text if second else "A distinct speaker requires separate consent.",
            "Original recording" if second else "Second demonstration not yet available.",
            second.reference_audio if second else None,
        ),
        Scene(
            38,
            45,
            "04 / A DIFFERENT SPEAKER",
            "Taqbaylit",
            second.target_text if second else "Hear the difference.",
            "Synthetic speech" if second else "Audio comparison pending cleared recordings.",
            second.generated_audio if second else None,
        ),
        Scene(
            45,
            50,
            "REFERENCE-CONDITIONED SYNTHESIS",
            "Made for Taqbaylit.",
            "24 kHz / Mono",
            "Reference audio + exact transcript → Kabyle speech",
        ),
        Scene(
            50,
            55,
            "DEVELOPED BY AƔBALU",
            "Matoub-4B",
            "Kabyle Speech Synthesis & Voice Cloning",
            "huggingface.co/agbalu/Matoub-4B",
        ),
    )


def locate_scene(scenes: Sequence[Scene], time_s: float) -> Scene:
    for scene in scenes:
        if scene.start_s <= time_s < scene.end_s:
            return scene
    message = f"No scene at time_s={time_s!r}"
    raise ValueError(message)


def calculate_envelope(values: Sequence[float], bins: int) -> tuple[float, ...]:
    if not values or bins <= 0 or any(not math.isfinite(value) for value in values):
        message = f"Invalid waveform: samples={len(values)}, bins={bins}"
        raise ValueError(message)
    return tuple(
        max(
            (
                abs(value)
                for value in values[index * len(values) // bins : (index + 1) * len(values) // bins]
            ),
            default=0.0,
        )
        for index in range(bins)
    )


def execute(arguments: Sequence[str], timeout_s: int = 120) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(arguments), capture_output=True, check=True, timeout=timeout_s)


def binary(output: Path, name: str) -> str:
    path = output / "toolchain" / name / "package" / name
    if not path.is_file():
        message = f"Missing local rendering binary: {path}"
        raise FileNotFoundError(message)
    return str(path.resolve())


def decode_audio(output: Path, path: Path) -> tuple[float, ...]:
    raw = execute(
        [
            binary(output, "ffmpeg"),
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(RATE_HZ),
            "-f",
            "f32le",
            "pipe:1",
        ]
    ).stdout
    values = tuple(float(item[0]) for item in struct.iter_unpack("<f", raw))
    if not values or max(abs(value) for value in values) >= 1:
        message = f"Empty or clipped audio: {path}"
        raise ValueError(message)
    return values


@cache
def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


class Canvas:
    def __init__(self, output: Path, vertical: bool, is_preview: bool) -> None:
        self.output = output
        self.vertical = vertical
        self.is_preview = is_preview
        self.width, self.height = (1080, 1920) if vertical else (1920, 1080)
        self.margin = 88 if vertical else 128
        self.content_width = self.width - self.margin * 2

    def text(
        self,
        image: Image.Image,
        text: str,
        position: tuple[int, int],
        style: tuple[int, str],
        *,
        headline: bool = False,
    ) -> int:
        font_path = self.output / "assets" / ("Manrope.ttf" if headline else "NotoSans.ttf")
        font = load_font(font_path, style[0])
        draw = ImageDraw.Draw(image)
        lines: list[str] = []
        for paragraph in text.split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}".strip()
                if line and draw.textlength(candidate, font=font) > self.content_width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        y = position[1]
        for line in lines:
            if draw.textlength(line, font=font) > self.content_width:
                message = f"Text exceeds safe width {self.content_width}: {line!r}"
                raise ValueError(message)
            draw.text((position[0], y), line, font=font, fill=style[1], anchor="lt")
            y += round(style[0] * 1.45)
        return y

    def compose(self, scene: Scene) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        x = self.margin
        self.text(image, "AƔBALU", (x, 100 if self.vertical else 60), (28, WHITE))
        if self.is_preview:
            self.text(
                image,
                "PRIVATE PREVIEW / SPEECH PENDING"
                if scene.audio is None
                else "PRIVATE PREVIEW / PUBLICATION BLOCKED",
                (x, self.height - 130 if self.vertical else self.height - 90),
                (19, MUTED),
            )
        y = 470 if self.vertical else 260
        self.text(image, scene.label, (x, y), (24, ACCENT))
        draw.line((x, y + 65, x + 70, y + 65), fill=ACCENT, width=3)
        title_size = 118 if scene.title == "Matoub-4B" else (76 if self.vertical else 96)
        title_bottom = self.text(
            image, scene.title, (x, y + 110), (title_size, WHITE), headline=True
        )
        body_bottom = self.text(
            image, scene.body, (x, title_bottom + 45), (38 if self.vertical else 40, WHITE)
        )
        self.text(
            image,
            scene.detail,
            (x, max(body_bottom + 55, 1360 if self.vertical else 800)),
            (27 if self.vertical else 26, MUTED),
        )
        return image

    def waveform(self, image: Image.Image, envelope: Sequence[float], fraction: float) -> None:
        draw = ImageDraw.Draw(image)
        y = 1200 if self.vertical else 710
        for index, amplitude in enumerate(envelope):
            x = self.margin + round(index * self.content_width / len(envelope))
            height = max(1, round(amplitude * 75))
            draw.line(
                (x, y - height, x, y + height),
                fill=ACCENT if index / len(envelope) <= fraction else "#3b4749",
                width=3,
            )


def prepare_audio(output: Path, scenes: Sequence[Scene]) -> dict[Path, tuple[float, ...]]:
    envelopes: dict[Path, tuple[float, ...]] = {}
    for scene in scenes:
        if scene.audio is not None:
            values = decode_audio(output, scene.audio)
            duration_s = len(values) / RATE_HZ
            if duration_s > scene.end_s - scene.start_s - 0.2:
                message = f"{scene.audio}: {duration_s}s exceeds scene; refusing to truncate speech"
                raise ValueError(message)
            envelopes[scene.audio] = calculate_envelope(values, 150)
    return envelopes


def write_silence(path: Path) -> None:
    with atomic_binary(path) as stream, wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE_HZ)
        audio.writeframes(bytes(RATE_HZ * DURATION_S * 2))


def mix_audio(output: Path, scenes: Sequence[Scene]) -> Path:
    path = output / "working" / "timeline.wav"
    write_silence(path)
    occupied = [scene for scene in scenes if scene.audio is not None]
    if not occupied:
        return path
    arguments = [binary(output, "ffmpeg"), "-v", "error", "-i", str(path)]
    filters: list[str] = []
    for index, scene in enumerate(occupied, 1):
        arguments.extend(["-i", str(scene.audio)])
        values = decode_audio(output, scene.audio) if scene.audio is not None else ()
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        peak = max(abs(value) for value in values)
        gain_db = min(3.0, -22 - 20 * math.log10(rms), -2 - 20 * math.log10(peak))
        # A single bounded gain preserves dynamics; references retain their original level.
        gain = f",volume={gain_db}dB" if "Synthetic" in scene.detail else ""
        filters.append(
            f"[{index}:a]aresample={RATE_HZ}{gain},adelay={int(scene.start_s * 1000)}[a{index}]"
        )
    filters.append(
        "[0:a]"
        + "".join(f"[a{i}]" for i in range(1, len(occupied) + 1))
        + f"amix=inputs={len(occupied) + 1}:normalize=0:duration=first[out]"
    )
    arguments.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(RATE_HZ),
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ]
    )
    mixed = execute(arguments).stdout
    with atomic_binary(path) as stream:
        stream.write(mixed)
    return path


def render_video(
    output: Path, scenes: Sequence[Scene], canvas: Canvas, envelopes: dict[Path, tuple[float, ...]]
) -> Path:
    name = "vertical" if canvas.vertical else "landscape"
    destination = output / f"matoub-4b-{name}-private-preview.mp4"
    if destination.exists():
        message = f"Refusing to overwrite existing video: {destination}"
        raise FileExistsError(message)
    templates = {scene: canvas.compose(scene) for scene in scenes}
    arguments = [
        binary(output, "ffmpeg"),
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{canvas.width}x{canvas.height}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-i",
        str(output / "working" / "timeline.wav"),
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
        str(RATE_HZ),
        "-ac",
        "1",
        "-t",
        str(DURATION_S),
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        "pipe:1",
    ]
    # MP4 faststart needs seeking, so encode into a temporary sibling before atomic promotion.
    temporary = destination.with_suffix(".partial.mp4")
    arguments[-1] = str(temporary)
    try:
        with (
            (output / f"{name}-render.log").open("wb") as log,
            subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=log, stderr=log) as encoder,
        ):
            if encoder.stdin is None:
                message = "FFmpeg input pipe unavailable"
                raise RuntimeError(message)
            for frame in range(DURATION_S * FPS):
                time_s = frame / FPS
                scene = locate_scene(scenes, time_s)
                image = templates[scene].copy()
                if scene.audio is not None:
                    canvas.waveform(
                        image,
                        envelopes[scene.audio],
                        (time_s - scene.start_s) / (scene.end_s - scene.start_s),
                    )
                fade = min(1.0, (time_s - scene.start_s) / 0.25, (scene.end_s - time_s) / 0.25)
                if fade < 1:
                    image = Image.blend(Image.new("RGB", image.size, BACKGROUND), image, fade)
                encoder.stdin.write(image.tobytes())
            encoder.stdin.close()
            require_equal(encoder.wait(timeout=120), 0, str(temporary))
        with temporary.open("rb") as stream, atomic_binary(destination) as target:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                target.write(chunk)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def subtitle_time(time_s: float) -> str:
    milliseconds = round(time_s * 1000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_text(path: Path, text: str) -> None:
    with atomic_binary(path) as stream:
        stream.write(text.encode("utf-8"))


def write_captions(output: Path, scenes: Sequence[Scene], production: Production) -> None:
    blocks = [
        f"{index}\n{subtitle_time(scene.start_s)} --> {subtitle_time(scene.end_s)}\n"
        f"{scene.title}\n{scene.body}\n"
        for index, scene in enumerate(scenes, 1)
    ]
    write_text(output / "english.srt", "\n".join(blocks))
    transcript = "\n\n".join(sample.target_text for sample in production.samples)
    write_text(
        output / "kabyle-transcript.txt",
        transcript + "\n"
        if transcript
        else (
            "No Kabyle speech is present. Verified target text and consented HPC synthesis "
            "are pending.\n"
        ),
    )


def save_image(path: Path, image: Image.Image) -> None:
    with atomic_binary(path) as stream:
        image.save(stream, format="PNG")


def extract_contact_sheet(output: Path, video: Path) -> None:
    frames: list[Image.Image] = []
    for time_s in (2, 6, 14, 20, 30, 36, 41, 47, 52):
        result = execute(
            [
                binary(output, "ffmpeg"),
                "-v",
                "error",
                "-ss",
                str(time_s),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ]
        )
        with Image.open(io.BytesIO(result.stdout)) as source:
            frame = source.convert("RGB")
        frame.thumbnail((480, 480))
        frames.append(frame)
    cell_width, cell_height = (
        max(frame.width for frame in frames),
        max(frame.height for frame in frames),
    )
    sheet = Image.new("RGB", (cell_width * 3, (cell_height + 38) * 3), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x, y = index % 3 * cell_width, index // 3 * (cell_height + 38)
        sheet.paste(frame, (x, y))
        draw.text(
            (x + 12, y + cell_height + 8),
            f"Frame {index + 1}",
            font=load_font(output / "assets" / "NotoSans.ttf", 18),
            fill=WHITE,
        )
    save_image(output / f"{video.stem}-contact-sheet.png", sheet)


def verify_video(output: Path, path: Path) -> dict[str, object]:
    probe = execute(
        [
            binary(output, "ffprobe"),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    metadata: dict[str, object] = json.loads(probe.stdout)
    write_json(path.with_suffix(".ffprobe.json"), metadata)
    execute([binary(output, "ffmpeg"), "-v", "error", "-i", str(path), "-f", "null", "-"])
    video_data = (
        execute(
            [
                binary(output, "ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,codec_name,nb_frames",
                "-of",
                "csv=p=0",
                str(path),
            ]
        )
        .stdout.decode()
        .strip()
    )
    vertical = "vertical" in path.name
    expected = f"h264,{1080 if vertical else 1920},{1920 if vertical else 1080},30/1,1650"
    require_equal(video_data, expected, str(path))
    audio = (
        execute(
            [
                binary(output, "ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "csv=p=0",
                str(path),
            ]
        )
        .stdout.decode()
        .strip()
    )
    require_equal(audio, "aac,24000,1", str(path))
    duration_s = float(
        execute(
            [
                binary(output, "ffprobe"),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ]
        ).stdout
    )
    if abs(duration_s - DURATION_S) > DURATION_TOLERANCE_S:
        message = f"{path}: duration_s={duration_s}, expected={DURATION_S}"
        raise ValueError(message)
    return {
        "path": str(path),
        "sha256": digest_file(path),
        "duration_s": duration_s,
        "video": video_data,
        "audio": audio,
        "decode_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("render", "verify"), default="render")
    parser.add_argument("--config", type=Path, default=Path("resources/launch_video.json"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    output: Path = arguments.output
    production = read_production(arguments.config)
    scenes = build_scenes(production.samples)
    output.mkdir(parents=True, exist_ok=True)
    if arguments.task == "render":
        envelopes = prepare_audio(output, scenes)
        mix_audio(output, scenes)
        write_captions(output, scenes, production)
        for vertical in (False, True):
            canvas = Canvas(output, vertical, True)
            video = render_video(output, scenes, canvas, envelopes)
            extract_contact_sheet(output, video)
        save_image(output / "thumbnail.png", Canvas(output, False, True).compose(scenes[0]))
    videos = sorted(output.glob("*.mp4"))
    require_equal(len(videos), 2, str(output))
    report = {
        "stamp": code_stamp(),
        "config_sha256": digest_file(arguments.config),
        "public_release_ready": False,
        "speech_samples_this_run": len(production.samples),
        "silent_preview": not production.samples,
        "videos": [verify_video(output, path) for path in videos],
        "limitations": [
            "Voice consent and verified Kabyle target text pending",
            "Written permission for attribution-free public distribution pending",
        ],
        "timeline": [
            {
                "start_s": scene.start_s,
                "end_s": scene.end_s,
                "label": scene.label,
                "audio": str(scene.audio) if scene.audio else None,
            }
            for scene in scenes
        ],
    }
    write_json(output / "production.stats.json", report)


if __name__ == "__main__":
    main()
