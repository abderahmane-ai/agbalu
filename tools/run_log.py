"""A readable view of a Phase 11 run log, for the person watching it.

The run writes one JSON object per line to `log.txt`, which stays the record. This turns each
event into one line a person can read at a glance: a step with its loss, moving average, rate
and the clock time the run should finish; an evaluation against its step-0 values; each stage
with how long it took. Progress inside a step and heartbeats are folded into one live line that
rewrites itself on a terminal. Library chatter that says nothing about the run is dropped;
anything else, a traceback above all, passes through untouched.
"""

from __future__ import annotations

import json
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

LOSS_WINDOW: Final = 20
BAR_WIDTH: Final = 24
SAMPLE_CHARS: Final = 90
GIGABYTE: Final = 1e9

NOISE: Final = (
    "You are sending unauthenticated requests",
    "Fetching ",
    "Loading weights",
    "Writing model shards",
    "is falling back to its reference PyTorch implementation",
    "fatal: not a git repository",
    "Stopping at filesystem boundary",
    "Overwriting default num_fewshot",
)
"""Lines third-party libraries print on every run; none of them says anything about this run."""


@dataclass(frozen=True, slots=True)
class Rendered:
    text: str
    live: bool
    """A live line describes the moment and is replaced by the next one."""


def clock(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%a %d %b %H:%M")


def bar(done: int, total: int) -> str:
    filled = round(BAR_WIDTH * done / total) if total else 0
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


def change(value: float, start: float) -> str:
    return f"{100 * (value - start) / start:+.1f}%"


def short(text: object) -> str:
    flat = str(text).replace("\n", " \\n ")
    return flat if len(flat) <= SAMPLE_CHARS else flat[: SAMPLE_CHARS - 3] + "..."


class View:
    """Event lines in, readable lines out, keeping what a line needs from earlier ones."""

    def __init__(self) -> None:
        self.losses: deque[float] = deque(maxlen=LOSS_WINDOW)
        self.codec_losses: deque[tuple[float, int]] = deque(maxlen=LOSS_WINDOW)
        self.total: int | None = None
        self.reference: dict[str, float] | None = None
        self.reference_step: int | None = None
        self.progress = ""

    def seed(self, line: str) -> None:
        """Take the baseline from the run's first evaluation, which a log tail may not reach."""
        stripped = line.strip()
        if not stripped.startswith("{"):
            return
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return
        scores = event.get("bpc") if isinstance(event, dict) else None
        step = event.get("step") if isinstance(event, dict) else None
        if isinstance(scores, dict) and isinstance(step, int):
            self.reference = {str(k): float(v) for k, v in scores.items()}
            self.reference_step = step

    def render(self, line: str) -> Rendered | None:
        stripped = line.rstrip("\n").split("\r")[-1].strip()
        if not stripped or any(marker in stripped for marker in NOISE):
            return None
        if not stripped.startswith("{"):
            live = "it/s]" in stripped or "s/it]" in stripped
            return Rendered(stripped, live=live)
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return Rendered(stripped, live=False)
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            return Rendered(stripped, live=False)
        return self.event(event)

    def event(self, event: dict[str, object]) -> Rendered | None:
        kind = str(event["event"])
        handler = getattr(self, f"on_{kind}", None)
        if handler is None:
            values = ", ".join(f"{k} {v}" for k, v in event.items() if k not in {"event", "time"})
            return Rendered(f"{kind}: {values}", live=False)
        result: Rendered | None = handler(event)
        return result

    def on_start(self, event: dict[str, object]) -> Rendered:
        if event.get("family") == "dihya":
            total = event.get("steps")
            self.total = total if isinstance(total, int) else None
            return Rendered(
                f"== run started: {event.get('steps')} steps, {event.get('train_rows')} rows, "
                f"{event.get('classes')} classes, {event.get('params')} params",
                live=False,
            )
        total = event.get("total_steps")
        self.total = total if isinstance(total, int) else None
        resumed = event.get("resumed_from")
        origin = f"resumed from {resumed}" if resumed else "fresh start"
        parameters = event.get("parameters")
        size = (
            f"{parameters / 1e6:.1f}M parameters"
            if isinstance(parameters, int) and parameters < GIGABYTE
            else f"{parameters / 1e9:.2f}B parameters"
            if isinstance(parameters, int)
            else ""
        )
        return Rendered(
            f"== run started at step {event.get('step')} of {self.total}, {origin}, {size}",
            live=False,
        )

    def on_step(self, event: dict[str, object]) -> Rendered:
        if event.get("family") == "codec_student":
            return self.on_student_step(event)
        if event.get("family") == "codec_tts":
            return self.on_codec_step(event)
        if event.get("family") == "dihya":
            step = int(str(event["step"]))
            total = event.get("total_steps", self.total)
            total = total if isinstance(total, int) else None
            where = f"step {step:,}/{total:,}" if total else f"step {step:,}"
            loss = float(str(event["loss"]))
            cls_l = event.get("cls_loss", 0.0)
            con_l = event.get("con_loss", 0.0)
            speed = event.get("speed", 0.0)
            lr = float(str(event.get("lr", 0.0)))
            return Rendered(
                f"{where} | loss {loss:.4f} (cls {cls_l}, con {con_l})"
                f" | lr {lr:.2e} | {speed} seq/s",
                live=False,
            )
        step = int(str(event["step"]))
        total = event.get("total_steps", self.total)
        total = total if isinstance(total, int) else None
        self.total = total
        loss = float(str(event["loss"]))
        self.losses.append(loss)
        mean = statistics.fmean(self.losses)
        where = f"step {step:,}/{total:,} {100 * step / total:5.1f}%" if total else f"step {step:,}"
        eta = event.get("eta_h")
        finish = (
            f"ETA {eta:.1f} h, done {clock(float(str(event['time'])) + eta * 3600)}"
            if isinstance(eta, int | float)
            else "ETA unknown"
        )
        memory = event.get("memory")
        gigabytes = f"{memory / GIGABYTE:.1f} GB" if isinstance(memory, int) else "-"
        skipped = "" if event.get("taken", True) else "  SKIPPED: non-finite gradient"
        utilization = event.get("mfu")
        mfu = f"{utilization:.0%}" if isinstance(utilization, int | float) else "-"
        accuracy = event.get("accuracy")
        masked = event.get("mask_probability")
        encoder_metrics = (
            f" | acc {accuracy:.3f} mask {masked:.3f}"
            if isinstance(accuracy, int | float) and isinstance(masked, int | float)
            else ""
        )
        self.progress = ""
        return Rendered(
            f"{where} | loss {loss:.4f} (avg{len(self.losses)} {mean:.4f})"
            f" | lr {float(str(event['lr'])):.2e} | grad {float(str(event['grad_norm'])):.2f}"
            f" | {float(str(event['tokens_per_s'])):,.0f} tok/s"
            f" | MFU {mfu} | {gigabytes}"
            f" | {int(str(event['tokens'])) / 1e6:,.1f}M tok | {finish}{encoder_metrics}{skipped}",
            live=False,
        )

    def on_codec_step(self, event: dict[str, object]) -> Rendered:
        step = int(str(event["step"]))
        total = int(str(event["total_steps"]))
        numerator = float(str(event["loss_numerator"]))
        count = int(str(event["supervised_tokens"]))
        if count <= 0:
            message = f"invalid codec loss denominator={count} at step={step}"
            raise ValueError(message)
        self.codec_losses.append((numerator, count))
        mean = sum(value for value, _ in self.codec_losses) / sum(n for _, n in self.codec_losses)
        eta_s = event.get("eta_s")
        eta = "ETA warming up"
        if isinstance(eta_s, int | float):
            finish = datetime.fromtimestamp(float(str(event["time"])) + eta_s, tz=UTC)
            eta = f"ETA {eta_s / 3600:.2f} h, done {finish:%Y-%m-%d %H:%M UTC}"
        memory = event.get("memory")
        memory_text = f"{memory / 2**30:.1f} GiB" if isinstance(memory, int) else "unmeasured"
        speed = float(str(event["supervised_tokens_per_s"]))
        audio = float(str(event["target_audio_s_per_s"]))
        return Rendered(
            f"step {step}/{total} {bar(step, total)} | codec CE {numerator / count:.4f}"
            f" (weighted {mean:.4f}) | lr {float(str(event['lr'])):.2e}"
            f" | grad {float(str(event['grad_norm'])):.3f} | {speed:,.0f} decisions/s"
            f" | {audio:.2f} audio-s/s | {memory_text} | {eta}",
            live=False,
        )

    def on_student_step(self, event: dict[str, object]) -> Rendered:
        stage = event.get("stage")
        if stage == "alignment":
            step, total = int(str(event["step"])), int(str(event["total_steps"]))
            eta = event.get("eta_s")
            eta_text = (
                f"ETA {float(eta) / 3600:.2f} h"
                if isinstance(eta, int | float)
                else "ETA warming up"
            )
            return Rendered(
                f"alignment step {step}/{total} {bar(step, total)}"
                f" | representation {float(str(event['representation_loss'])):.5f}"
                f" | lr {float(str(event['lr'])):.2e}"
                f" | grad {float(str(event['grad_norm'])):.3f}"
                f" | {float(str(event['supervised_tokens_per_s'])):,.0f} decisions/s"
                f" | {eta_text}",
                live=False,
            )
        rendered = self.on_codec_step(event)
        objective = (
            f"alignment {float(str(event['representation_loss'])):.5f}"
            if stage == "alignment"
            else f"CE {float(str(event['audio_ce_nats'])):.4f}"
            f" + KD {float(str(event['kd_nats'])):.4f}"
        )
        return Rendered(f"{stage}: {objective} | {rendered.text}", live=False)

    def on_progress(self, event: dict[str, object]) -> Rendered:
        done, total = event.get("done"), event.get("total")
        eta = event.get("eta_s")
        left = f", {eta:.0f} s left" if isinstance(eta, int | float) else ""
        if isinstance(done, int) and isinstance(total, int):
            self.progress = f"  {event.get('task')} {bar(done, total)} {done}/{total}{left}"
        else:
            self.progress = f"  {event.get('task')}: {done}{left}"
        return Rendered(self.progress, live=True)

    def on_heartbeat(self, event: dict[str, object]) -> Rendered:
        gpu = event.get("gpu_allocated_gib")
        memory = f", GPU {gpu} GiB" if gpu is not None else ""
        phase = event.get("phase") or event.get("part") or event.get("label")
        alive = f"  (alive {clock(float(str(event['time'])))[-5:]}, {phase}{memory})"
        return Rendered((self.progress or f"  {event.get('label')}") + alive, live=True)

    def on_stage_start(self, event: dict[str, object]) -> Rendered:
        details = ", ".join(
            f"{k} {v}" for k, v in event.items() if k not in {"event", "time", "stage"}
        )
        return Rendered(
            f"-> {event.get('stage')}" + (f" ({details})" if details else ""), live=False
        )

    def on_stage_end(self, event: dict[str, object]) -> Rendered:
        return Rendered(f"   {event.get('stage')} done in {event.get('seconds')} s", live=False)

    def on_stage_failed(self, event: dict[str, object]) -> Rendered:
        return Rendered(
            f"!! {event.get('stage')} FAILED after {event.get('seconds')} s: {event.get('error')}",
            live=False,
        )

    def on_eval(self, event: dict[str, object]) -> Rendered:
        if event.get("family") in {"codec_tts", "codec_student"}:
            return Rendered(
                f"== codec validation step {event.get('step')}: CE {event.get('audio_ce_nats')}"
                f" | {event.get('examples')} examples (teacher forced)",
                live=False,
            )
        if event.get("family") == "restoration":
            offline = event.get("offline")
            streaming = event.get("streaming")
            offline = offline if isinstance(offline, dict) else {}
            streaming = streaming if isinstance(streaming, dict) else {}
            return Rendered(
                f"== restoration eval at step {event.get('step')}: "
                f"punctuation F1 {offline.get('punctuation_macro_f1')}"
                f" | stream F1 {streaming.get('punctuation_macro_f1')}"
                f" | casing F1 {offline.get('casing_macro_f1')}"
                f" | false insertions/1k words {offline.get('false_insertions_per_1000_words')}"
                f" | selection {event.get('score')}",
                live=False,
            )
        if event.get("family") == "encoder":
            health, tasks = event.get("mlm"), event.get("downstream")
            health = health if isinstance(health, dict) else {}
            tasks = tasks if isinstance(tasks, dict) else {}
            clean = tasks.get("surface/canonical")
            clean = clean if isinstance(clean, dict) else {}
            accuracy = clean.get("nonpunct_accuracy")
            score = f" | surface POS {accuracy:.2%}" if isinstance(accuracy, float) else ""
            return Rendered(
                f"== encoder eval at step {event.get('step')}: "
                f"MLM CE {health.get('ce')} | accuracy {health.get('accuracy')}{score}",
                live=False,
            )
        if event.get("family") == "dihya":
            acc = event.get("accuracy")
            w_rec = event.get("worst_group_recall")
            worst_grp = event.get("worst_group")
            w_cls = event.get("worst_class")
            w_cls_rec = event.get("worst_class_recall")
            grp_str = f" [{worst_grp}]" if worst_grp else ""
            cls_str = f" | worst class {w_cls_rec} [{w_cls}]" if w_cls else ""
            return Rendered(
                f"== dihya eval at step {event.get('step')}: "
                f"accuracy {acc} | worst recall {w_rec}{grp_str}{cls_str}",
                live=False,
            )
        scores = event.get("bpc")
        if not isinstance(scores, dict):
            return Rendered(f"eval at step {event.get('step')}: {scores}", live=False)
        step = event.get("step")
        given = event.get("reference")
        if isinstance(given, dict) and given:
            self.reference = {str(k): float(v) for k, v in given.items()}
            self.reference_step = 0
        elif self.reference is None:
            self.reference = {str(k): float(v) for k, v in scores.items()}
            self.reference_step = step if isinstance(step, int) else None
        compare = self.reference_step != step
        parts = []
        for language in ("kab", "eng", "fra"):
            value = float(scores[language])
            start = self.reference.get(language)
            delta = (
                f" ({change(value, start)} vs step {self.reference_step} {start:.3f})"
                if start and compare
                else ""
            )
            parts.append(f"{language} {value:.3f}{delta}")
        loss = event.get("validation_loss")
        if loss is not None:
            chosen_on = f"validation loss {float(str(loss)):.4f}"
        else:
            chosen_on = "drift guard ok" if event.get("eligible") else "DRIFT GUARD FAILED"
        return Rendered(
            f"== eval at step {event.get('step')}: held-out bpc "
            + " | ".join(parts)
            + f" | {chosen_on}",
            live=False,
        )

    def on_best(self, event: dict[str, object]) -> Rendered:
        if event.get("family") in {"codec_tts", "codec_student"}:
            return Rendered(
                f"   selected {event.get('role')} at step {event.get('step')}"
                f" | CE {event.get('audio_ce_nats')}",
                live=False,
            )
        if event.get("family") == "restoration":
            return Rendered(
                f"   new restoration checkpoint: score {event.get('score')}"
                f" at step {event.get('step')}",
                live=False,
            )
        if event.get("family") == "encoder":
            return Rendered(
                f"   new deployment checkpoint: POS {event.get('score')}"
                f" at step {event.get('step')}",
                live=False,
            )
        if event.get("family") == "dihya":
            worst_grp = event.get("worst_group")
            w_cls = event.get("worst_class")
            grp_str = f" [{worst_grp}]" if worst_grp else ""
            cls_str = f" class [{w_cls}]" if w_cls else ""
            return Rendered(
                f"   new dihya checkpoint: score {event.get('score')}{grp_str}{cls_str} "
                f"at step {event.get('step')}",
                live=False,
            )
        loss = event.get("validation_loss")
        chosen_on = (
            f"validation loss {float(str(loss)):.4f}"
            if loss is not None
            else f"Kabyle bpc {float(str(event['kab_bpc'])):.3f}"
        )
        return Rendered(
            f"   new best checkpoint: {chosen_on} at step {event.get('step')}",
            live=False,
        )

    def on_probe(self, event: dict[str, object]) -> Rendered:
        texts = event.get("texts")
        lines = [f"   samples at step {event.get('step')}:"]
        if isinstance(texts, dict):
            if event.get("family") == "encoder":
                lines.extend(
                    f"     {p} => {str(t).replace(chr(10), ' ')}" for p, t in texts.items()
                )
            else:
                lines.extend(f"     {short(p)!s} => {short(t)}" for p, t in texts.items())
        return Rendered("\n".join(lines), live=False)

    def on_checkpoint(self, event: dict[str, object]) -> Rendered:
        return Rendered(f"   checkpoint {event.get('path')}", live=False)

    def on_diagnostics(self, event: dict[str, object]) -> Rendered:
        activations = event.get("activation_rms")
        values = (
            [float(value) for value in activations if isinstance(value, int | float)]
            if isinstance(activations, list)
            else []
        )
        scale = f"{min(values):.3f}–{max(values):.3f}" if values else "unmeasured"
        return Rendered(
            f"   encoder health at step {event.get('step')}: layer activation RMS {scale}",
            live=False,
        )

    def on_result(self, event: dict[str, object]) -> Rendered | None:
        if event.get("family") == "restoration":
            return Rendered(
                "== restoration checks passed"
                if event.get("passed") is True
                else "== restoration task completed",
                live=False,
            )
        if event.get("family") == "encoder":
            return (
                Rendered("== encoder checks passed", live=False)
                if event.get("passed") is True
                else None
            )
        if event.get("family") == "dihya":
            return Rendered(
                f"== dihya smoke passed ({event.get('rate')} seq/s)"
                if event.get("passed") is True
                else "== dihya task completed",
                live=False,
            )
        return Rendered(str(event), live=False)

    def on_end(self, event: dict[str, object]) -> Rendered:
        if event.get("family") in {"codec_tts", "codec_student"}:
            return Rendered(
                f"== codec TTS ended ({event.get('ended')}) at step {event.get('step')}"
                f" | {event.get('updates_this_run')} updates this invocation"
                f" | best CE {event.get('best_ce')} | best speech {event.get('best_speech')}",
                live=False,
            )
        rate = event.get("tokens_per_second")
        speed = f"{rate:,.0f} tok/s" if isinstance(rate, int | float) else "-"
        if event.get("family") == "restoration":
            return Rendered(
                f"== restoration ended ({event.get('ended')}) at step {event.get('step')}: {speed},"
                f" best selection score {event.get('best_score')}",
                live=False,
            )
        if event.get("family") == "encoder":
            return Rendered(
                f"== encoder ended ({event.get('ended')}) at step {event.get('step')}: {speed},"
                f" best MLM CE {event.get('best_ce')}, best POS {event.get('best_pos')}",
                live=False,
            )
        if event.get("family") == "dihya":
            return Rendered(
                f"== dihya ended ({event.get('ended')}) at step {event.get('step')}: {speed},"
                f" best score {event.get('best_score')}",
                live=False,
            )
        return Rendered(
            f"== run ended ({event.get('ended')}) at step {event.get('step')}: {speed},"
            f" best Kabyle bpc {event.get('best_kab_bpc')} at step {event.get('best_step')}",
            live=False,
        )

    def on_save_best(self, event: dict[str, object]) -> None:
        del event

    def on_config_matches(self, event: dict[str, object]) -> Rendered:
        return Rendered(f"   resuming: config matches ({event.get('hash')})", live=False)

    def on_heldout_scored(self, event: dict[str, object]) -> Rendered:
        return Rendered(
            f"   held-out {event.get('language')}: {event.get('documents')} documents scored",
            live=False,
        )

    def on_retention_task(self, event: dict[str, object]) -> Rendered:
        return Rendered(f"-> lm-eval {event.get('task')} ({event.get('shots')}-shot)", live=False)


class Printer:
    """Writes rendered lines; on a terminal a live line rewrites itself in place."""

    def __init__(self, write: Callable[[str], object], *, terminal: bool) -> None:
        self.write = write
        self.terminal = terminal
        self.showing_live = False

    def show(self, rendered: Rendered | None) -> None:
        if rendered is None:
            return
        out = self.write
        if rendered.live:
            if self.terminal:
                out("\r\x1b[2K" + rendered.text)
                self.showing_live = True
            return
        prefix = "\r\x1b[2K" if self.showing_live else ""
        out(prefix + rendered.text + "\n")
        self.showing_live = False


def summary(lines: list[str], keep: int, baseline: str = "") -> list[str]:
    """The last `keep` permanent lines of a log, then where the run stands right now."""
    view = View()
    view.seed(baseline)
    permanent: list[str] = []
    live = ""
    for line in lines:
        rendered = view.render(line)
        if rendered is None:
            continue
        if rendered.live:
            live = rendered.text
        else:
            permanent.append(rendered.text)
            live = ""
    return permanent[-keep:] + ([live.strip()] if live else [])
