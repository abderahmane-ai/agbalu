"""Shared Modal objects: the app, the image, and the volumes."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final

import modal

APP_NAME: Final = "agbalu"


def call_owner() -> str:
    """The run-lock owner for a Modal function: the *call* id, not the container's task id.

    Modal restarts a preempted container with the same input and a **new task id**, so a
    task-keyed lock refuses its own retry: the repack was preempted at shard 1, and every one
    of the five retries then died on `RunLockedError` against a lock 11 to 15 minutes old,
    with `STALE_AFTER` at 45. The job cannot make progress and the container is billed for
    failing. `asr_finetune` carries the same lock and the same `retries`, so a preemption
    seventeen hours into training would have done the same thing.

    A call id is stable across Modal's own retries and differs between two separate launches,
    which is precisely the distinction the lock exists to draw: refuse a second operator
    launch, allow the same job to resume. `agbalu.model.lock` stays free of any modal import,
    so the owner is supplied here rather than resolved in there.
    """
    from agbalu.model.lock import default_owner

    call_id = modal.current_function_call_id()
    return f"call-{call_id}" if call_id else default_owner()


PYTHON_VERSION: Final = "3.12"
TORCH_VERSION: Final = "2.11.0"

DATA_VOLUME_NAME: Final = "agbalu-data"
CHECKPOINT_VOLUME_NAME: Final = "agbalu-checkpoints"
MODELS_VOLUME_NAME: Final = "agbalu-models"

DATA_PATH: Final = PurePosixPath("/data")
CHECKPOINT_PATH: Final = PurePosixPath("/checkpoints")
MODELS_PATH: Final = PurePosixPath("/models")

app: Final = modal.App(APP_NAME)

data_volume: Final = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume: Final = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
models_volume: Final = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

VOLUMES: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]] = {
    str(DATA_PATH): data_volume,
    str(CHECKPOINT_PATH): checkpoint_volume,
}
"""Spelt out rather than inferred: `App.function` takes the union in both key and value,
and `dict` is invariant in both, so a narrower inferred type does not satisfy it."""

PIP_PACKAGES: Final[tuple[str, ...]] = (
    f"torch=={TORCH_VERSION}",
    "numpy>=2.0",
    "sentencepiece>=0.2",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pyarrow>=17.0",
)
"""Every third party the container's import graph reaches. `pyyaml` and `pydantic` are
transitive — `agbalu.model.data` imports `agbalu.tokenizer.spec`, which reaches
`agbalu.normalise.rules`. `tests/unit/test_modal_image.py` recomputes the closure and
fails if this list drifts from it."""

IMPORT_NAME: Final[dict[str, str]] = {"pyyaml": "yaml"}
"""Distribution names that differ from the module they provide."""

REMOTE_FLORES: Final = "bench/flores"
"""Where `modal-upload` puts FLORES+ on the data volume. Here rather than in `bench`, so
`pilot` can read the English devtest as an out-of-domain probe without importing a module
whose own imports reach pyarrow."""

RESOURCES: Final = Path("resources")
"""`normalise.rules` resolves `resources/homoglyphs.yaml` relative to the working
directory, which is `/root` in the container."""

image: Final = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PIP_PACKAGES)
    # Both packages, not the 4.1 GB of data/ or artifacts/ beside them. `modal_app` has
    # to be here too: the entrypoint file alone lands at /root/train.py, so `modal_app.common`
    # is unresolvable in the container unless the package itself is shipped.
    .add_local_python_source("agbalu", "modal_app")
    .add_local_dir(RESOURCES, remote_path=f"/root/{RESOURCES}")
)

TIFINAGH_PIP_PACKAGES: Final[tuple[str, ...]] = (*PIP_PACKAGES, "safetensors>=0.4")

tifinagh_image: Final = image.pip_install(*TIFINAGH_PIP_PACKAGES[len(PIP_PACKAGES) :])
"""One layer on the training image rather than a fourth image of its own.

`safetensors` is what `Transliterator.load` reads a *published* directory with, which is
the path the release is loaded through — the closure test is what caught that the base
image does not carry it, before a GPU container did. Derived rather than added to
`PIP_PACKAGES`, which would invalidate the encoder's training image for a package it
never imports."""


BENCH_VOLUMES: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]] = {
    str(DATA_PATH): data_volume,
    str(MODELS_PATH): models_volume,
    str(CHECKPOINT_PATH): checkpoint_volume.read_only(),
}
"""The checkpoint volume is read-only here: `--weights` scores a fine-tune from it, and a
scoring job has no business writing training state back."""

BENCH_PIP_PACKAGES: Final[tuple[str, ...]] = (
    f"torch=={TORCH_VERSION}",
    "transformers>=4.44",
    # Reached through `agbalu.model.data`, which the sentiment benchmark imports to load the
    # encoder. Transitive through torch, named because the closure test is what makes it
    # checkable — and it was absent when that benchmark was first written.
    "numpy>=2.0",
    "sacrebleu>=2.4",
    "sentencepiece>=0.2",
    "pyarrow>=17.0",
    # Named rather than left to arrive with transformers: `resolve_weights` calls
    # `snapshot_download` itself when the checkpoint volume has no fine-tune on it, and the
    # closure test is what caught the same omission in the ASR image.
    "huggingface_hub>=0.30",
    "pyyaml>=6.0",
    "pydantic>=2.0",
)
"""Enumerated rather than extended from `PIP_PACKAGES`.

`sentencepiece` is needed at runtime but imported by neither package's public API — the
OPUS-MT tokenizers load `source.spm`/`target.spm`, and sacrebleu's `flores200` tokenizer
(spBLEU) does `import sentencepiece`. `tests/unit/test_modal_image.py` records it as a
runtime-only dependency, since an AST closure over our own code cannot see either use."""

bench_image: Final = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*BENCH_PIP_PACKAGES)
    # HF_HOME on the volume so a second run reuses the weights instead of refetching
    # ~20 GB. Baking them into the image instead would refetch on every rebuild.
    .env({"HF_HOME": str(MODELS_PATH), "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("agbalu", "modal_app")
    .add_local_dir(RESOURCES, remote_path=f"/root/{RESOURCES}")
)
"""Separate from `image` on purpose: adding these layers to the training image would
invalidate it and force a rebuild between a smoke and the run it was measuring."""

MT_VOLUMES: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]] = {
    str(DATA_PATH): data_volume,
    str(CHECKPOINT_PATH): checkpoint_volume,
    str(MODELS_PATH): models_volume,
}
"""All three: the corpus is on data, the trimmed base on models, the run on checkpoints."""

MT_PIP_PACKAGES: Final[tuple[str, ...]] = (
    f"torch=={TORCH_VERSION}",
    "transformers>=4.44",
    "accelerate>=0.34",
    "sentencepiece>=0.2",
    "pyarrow>=17.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
)
"""`accelerate` is what `Trainer` uses to place the model and step the optimiser, imported
by transformers rather than by us. The last three arrive through package `__init__` files
rather than through anything this entrypoint asks for: `agbalu.mt.data` needs one frozenset
from `agbalu.parallel.quality`, and importing it executes `agbalu/parallel/__init__.py`,
whose reach includes the parquet readers, the registry schema and the normaliser."""

mt_image: Final = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*MT_PIP_PACKAGES)
    .env({"HF_HOME": str(MODELS_PATH), "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("agbalu", "modal_app")
    # `normalise.rules` is in the closure and resolves this at import time.
    .add_local_dir(RESOURCES, remote_path=f"/root/{RESOURCES}")
)

LLM_VOLUMES: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]] = {
    str(DATA_PATH): data_volume,
    str(MODELS_PATH): models_volume,
}
"""The evaluation sets, FLORES+ and the results are on data; the 10.25 GB base checkpoint
is cached on models. Nothing here writes a checkpoint, so that volume is not mounted."""

LLM_PIP_PACKAGES: Final[tuple[str, ...]] = (
    f"torch=={TORCH_VERSION}",
    # 5.12 is the floor, not decoration: `transformers.models.qwen3_5` defines the base, and
    # an older resolution installs a library that cannot load it. Its Gated DeltaNet fast path wants
    # `flash-linear-attention` and `causal-conv1d`; scoring falls back to
    # `torch_chunk_gated_delta_rule` with a warning, so they are a training-image decision —
    # `causal-conv1d` ships an sdist only and compiles under nvcc.
    "transformers>=5.12",
    # Reached through `modal_app.bench`, which `modal_app.llm` imports for the results shape:
    # `resolve_weights` there calls `snapshot_download`. Transitive through transformers
    # anyway, named because the closure test is what makes that checkable.
    "huggingface_hub>=0.30",
    # `device_map` loads the 10.25 GB checkpoint straight onto the GPU, and it is
    # `accelerate` that transformers dispatches to in order to do it.
    "accelerate>=0.34",
    "sacrebleu>=2.4",
    "sentencepiece>=0.2",
    "pyarrow>=17.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
)
"""`agbalu.bench.mt` scores the translation prompts, so the last four arrive the same way
they do in the bench image — through `agbalu/bench/__init__.py` and the normaliser."""

llm_image: Final = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*LLM_PIP_PACKAGES)
    .env(
        {
            "HF_HOME": str(MODELS_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Scoring alternates one wide logits allocation with narrow fp32 slices, which
            # is the shape that fragments a caching allocator: the first OOM here failed
            # with 2.45 GiB reserved but unallocated.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source("agbalu", "modal_app")
    .add_local_dir(RESOURCES, remote_path=f"/root/{RESOURCES}")
)
"""Its own image rather than `bench_image`: the transformers floor above would invalidate
the bench and synthesis images, which do not need it, for a rebuild neither asked for."""

JUGURTHA_VOLUMES: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]] = {
    str(DATA_PATH): data_volume,
    str(MODELS_PATH): models_volume,
    str(CHECKPOINT_PATH): checkpoint_volume,
}
"""The corpus and the packed blocks are on data, the base is cached on models, the run
writes to checkpoints."""

JUGURTHA_PIP_PACKAGES: Final[tuple[str, ...]] = (
    f"torch=={TORCH_VERSION}",
    "transformers>=5.12",
    "numpy>=2.0",
    # `AutoTokenizer` resolves the base's tokenizer through it, and `from_pretrained`
    # dispatches device placement through accelerate.
    "sentencepiece>=0.2",
    "accelerate>=0.34",
    # 18 of the base's 24 layers are Gated DeltaNet. Without these two, transformers falls
    # back to `torch_chunk_gated_delta_rule` and logs a warning rather than failing, so the
    # cost of omitting them is a slow paid run and not an error.
    "flash-linear-attention>=0.5",
    "causal-conv1d>=1.6",
    # The 248,320-token head makes the fp32 logit surface, not the weights, what caps the
    # micro-batch. `agbalu.llm.loss` refuses to fall back to materialising it on CUDA.
    "cut-cross-entropy>=25.1",
)

jugurtha_image: Final = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    # `causal-conv1d` ships an sdist only and compiles a CUDA extension, so the toolchain
    # has to be in the image before pip runs. `flash-linear-attention` is Triton and needs
    # neither, but it is installed in the same layer so one rebuild covers both.
    .apt_install("git", "build-essential", "ninja-build")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6", "MAX_JOBS": "4"})
    .pip_install(f"torch=={TORCH_VERSION}")
    .pip_install(*JUGURTHA_PIP_PACKAGES)
    .env(
        {
            "HF_HOME": str(MODELS_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # NCCL over PCIe between two A10s in one container. Peer-to-peer is not
            # available on this pairing and probing for it stalls the first all-reduce.
            "NCCL_P2P_DISABLE": "1",
        }
    )
    .add_local_python_source("agbalu", "modal_app")
    .add_local_dir(RESOURCES, remote_path=f"/root/{RESOURCES}")
)
"""Torch first and alone: `causal-conv1d` imports torch inside its `setup.py`, so a single
`pip_install` that lists both resolves them in one pass and the build sees no torch.

8.6 is the A10's compute capability. Left unset, the extension builds for every architecture
the toolkit knows, which is many minutes of nvcc for code no container here will run."""

ASR_VOLUMES: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]] = {
    str(DATA_PATH): data_volume,
    str(CHECKPOINT_PATH): checkpoint_volume,
    str(MODELS_PATH): models_volume,
}
"""The 7.89 GiB of Common Voice audio and the built splits are on data, the SSL encoder
is cached on models, and the fine-tune writes to checkpoints."""

ASR_CPU: Final = 8.0
"""Modal's default request is 0.125 cores, which is what the first fine-tune ran on.

Every clip is an mp3 decode and a 32 kHz → 16 kHz soxr resample, ~185 of them per
optimiser step, and the first run measured **304 ms per clip in training against 315 ms
in validation** — no backward pass, same cost — while the card sat at 1.08 TFLOP/s of a
125 TFLOP/s peak. The audio path, not the model, is what that container was spending its
time on.

Eight cores took the step from 55.5 s to 7.45. **Twelve with twenty-four threads made it
8.3 — worse — so this does not go up**; see `LOADER_THREADS`. A core is $0.047/hour
against the A10's $1.10, and the run is charged for every core it asks for."""

ASR_PIP_PACKAGES: Final[tuple[str, ...]] = (
    f"torch=={TORCH_VERSION}",
    "transformers>=5.12",
    "accelerate>=0.34",
    # Asked for by name rather than left to arrive with transformers: `fetch` calls
    # `snapshot_download` itself, and the closure test is what caught the omission.
    "huggingface_hub>=0.30",
    # Common Voice ships mp3. libsndfile gained mp3 in 1.1.0 and the wheel bundles
    # 1.2.2, so this decodes the clips with no ffmpeg in the image.
    "soundfile>=0.13",
    # `Wav2Vec2FeatureExtractor` takes 16 kHz; Common Voice mp3 is 32 kHz or 48 kHz,
    # and `soxr` is what `librosa.resample` uses for the highest-quality path.
    "librosa>=0.11",
    "numpy>=1.26,<2.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pyctcdecode>=0.5.0",
    "https://github.com/kpu/kenlm/archive/master.zip",
)
"""No `datasets`: the mirror is a script-based repo, which `datasets` 4 will not execute,
and the audio is read straight out of the downloaded tar shards instead. WER and CER are
`agbalu.speech.metrics`, so there is no `jiwer` either — the normalisation policy is the
measurement and it is not delegated."""


def _audio_base(packages: tuple[str, ...]) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*packages)
        .env(
            {
                "HF_HOME": str(MODELS_PATH),
                "HF_XET_HIGH_PERFORMANCE": "1",
                # Duration bucketing makes every batch a different shape, which is the shape
                # that fragments a caching allocator: the first OOM here failed with 12.39 GiB
                # reserved but unallocated against a 1.95 GiB request.
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
    )


def _with_local_sources(base: modal.Image) -> modal.Image:
    return base.add_local_python_source("agbalu", "modal_app").add_local_dir(
        # `normalise.rules` resolves this at import time, and `speech.vocabulary` reads
        # `ALPHABET` from it to decide what may become a CTC class.
        RESOURCES,
        remote_path=f"/root/{RESOURCES}",
    )


asr_image: Final = _with_local_sources(_audio_base(ASR_PIP_PACKAGES))

TTS_PIP_PACKAGES: Final[tuple[str, ...]] = (*ASR_PIP_PACKAGES, f"torchaudio=={TORCH_VERSION}")
"""The ASR stack plus `torchaudio`, whose `compliance.kaldi.fbank` is the filterbank the
Sidon feature predictor was traced against. Pinned to `TORCH_VERSION`: torchaudio tracks
torch release for release, and a mismatched pair resolves torch backwards."""

tts_image: Final = _with_local_sources(
    _audio_base(ASR_PIP_PACKAGES).pip_install(f"torchaudio=={TORCH_VERSION}")
)
"""Layered on the ASR pip step rather than declared from `TTS_PIP_PACKAGES` in one
call, so the expensive layer — torch, transformers and kenlm from source — is the same
cache entry both images resolve to. And layered *before* the local sources, because
`add_local_*` defaults to `copy=False`, which adds the files at container start rather
than as an image layer: `asr_image.pip_install(...)` is a build step after a mount and
modal refuses it."""

STYLETTS2_REPO: Final = "https://github.com/semidark/StyleTTS2.git"
STYLETTS2_COMMIT: Final = "b1956da84bf4a6ccc88f2440078024f1c4bfec7d"
STYLETTS2_PATH: Final = PurePosixPath("/opt/StyleTTS2")
"""The patched StyleTTS2 the Kokoro recipe trains through, pinned to a commit rather than
to `main`.

`hexgrad/Kokoro` ships inference only, so the fine-tuning path is this fork
(Apache-2.0, verified live 2026-08-15). The patches that matter: the modules use torch's
`parametrizations.weight_norm`, whose state-dict keys differ from the legacy API, and
`text_utils` reads the Kokoro index assignment instead of StyleTTS2's own — using the wrong
one scrambles every pre-trained embedding. It also carries the ASR aligner, the JDC pitch
extractor and PL-BERT as real files, ~141 MB, so the clone is the whole dependency.

Pinned because an image built from a moving `main` is not the image that was smoked."""

MATOUB_PIP_PACKAGES: Final[tuple[str, ...]] = (
    "munch>=4.0",
    "pydub>=0.25",
    "nltk>=3.9",
    "matplotlib>=3.9",
    "einops>=0.8",
    "einops-exts>=0.0.4",
    "tensorboard>=2.18",
    # `meldataset` builds a DataFrame of the filelist and `train_first` is a click
    # command. Neither is in StyleTTS2's `requirements.txt`, which is why building the
    # image from that file alone put a container on a card and killed it at import.
    "pandas>=2.2",
    "click>=8.1",
    "monotonic-align @ git+https://github.com/resemble-ai/monotonic_align.git",
)
"""What the recipe actually imports beyond the audio stack, computed from the closure of
`train_first.py` and `train_second.py` rather than from its `requirements.txt`. `torch`,
`torchaudio`, `transformers`, `accelerate`, `librosa`, `soundfile`, `numpy` and `pyyaml`
already arrive with the TTS image and are not repeated.

No `misaki` and no `espeak-ng`. It appears in that closure but only inside a function in
`kokoro_tb_utils`, and both scripts already handle its absence — it powers TensorBoard
sample synthesis, and this project's front end is `agbalu.tts.g2p`, which emits IPA
directly. Leaving it out also avoids the failure its troubleshooting guide leads with, a
bundled espeak loader whose prebuilt binary carries CI paths."""

matoub_image: Final = _with_local_sources(
    _audio_base(ASR_PIP_PACKAGES)
    .pip_install(f"torchaudio=={TORCH_VERSION}")
    .apt_install("git")
    .pip_install(*MATOUB_PIP_PACKAGES)
    .run_commands(
        f"git clone {STYLETTS2_REPO} {STYLETTS2_PATH}",
        f"git -C {STYLETTS2_PATH} checkout {STYLETTS2_COMMIT}",
    )
    .env({"PYTHONPATH": str(STYLETTS2_PATH)})
)
"""Shares the ASR image's torch layer and the TTS image's torchaudio layer, then diverges.

`apt_install("git")` precedes the pip step because `monotonic-align` installs from a git
URL, and both precede `_with_local_sources`: `add_local_*` defaults to `copy=False`, which
mounts at container start rather than as a layer, so no build step may follow it."""

KENLM_PIP_PACKAGES: Final[tuple[str, ...]] = ("pyyaml>=6.0", "pydantic>=2.0")
"""`agbalu.normalise` is the whole import closure here — the corpus is read as JSONL and
written as plain text, and no model is loaded."""

kenlm_image: Final = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    # `pip install kenlm` builds the *query* extension only. `lmplz` and `build_binary`
    # are separate C++ programs that need cmake and Boost, which is why the binary this
    # project ships was first built by hand in a container that no longer exists.
    # `git` because `debian_slim` has none, and the clone below is the first thing that
    # runs: the build failed at step 1 with `git: not found` the first time anything
    # actually built this image, which was long after the model it produces was published.
    .apt_install("build-essential", "cmake", "git", "libboost-all-dev", "libbz2-dev", "liblzma-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/kpu/kenlm.git /opt/kenlm",
        "cmake -S /opt/kenlm -B /opt/kenlm/build -DCMAKE_BUILD_TYPE=Release",
        "cmake --build /opt/kenlm/build -j",
    )
    .env({"PATH": "/opt/kenlm/build/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install(*KENLM_PIP_PACKAGES)
    .add_local_python_source("agbalu", "modal_app")
    .add_local_dir(RESOURCES, remote_path=f"/root/{RESOURCES}")
)
"""Its own image because the toolchain is 900 MB of compilers that no other job needs, and
adding it to `asr_image` would invalidate the training image for a rebuild it did not ask
for. Built once and cached; the model is built once and lives on the volume."""

KENLM_CPU: Final = 8.0
KENLM_TIMEOUT: Final = 2 * 60 * 60
"""3,041,989 sentences at order 5. `lmplz` is I/O and sort bound, and `-S 60%` bounds what
it may take of the container's memory rather than letting it be killed for asking."""

BENCH_TIMEOUT: Final = 2 * 60 * 60

TTS_TIMEOUT: Final = 2 * 60 * 60
"""The baseline is 1,000 syntheses and two decodes of them, which is minutes. The
budget covers the first call on a cold volume, where the real audio of the prompt set
is a page nobody has read — the failure that cost ~$14 in Phase 5."""

TTS_CORPUS_TIMEOUT: Final = 12 * 60 * 60
"""Task 12.4 writes two arms of a voice, so the budget is dominated by neither the model
nor the decode: 35,428 mp3 reads and up to twice that many small wav writes onto the
volume. **A voice is one container**, so this bounds a single voice and the driver that
waits on both. The build also commits every `COMMIT_EVERY` batches and resumes at clip
granularity, so a preemption or timeout at any point loses at most that much and the next
spawn picks up where it left off — the timeout is a ceiling, not a restart cost."""

TTS_VOICE_CPU: Final = 16.0
"""One voice's container: an A10 and sixteen cores.

Everything the render does per clip outside the vocoder is C with the GIL released — a
volume fetch, an mp3 decode, a soxr resample, two wav writes — and all of it now runs on
threads *beside* the device rather than in front of it, so the cores are what decides
whether the A10 ever waits. `ASR_CPU`'s eight-thread wall was measured in the training
container, where the main thread also drives the GPU and holds the GIL to do it; here the
main thread issues one traced forward per sixteen clips and holds nothing.

Sixteen cores is $0.75/hour against the A10's $1.10 in the same container, so this is the
cheap half of the bill and starving it is what makes the expensive half idle."""

TTS_DRIVER_CPU: Final = 1.0
"""The driver holds a container only to fan the voices out and assemble their reports, so
it asks for a core rather than Modal's default eighth of one — and no GPU, which is the
point: waiting on `.map` from inside an A10 container bills the card to watch."""

LLM_TIMEOUT: Final = 4 * 60 * 60
"""Perplexity is minutes; the budget is the translation prompts, which decode one sentence
at a time over up to seven directions."""

JUGURTHA_TIMEOUT: Final = 23 * 60 * 60
"""One epoch is 17.1 hours of arithmetic on two A10s at the measured rate, plus the load and
the evaluations. Modal's ceiling is 24 hours, so an epoch is the largest unit that can be
bought at once — which is why `epochs` defaults to 1 and the run resumes rather than
restarting."""

JUGURTHA_PACK_CPU: Final = 8.0
"""Packing tokenises 5,254,022 documents. Modal's default request is an eighth of a core,
and the tokenizer is the whole job."""

ASR_FETCH_TIMEOUT: Final = 2 * 60 * 60
"""7.89 GiB of tar shards from the Hub, extracted on the volume."""

ASR_REPACK_CPU: Final = 16.0
"""The repack asks for twice `ASR_CPU` and no GPU, and both halves of that are deliberate.

`LOADER_THREADS`' eight-thread wall was measured *in the training container*, where the
main thread is also driving an A10 and holding the GIL to do it. Here there is no GPU
work to contend with, and the per-clip cost is a volume fetch plus an mp3 decode — the
fetch releases the GIL entirely and `soundfile`/`soxr` do the decode in C. So the ceiling
that applies during training does not apply here, and the rate this job gets is unmeasured
until it runs: it logs its own clips/second from the first shard so the operator sees it
in minutes rather than at the end.

A core is $0.047/hour, so sixteen is $0.75/hour against the $1.48 an A10 container costs
with eight — which is the whole point of doing this without a GPU attached."""

ASR_REPACK_TIMEOUT: Final = 6 * 60 * 60
"""Bounded by the cold-read rate, which is the thing being fixed and therefore unknown
here: 182,483 clips at the 276 ms/clip the cold run measured with eight threads would be
14 hours, and the same work at sixteen threads' worth of concurrency should be a fraction
of that. Six hours is not a prediction — the pack is written in shards of `SHARD_CLIPS`
and committed per shard, so a timeout loses at most one shard and the next call resumes
from the gap rather than from the start."""

ASR_TIMEOUT: Final = 20 * 60 * 60
"""144.31 hours of audio. The reference w2v-BERT fine-tune ran 10 epochs on a 16 GB V100;
the A10G is faster and this is one language, but the budget is deliberately under Modal's
24-hour ceiling so a resume has room rather than racing it."""

MT_TIMEOUT: Final = 12 * 60 * 60
"""Half the ceiling. 512k examples at the 600M's throughput is a few hours; the headroom
covers the 1.3B and a resumed retry."""

MATOUB_GPU: Final = "A10G"
MATOUB_CPU: Final = 8.0
"""Eight cores, for the same reason the ASR fine-tune asks for them: every sample is a
24 kHz WAV read plus a mel, and Modal's default request is an eighth of a core, which is
how a training step ends up costing what a validation step costs. Eight was the measured
wall on the ASR path and twelve made it worse; there is no reason to rediscover that."""

MATOUB_BATCH: Final = 4
"""Stage 2 is the constraint, not Stage 1. Its WavLM discriminator is what forced a 24 GB
RTX 3090 down from the recipe's batch of 12 to 4, reported by the recipe's own users; the
A10G is also 24 GiB. The smoke measures it before the run commits — this is the starting
point, not a finding."""

MATOUB_TIMEOUT: Final = 24 * 60 * 60

TRAIN_TIMEOUT: Final = 24 * 60 * 60
"""Modal's ceiling. The run is expected to finish inside it; the retry policy below is
what covers preemption, and resume is what makes a retry cheap rather than fatal."""

RETRIES: Final = modal.Retries(max_retries=5, backoff_coefficient=2.0, initial_delay=10.0)
"""A preempted container comes back and resumes from `latest.pt`. Retries are safe here
precisely because `Trainer.maybe_resume` makes the function idempotent in effect."""
