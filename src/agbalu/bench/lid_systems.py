"""The public language identifiers scored by task 7.4.

Both are fastText classifiers, and both are `reference` tier: scored, never
redistributed, never ingested. `fasttext` is imported inside the loader so the
gate (`make check`) does not need the models extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from agbalu.acquire.storage import sha256_file

GLOTLID_MODEL: Final = Path("data/raw/hf.glotlid-model/model.bin")
NLLB_LID_MODEL: Final = Path("data/raw/hf.nllb-lid218e/model.bin")

LABEL_PREFIX: Final = "__label__"

SYSTEMS: Final[dict[str, Path]] = {
    "glotlid": GLOTLID_MODEL,
    "nllb-lid218e": NLLB_LID_MODEL,
}
"""Identifier name to the model file `make acquire TASK=siblings` lands."""


class LidModelError(Exception):
    """A model file is missing, or fastText is not installed."""


class _FastTextModel(Protocol):
    @property
    def labels(self) -> list[str]: ...

    def predict(self, text: str, k: int) -> tuple[tuple[str, ...], Any]: ...


def _load(path: Path) -> _FastTextModel:
    if not path.is_file():
        msg = (
            f"model not found: {path}. Fetch it with "
            f"`python -m agbalu.acquire.cli fetch --id hf.glotlid-model --id hf.nllb-lid218e`"
        )
        raise LidModelError(msg)
    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = "fasttext is not installed; install the `models` extra"
        raise LidModelError(msg) from exc
    # fastText writes a deprecation warning to stderr on load and offers no flag.
    fasttext.FastText.eprint = lambda *_args, **_kwargs: None
    return cast(_FastTextModel, fasttext.load_model(str(path)))


@dataclass(slots=True)
class FastTextIdentifier:
    """One fastText language identifier, scored as a closed-set classifier.

    Labels are returned verbatim after stripping `__label__`. No remapping: a
    system answering `ber_Latn` or `zgh_Tfng` for Kabyle has made a real error,
    and normalising it into a hit would erase the confusion being measured.
    """

    _name: str
    path: Path
    _model: _FastTextModel | None = None
    _revision: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        if not self._revision:
            digest, _ = sha256_file(self.path)
            self._revision = digest[:12]
        return self._revision

    @property
    def labels(self) -> frozenset[str]:
        """Every label the model can emit, `__label__` stripped."""
        return frozenset(x.removeprefix(LABEL_PREFIX) for x in self._loaded().labels)

    def _loaded(self) -> _FastTextModel:
        if self._model is None:
            self._model = _load(self.path)
        return self._model

    def identify(self, texts: Sequence[str]) -> list[str]:
        model = self._loaded()
        labels: list[str] = []
        for text in texts:
            # fastText treats a newline as a document boundary and raises on it.
            predicted, _ = model.predict(text.replace("\n", " "), 1)
            labels.append(predicted[0].removeprefix(LABEL_PREFIX) if predicted else "")
        return labels


def build(name: str) -> FastTextIdentifier:
    """The identifier registered under `name`.

    Raises:
        LidModelError: `name` is not a known system.
    """
    path = SYSTEMS.get(name)
    if path is None:
        known = ", ".join(sorted(SYSTEMS))
        msg = f"unknown LID system {name!r}; known systems are {known}"
        raise LidModelError(msg)
    return FastTextIdentifier(_name=name, path=path)
