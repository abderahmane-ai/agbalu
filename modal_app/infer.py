"""Fill-mask inference against a checkpoint on the volume.

Short and CPU-only by default: a single 31M-parameter forward pass over one sentence does
not need a GPU, and requesting one while the pretraining run holds an A10 only queues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

from agbalu.model.config import DEFAULT_RUN_NAME
from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    VOLUMES,
    app,
    image,
)

TOKENIZER: Final = "agbalu-tok-base-16k.model"

DEFAULT_TEXT: Final = "Amɣar amussnaw n taddart yeffeɣ-d seg [MASK] ameqqran."

log: Final = logging.getLogger("agbalu.model")


@app.function(image=image, volumes=VOLUMES, timeout=10 * 60)
def predict(text: str = DEFAULT_TEXT, top_k: int = 5, checkpoint: str = "") -> list[dict[str, Any]]:
    """Ranked candidates per `[MASK]`. Prefers `best.pt`, falls back to `latest.pt`."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    from agbalu.model.checkpoint import BEST, LATEST
    from agbalu.model.config import ModelError
    from agbalu.model.infer import fill_mask, load_encoder
    from agbalu.tokenizer.evaluate import load

    run = Path(CHECKPOINT_PATH) / DEFAULT_RUN_NAME
    names = [checkpoint] if checkpoint else [BEST, LATEST]
    chosen = next((n for n in names if (run / n).is_file()), None)
    if chosen is None:
        message = f"no checkpoint among {names} in {run}; has the run written one yet?"
        raise ModelError(message)

    log.info("fill-mask | %s | %s", run / chosen, text)
    model = load_encoder(run, "kab", name=chosen)
    predictions = fill_mask(text, model, load(Path(DATA_PATH) / TOKENIZER), top_k=top_k)
    for prediction in predictions:
        best = prediction.candidates[0]
        log.info(
            "  position %d -> %r (%.1f%%)", prediction.index, best.piece, best.probability * 100
        )
    return [
        {
            "index": prediction.index,
            "candidates": [
                {"piece": c.piece, "probability": round(c.probability, 6)}
                for c in prediction.candidates
            ],
        }
        for prediction in predictions
    ]


@app.local_entrypoint()
def fill_mask(text: str = DEFAULT_TEXT, top_k: int = 5, checkpoint: str = "") -> None:
    print(f"\n{text}\n")
    for prediction in predict.remote(text=text, top_k=top_k, checkpoint=checkpoint):
        print(f"[MASK] at position {prediction['index']}")
        for rank, candidate in enumerate(prediction["candidates"], 1):
            print(f"  {rank}. {candidate['piece']!r}  {candidate['probability'] * 100:.2f}%")
