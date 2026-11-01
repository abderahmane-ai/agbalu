"""Supply small deterministic restoration tensors."""

import numpy as np
import pytest
import torch

from agbalu.encoder.configuration import Geometry
from agbalu.restoration.configuration import Settings
from agbalu.restoration.dataset import Example
from agbalu.restoration.model import Restorer


@pytest.fixture
def settings() -> Settings:
    return Settings(
        geometry=Geometry(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            layers=2,
            heads=2,
            block_layers=1,
            buckets=8,
            context_tokens=64,
            dropout=0,
            backend="reference",
        ),
        core_tokens=32,
        left_tokens=15,
        streaming_core_tokens=16,
        lookahead_tokens=8,
        micro_windows=1,
        batch_windows=2,
        compile=False,
        workers=0,
        require_smoke=False,
        minimum_words=3,
    )


@pytest.fixture
def model(settings: Settings) -> Restorer:
    torch.manual_seed(17)
    return Restorer(settings.geometry, ("@", "@»", "@«")).eval()


@pytest.fixture
def example() -> Example:
    return Example(
        np.asarray([2, 6, 7, 8, 9, 3], np.int64),
        np.asarray([[1, 3], [3, 4], [4, 5]], np.int64),
        np.asarray([0, 1, 0, 2], np.int64),
        np.asarray([1, 0, 0], np.int64),
        np.asarray([0, 0, 0, 0], np.int64),
        (True, True),
    )
