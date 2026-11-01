"""Verify full-size restoration accounting before accelerator allocation."""

from agbalu.encoder.configuration import Geometry
from agbalu.restoration.model import Restorer


def test_exact_full_size_parameter_ledger() -> None:
    model = Restorer(Geometry(backend="reference"), ("@", "@»", "@«"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 75_148_559
