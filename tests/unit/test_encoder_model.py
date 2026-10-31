"""Verify the complete encoder on isolated documents."""

import torch

from agbalu.encoder.attention import make_layout
from agbalu.encoder.configuration import Geometry
from agbalu.encoder.model import Encoder


def tiny() -> Encoder:
    torch.manual_seed(3)
    return Encoder(
        Geometry(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            layers=2,
            block_layers=1,
            heads=2,
            dropout=0,
            backend="reference",
        )
    )


def test_packed_documents_equal_separate_documents() -> None:
    model = tiny().eval()
    ids = torch.tensor([2, 6, 7, 3, 2, 8, 9, 3])
    docs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    positions = torch.arange(4).repeat(2)
    layout = make_layout(docs, positions, torch.ones(8, dtype=torch.bool), flex=False)
    packed = model.contextualise_packed(ids, layout)
    separate = torch.cat(
        [
            model.contextualise(ids[i : i + 4][None], torch.ones(1, 4, dtype=torch.bool))[0]
            for i in (0, 4)
        ]
    )
    torch.testing.assert_close(packed, separate, atol=1e-6, rtol=1e-5)
    changed = ids.clone()
    changed[5] = 15
    torch.testing.assert_close(model.contextualise_packed(changed, layout)[:4], packed[:4])


def test_every_relative_path_receives_a_gradient() -> None:
    model = tiny()
    ids = torch.tensor([2, 6, 7, 3])
    layout = make_layout(
        torch.zeros(4, dtype=torch.long),
        torch.arange(4),
        torch.ones(4, dtype=torch.bool),
        flex=False,
    )
    loss, correct = model.forward(ids, layout, torch.tensor([-100, 6, 7, -100]))
    assert correct.item() <= 2
    assert torch.isfinite(loss)
    torch.autograd.backward(loss)
    for parameter in [
        model.relative,
        model.position_q.weight,
        model.position_k.weight,
        model.embedding.weight,
    ]:
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_padding_does_not_change_real_outputs() -> None:
    model = tiny().eval()
    clean = model.contextualise(torch.tensor([[2, 6, 3]]), torch.ones(1, 3, dtype=torch.bool))
    padded = model.contextualise(
        torch.tensor([[2, 6, 3, 0]]), torch.tensor([[True, True, True, False]])
    )
    torch.testing.assert_close(clean, padded[:, :3])
