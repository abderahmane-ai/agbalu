"""Verify native restoration heads are independent of batch neighbors."""

import copy

import torch

from agbalu.restoration.dataset import Example, collate
from agbalu.restoration.model import Restorer, joint_probabilities
from agbalu.restoration.objective import loss_sums, target_counts


def test_batched_and_single_rows_agree(model: Restorer, example: Example) -> None:
    batch = collate([example, example], 64)
    alone = collate([example], 64)
    with torch.inference_mode():
        expected = model.forward(alone)
        produced = model.forward(batch)
    torch.testing.assert_close(expected.punctuation[0], produced.punctuation[1])
    probabilities = joint_probabilities(produced)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))


def test_unused_mlm_weights_are_absent_and_embeddings_train(model: Restorer) -> None:
    names = dict(model.named_parameters())
    assert "encoder.head.weight" not in names
    assert "encoder.vocabulary_bias" not in names
    assert names["encoder.embedding.weight"].requires_grad


def test_finite_gradients_reach_the_pretrained_body(model: Restorer, example: Example) -> None:
    batch = collate([example], 64)
    output = model.forward(batch)
    loss = output.punctuation.square().mean() + output.casing.square().mean()
    torch.autograd.backward(loss)
    gradient = model.encoder.embedding.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0
    assert bool(torch.isfinite(gradient).all())


def test_packed_and_separate_windows_have_the_same_gradients(
    model: Restorer, example: Example
) -> None:
    separate = copy.deepcopy(model)
    packed_batch = collate([example] * 10, 64)
    counts = target_counts(packed_batch)
    loss = loss_sums(model.forward(packed_batch), packed_batch).normalized(counts)
    torch.autograd.backward(loss)
    for _ in range(10):
        single = collate([example], 64)
        torch.autograd.backward(loss_sums(separate.forward(single), single).normalized(counts))
    for (name, expected), (_, actual) in zip(
        separate.named_parameters(), model.named_parameters(), strict=True
    ):
        if not expected.requires_grad:
            assert actual.grad is None
            continue
        assert expected.grad is not None, name
        assert actual.grad is not None, name
        torch.testing.assert_close(expected.grad, actual.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_casing_gradients_reach_shared_encoder_backbone(model: Restorer, example: Example) -> None:
    batch = collate([example], 64)
    scores = model.forward(batch)
    torch.autograd.backward(scores.casing.square().sum())
    assert model.encoder.embedding.weight.grad is not None
    assert float(model.encoder.embedding.weight.grad.abs().sum()) > 0
    assert model.punctuation.weight.grad is None
    assert model.casing[0].weight.grad is not None


def test_neighbor_changes_cannot_change_real_component_predictions(
    model: Restorer, example: Example
) -> None:
    batch = collate([example, example], 64)
    valid = batch.case_spans[1, :, 1] > batch.case_spans[1, :, 0]
    with torch.inference_mode():
        before = model.forward(batch)
        batch.ids.masked_fill_(batch.documents == 0, 11)
        after = model.forward(batch)
    torch.testing.assert_close(
        before.casing[1, valid], after.casing[1, valid], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        before.punctuation[1, :4], after.punctuation[1, :4], atol=1e-5, rtol=1e-5
    )
    assert float((before.casing[0, valid] - after.casing[0, valid]).abs().max()) > 0
