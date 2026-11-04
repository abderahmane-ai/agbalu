"""Match accumulated codec gradients to a concatenated reference."""

import torch

from agbalu.codec_tts.layout import CODEBOOKS, VOCABULARY
from agbalu.codec_tts.objective import codec_loss


def test_unequal_microbatch_gradient_matches_token_sum() -> None:
    torch.manual_seed(4)
    weight = torch.randn((3, CODEBOOKS * VOCABULARY), requires_grad=True)
    inputs = torch.randn((5, 3))
    labels = torch.zeros((5, CODEBOOKS), dtype=torch.long)
    reference = codec_loss((inputs @ weight).reshape(5, CODEBOOKS, VOCABULARY), labels)
    torch.autograd.backward(reference.numerator / reference.count)
    assert weight.grad is not None
    expected = weight.grad.detach().clone()
    weight.grad.zero_()
    for start, end in ((0, 1), (1, 5)):
        loss = codec_loss(
            (inputs[start:end] @ weight).reshape(end - start, CODEBOOKS, VOCABULARY),
            labels[start:end],
        )
        torch.autograd.backward(loss.numerator / (5 * CODEBOOKS))
    assert weight.grad is not None
    torch.testing.assert_close(weight.grad, expected)
