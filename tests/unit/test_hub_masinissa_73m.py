"""Hold the Masinissa-73M Hub port to exact mathematical parity with the native Encoder."""

from __future__ import annotations

import torch

from agbalu.encoder.configuration import Geometry
from agbalu.encoder.model import Encoder
from agbalu.hub.masinissa_73m.configuration_masinissa import MasinissaConfig
from agbalu.hub.masinissa_73m.modeling_masinissa import MasinissaForMaskedLM


def _build_paired_models() -> tuple[Encoder, MasinissaForMaskedLM]:
    torch.manual_seed(20261106)
    geometry = Geometry(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        heads=4,
        layers=4,
        block_layers=2,
        buckets=16,
        context_tokens=128,
        dropout=0.0,
        backend="reference",
    )
    native = Encoder(geometry).eval()
    config = MasinissaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=4,
        block_layers=2,
        position_buckets=16,
        max_position_embeddings=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    hub = MasinissaForMaskedLM(config)
    hub.train(False)

    # Populate non-zero depth routing queries so depth-attention scaling is exercised
    with torch.no_grad():
        for name, param in native.named_parameters():
            if name.endswith(".query"):
                param.normal_(mean=0.0, std=0.5)

    state = {}
    for key, value in native.state_dict().items():
        if key.startswith(("head.", "head_norm.", "vocabulary_bias")):
            state[key] = value.clone()
        else:
            state[f"masinissa.{key}"] = value.clone()
    hub.load_state_dict(state, strict=True)
    return native, hub


def test_masinissa_73m_hub_matches_native_encoder_and_is_batch_invariant() -> None:
    native, hub = _build_paired_models()
    input_ids = torch.tensor(
        [
            [2, 14, 66, 4, 64, 194, 31, 3],
            [2, 51, 32, 4, 99, 3, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        nat_hidden = native.contextualise(input_ids, attention_mask)
        nat_logits = native.logits(nat_hidden)

        hub_out = hub(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        assert hub_out.hidden_states is not None
        hub_hidden = hub_out.hidden_states[-1]
        hub_logits = hub_out.logits

        # Verify row 1 inside a batch vs passed alone (no q[0]/k[0] cross-row leakage)
        hub_single = hub(
            input_ids=input_ids[1:2, :6],
            attention_mask=attention_mask[1:2, :6],
            output_hidden_states=True,
        )
        assert hub_single.hidden_states is not None

    assert torch.allclose(nat_hidden, hub_hidden, atol=1e-5)
    # Compare valid positions for logits
    valid = attention_mask.bool()
    assert torch.allclose(nat_logits[valid], hub_logits[valid], atol=1e-4)
    assert torch.allclose(hub_hidden[1:2, :6], hub_single.hidden_states[-1], atol=1e-5)
    assert torch.allclose(hub_logits[1:2, :6], hub_single.logits, atol=1e-4)
