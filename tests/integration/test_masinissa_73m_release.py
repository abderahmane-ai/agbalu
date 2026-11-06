"""Verify that staged Masinissa-73M loads cleanly into transformers alone."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

RELEASE_DIR = Path("artifacts/release/Masinissa-73M")


@pytest.mark.skipif(
    not (RELEASE_DIR / "model.safetensors").is_file(),
    reason="model.safetensors not staged yet",
)
def test_masinissa_73m_files_present() -> None:
    expected_files = {
        "README.md",
        "config.json",
        "configuration_masinissa.py",
        "modeling_masinissa.py",
        "mammeri-tok-base-16k.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    }
    present = {p.name for p in RELEASE_DIR.iterdir()}
    assert expected_files.issubset(present), f"Missing: {expected_files - present}"


@pytest.mark.skipif(
    not (RELEASE_DIR / "model.safetensors").is_file(),
    reason="model.safetensors not staged yet",
)
def test_masinissa_73m_transformers_loading() -> None:
    config = AutoConfig.from_pretrained(str(RELEASE_DIR), trust_remote_code=True)
    assert config.hidden_size == 512
    assert config.num_hidden_layers == 20
    assert config.num_attention_heads == 8
    assert config.vocab_size == 16000

    tokenizer = AutoTokenizer.from_pretrained(str(RELEASE_DIR), trust_remote_code=True)
    assert len(tokenizer) == 16000

    model = AutoModelForMaskedLM.from_pretrained(str(RELEASE_DIR), trust_remote_code=True)
    model.eval()

    text = "Tamaziɣt d [MASK] n imezwura-nneɣ."
    encoded = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**encoded).logits

    assert logits.shape == (1, encoded["input_ids"].shape[1], 16000)
    mask_pos = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
    top_token = str(tokenizer.decode([int(logits[0, mask_pos].argmax().item())])).strip()
    assert top_token == "tayemmat"

    # Base AutoModel test
    base = AutoModel.from_pretrained(str(RELEASE_DIR), trust_remote_code=True)
    base.eval()
    with torch.inference_mode():
        hidden = base(**encoded).last_hidden_state
    assert hidden.shape == (1, encoded["input_ids"].shape[1], 512)
