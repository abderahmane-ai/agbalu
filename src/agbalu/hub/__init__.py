"""Implementations copied verbatim into the published repositories.

A downloader has `transformers` and `torch` and nothing else. `agbalu.model.modeling` and
`agbalu.tifinagh.model` are the definitions the weights were trained under and they import
from this project, so neither can be what a published repository ships. The modules here
are the same architectures written against `transformers` alone, which is what makes
`from_pretrained(..., trust_remote_code=True)` construct them on a machine where this
repository does not exist.

Two rules hold this package together, and both are asserted in
`tests/unit/test_hub_staging.py`: nothing under it may import `agbalu`, and each module
must reproduce its training counterpart's output tensor for tensor on the same weights.

Release names appear as identifiers here and nowhere else in the tree. On the Hub the
class name and `model_type` *are* the public contract — `auto_map` names them and
`model_type` must not collide with another repository's — which is the same exemption
CLAUDE.md §4 already grants artifact filenames.
"""
