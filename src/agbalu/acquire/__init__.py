"""Acquisition layer: fetch every registered source and record where it came from.

Phase 1 does no transformation. It resolves a `Source` from the registry to bytes
on a storage target, hashes them, and appends an immutable provenance record.
"""
