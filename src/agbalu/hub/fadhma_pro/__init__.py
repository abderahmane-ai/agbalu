"""Standalone Fadhma-Pro ASR model for Hugging Face Hub."""

from .configuration_fadhma_pro import FadhmaProConfig
from .modeling_fadhma_pro import FadhmaProForCTC

__all__ = ["FadhmaProConfig", "FadhmaProForCTC"]
