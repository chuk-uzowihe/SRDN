"""Baseline architectures for the SRDN comparison (loaded from refs/ where external)."""
from .transformer import TransformerLM
from .mamba3 import Mamba3LM
from .m2rnn_canon import M2RNNCanonLM
from .gdn2 import GDN2LM

__all__ = ["TransformerLM", "Mamba3LM", "M2RNNCanonLM", "GDN2LM"]
