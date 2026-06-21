"""SRDN: recurrence-complete gated delta-rule linear attention with state-conditioned projections.

The shared scaffold (Block, SRDNLM, ChannelMixer) lives here; the token mixers --
the only thing that differs across architectures -- live in `ops/`. Build any model
via the builders.
"""
from .core import Block, SRDNLM
from .channel import ChannelMixer
from .builders import (build_srdn, build_transformer, build_mamba3, build_m2rnn,
                       build_gdn2, BUILDERS)

__all__ = ["Block", "SRDNLM", "ChannelMixer", "build_srdn", "build_transformer",
           "build_mamba3", "build_m2rnn", "build_gdn2", "BUILDERS"]
