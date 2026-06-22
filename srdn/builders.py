"""Builders: assemble an LM for each architecture as a stack of `Block`s, where each
Block = (token mixer from ops/, the SHARED ChannelMixer). The channel mixer (FFN/MoE)
and all scaffold are identical across architectures -- only the token mixer differs.

Default channel mixer = dense SwiGLU FFN (ffn_mult), identical for every arch. MoE is
opt-in (moe_ffn=True), also identical across archs.
"""
from __future__ import annotations

from .channel import ChannelMixer
from .core import Block, SRDNLM


def _channels(n, d_model, ffn_mult, moe_ffn, n_experts, top_k, expert_mult):
    return [ChannelMixer(d_model, ffn_mult, moe_ffn=moe_ffn, n_experts=n_experts,
                         top_k=top_k, expert_mult=expert_mult) for _ in range(n)]


def _assemble(vocab_size, d_model, mixers, *, ffn_mult, moe_ffn, n_experts, top_k, expert_mult):
    chans = _channels(len(mixers), d_model, ffn_mult, moe_ffn, n_experts, top_k, expert_mult)
    blocks = [Block(m, c) for m, c in zip(mixers, chans)]
    return SRDNLM(vocab_size, d_model, blocks)


def build_srdn(vocab_size, d_model, n_layers, n_heads, d_head, ffn_mult=2.0, *,
               short_conv=False, conv_size=4, rezero_init=0.0,
               moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.srdn import SRDNMixer
    mixers = [SRDNMixer(d_model, n_heads, d_head, short_conv=short_conv, conv_size=conv_size,
                        rezero_init=rezero_init) for _ in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


def build_transformer(vocab_size, d_model, n_layers, n_heads, ffn_mult=2.0, *,
                      max_seq_len=4096, moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.attention import AttentionMixer
    mixers = [AttentionMixer(d_model, n_heads, max_seq_len=max_seq_len) for _ in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


def build_mamba3(vocab_size, d_model, n_layers, ffn_mult=2.0, *, state_size=128, expand=2,
                 head_dim=64, n_groups=1, chunk_size=64,
                 moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.mamba3 import Mamba3Mixer
    mixers = [Mamba3Mixer(d_model, state_size=state_size, expand=expand, head_dim=head_dim,
                          n_groups=n_groups, chunk_size=chunk_size) for _ in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


def build_m2rnn(vocab_size, d_model, n_layers, n_heads, head_dim, ffn_mult=2.0, *,
                kernel_size=4, moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.m2rnn import M2RNNMixer
    mixers = [M2RNNMixer(d_model, n_heads, head_dim, kernel_size=kernel_size,
                         n_layers=n_layers, layer_idx=i) for i in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


def build_gdn2(vocab_size, d_model, n_layers, n_heads, head_dim, ffn_mult=2.0, *,
               expand_v=1.0, use_short_conv=True, allow_neg_eigval=True, gdn2_repo=None,
               moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.gdn2 import GDN2Mixer
    mixers = [GDN2Mixer(d_model, n_heads, head_dim, expand_v=expand_v, use_short_conv=use_short_conv,
                        allow_neg_eigval=allow_neg_eigval, gdn2_repo=gdn2_repo) for _ in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


BUILDERS = {"srdn": build_srdn, "transformer": build_transformer, "mamba3": build_mamba3,
            "m2rnn": build_m2rnn, "gdn2": build_gdn2}

__all__ = ["build_srdn", "build_transformer", "build_mamba3", "build_m2rnn", "build_gdn2", "BUILDERS"]
