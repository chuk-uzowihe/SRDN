"""Builders: assemble an LM for each architecture as a stack of `Block`s, where each
Block = (token mixer from ops/, the SHARED ChannelMixer). The channel mixer (FFN/MoE)
and all scaffold are identical across architectures -- only the token mixer differs.

Default channel mixer = dense SwiGLU FFN (ffn_mult), identical for every arch. MoE is
opt-in (moe_ffn=True), also identical across archs. srdn and rwkv7 additionally offer
faithful_channel_mix (RWKV-7's native token-shift + squared-ReLU channel mix) so those
cells can conform to real RWKV-7 end to end.
"""
from __future__ import annotations

from .channel import ChannelMixer
from .core import Block, SRDNLM


def _channels(n, d_model, ffn_mult, moe_ffn, n_experts, top_k, expert_mult):
    return [ChannelMixer(d_model, ffn_mult, moe_ffn=moe_ffn, n_experts=n_experts,
                         top_k=top_k, expert_mult=expert_mult) for _ in range(n)]


def _assemble(vocab_size, d_model, mixers, *, ffn_mult, moe_ffn, n_experts, top_k, expert_mult,
              pos_embed=None):
    chans = _channels(len(mixers), d_model, ffn_mult, moe_ffn, n_experts, top_k, expert_mult)
    blocks = [Block(m, c) for m, c in zip(mixers, chans)]
    return SRDNLM(vocab_size, d_model, blocks, pos_embed=pos_embed)


def build_srdn(vocab_size, d_model, n_layers, ffn_mult=2.0, *, head_dim=32,
               value_dim=None, content_read_mode="per_proj", use_lora=True, lora_rank=0,
               neg_eigval=False, fuse_scan=True, scan_block=16, read_rank=None,
               faithful_channel_mix=True, hidden_ratio=4,
               moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    """SRDN: state-reading RWKV-7 (recurrence-complete). Zero-init adapters => exactly
    RWKV-7 at init (no ReZero gamma -- see ops/srdn.py for why gamma * zero-init-LoRA is a
    dead saddle). The ablation axes are content_read_mode (1/2/>2 queries), read_rank
    (default None = head_dim, the "lite" low-rank read queries; 0 = full-rank), use_lora,
    and neg_eigval.

    faithful_channel_mix=True (default) uses RWKV-7's native channel mix (token-shift lerp +
    squared-ReLU, hidden_ratio 4) so the cell conforms to real RWKV-7 end to end; pass False
    for the scaffold-matched shared SwiGLU (comparable to the other archs, and the only
    variant with a verified rollout `step` -- the native channel mix is full-sequence-forward
    only)."""
    from .ops.srdn import SRDNMixer
    from .ops.rwkv7 import VFirstBus
    bus = VFirstBus()  # one v_first carrier shared across the stack's RWKV layers
    mixers = [SRDNMixer(d_model, head_dim=head_dim, layer_idx=i, num_layers=n_layers,
                        value_dim=value_dim, bus=bus, content_read_mode=content_read_mode,
                        use_lora=use_lora, lora_rank=lora_rank, neg_eigval=neg_eigval,
                        fuse_scan=fuse_scan, scan_block=scan_block,
                        read_rank=read_rank) for i in range(n_layers)]
    if faithful_channel_mix:
        from .ops.rwkv7 import RWKVChannelMixer
        chans = [RWKVChannelMixer(d_model, hidden_ratio=hidden_ratio, layer_idx=i, num_layers=n_layers)
                 for i in range(n_layers)]
        return SRDNLM(vocab_size, d_model, [Block(m, c) for m, c in zip(mixers, chans)])
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


def build_transformer(vocab_size, d_model, n_layers, n_heads, ffn_mult=2.0, *,
                      max_seq_len=4096, moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.attention import AttentionMixer, sinusoidal_table
    mixers = [AttentionMixer(d_model, n_heads, max_seq_len=max_seq_len) for _ in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult,
                     pos_embed=sinusoidal_table(max_seq_len, d_model))


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
               expand_v=1.0, use_short_conv=True, allow_neg_eigval=True,
               moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.gdn2 import GDN2Mixer
    mixers = [GDN2Mixer(d_model, n_heads, head_dim, expand_v=expand_v, use_short_conv=use_short_conv,
                        allow_neg_eigval=allow_neg_eigval) for _ in range(n_layers)]
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


def build_rwkv7(vocab_size, d_model, n_layers, ffn_mult=2.0, *, head_dim=32, value_dim=None,
                faithful_channel_mix=False, hidden_ratio=4,
                moe_ffn=False, n_experts=4, top_k=2, expert_mult=2.3) -> SRDNLM:
    from .ops.rwkv7 import RWKV7Mixer, VFirstBus
    bus = VFirstBus()  # one v_first carrier shared across the stack's RWKV layers
    mixers = [RWKV7Mixer(d_model, head_dim=head_dim, layer_idx=i, num_layers=n_layers,
                         value_dim=value_dim, bus=bus) for i in range(n_layers)]
    if faithful_channel_mix:
        # paper-faithful RWKV-7: native channel mix (token-shift + squared-ReLU) instead of the
        # shared SwiGLU. NOT scaffold-matched to the other archs -- a separate, labeled variant.
        from .ops.rwkv7 import RWKVChannelMixer
        chans = [RWKVChannelMixer(d_model, hidden_ratio=hidden_ratio, layer_idx=i, num_layers=n_layers)
                 for i in range(n_layers)]
        return SRDNLM(vocab_size, d_model, [Block(m, c) for m, c in zip(mixers, chans)])
    return _assemble(vocab_size, d_model, mixers, ffn_mult=ffn_mult, moe_ffn=moe_ffn,
                     n_experts=n_experts, top_k=top_k, expert_mult=expert_mult)


BUILDERS = {"srdn": build_srdn, "transformer": build_transformer, "mamba3": build_mamba3,
            "m2rnn": build_m2rnn, "gdn2": build_gdn2, "rwkv7": build_rwkv7}

__all__ = ["build_srdn", "build_transformer", "build_mamba3", "build_m2rnn", "build_gdn2",
           "build_rwkv7", "BUILDERS"]
