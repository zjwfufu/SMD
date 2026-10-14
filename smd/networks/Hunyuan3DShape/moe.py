# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored & adapted from Tencent Hunyuan3D-2.1
#   hy3dshape/hy3dshape/models/denoisers/moe_layers.py
# Original: Copyright (C) 2024 THL A29 Limited (Tencent), TENCENT HUNYUAN
# NON-COMMERCIAL LICENSE AGREEMENT. The expert FeedForward (originally
# diffusers.models.attention.FeedForward with activation_fn="gelu") is vendored
# here with the SAME submodule layout (net.0.proj -> GELU, net.2 -> Linear) so
# pretrained MoE weights load unchanged, without depending on diffusers internals.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GELUProj(nn.Module):
    """diffusers-compatible GELU activation block: Linear projection then GELU.

    State-dict keys: ``proj.weight`` / ``proj.bias`` (matches diffusers GELU).
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    """Minimal drop-in for diffusers ``FeedForward(activation_fn="gelu")``.

    Submodule layout matches diffusers exactly so checkpoint keys line up::

        net.0.proj.{weight,bias}   # GELU input projection (dim -> inner_dim)
        net.2.{weight,bias}        # output projection (inner_dim -> dim)

    ``net.1`` is a dropout placeholder (identity at p=0) to preserve indices.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * 4)
        assert activation_fn == "gelu", f"vendored FeedForward only supports gelu, got {activation_fn}"
        self.net = nn.ModuleList(
            [
                _GELUProj(dim, inner_dim, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class AddAuxiliaryLoss(torch.autograd.Function):
    """Inject the MoE gating auxiliary loss into the backward graph."""

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
                aux_loss = aux_loss * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MoEBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_experts=8,
        moe_top_k=2,
        activation_fn="gelu",
        dropout=0.0,
        final_dropout=False,
        ff_inner_dim=None,
        ff_bias=True,
    ):
        super().__init__()
        self.moe_top_k = moe_top_k
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    final_dropout=final_dropout,
                    inner_dim=ff_inner_dim,
                    bias=ff_bias,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = MoEGate(embed_dim=dim, num_experts=num_experts, num_experts_per_tok=moe_top_k)
        self.shared_experts = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.moe_top_k, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts):
                tmp = expert(hidden_states[flat_topk_idx == i])
                y[flat_topk_idx == i] = tmp.to(hidden_states.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.moe_top_k
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache
