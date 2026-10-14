# SPDX-License-Identifier: Apache-2.0

"""Ulysses context parallel primitives for the MiniMax H3 packed transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class MiniMaxH3ContextParallelState:
    degree: int
    rank: int
    group: dist.ProcessGroup | None
    group_ranks: tuple[int, ...]
    data_parallel_rank: int
    data_parallel_size: int

    @property
    def enabled(self) -> bool:
        return self.degree > 1

    @property
    def source_rank(self) -> int:
        return self.group_ranks[0]

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.broadcast(tensor, src=self.source_rank, group=self.group)
        return tensor

    def __deepcopy__(self, _memo) -> MiniMaxH3ContextParallelState:
        return self


_STATE_CACHE: dict[tuple[int, int, int], MiniMaxH3ContextParallelState] = {}


class _AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.Tensor:
        ctx.group = group
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output.contiguous(), group=ctx.group)
        return None, grad_input


class _AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        rank: int,
        degree: int,
        dim: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.rank = rank
        ctx.degree = degree
        ctx.dim = dim
        gathered = [torch.empty_like(tensor) for _ in range(degree)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, None, None, None, torch.Tensor]:
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return None, None, None, None, grad_output.chunk(ctx.degree, dim=ctx.dim)[ctx.rank].contiguous()


def get_context_parallel_state(degree: int) -> MiniMaxH3ContextParallelState:
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"MiniMax H3 context_parallel_size must be positive, got {degree}.")
    if not dist.is_initialized():
        if degree != 1:
            raise RuntimeError("MiniMax H3 context parallelism requires an initialized distributed process group.")
        return MiniMaxH3ContextParallelState(1, 0, None, (0,), 0, 1)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if degree == 1:
        return MiniMaxH3ContextParallelState(1, 0, None, (rank,), rank, world_size)
    if world_size % degree:
        raise ValueError(f"Distributed world size {world_size} must be divisible by context_parallel_size={degree}.")
    key = (degree, world_size, rank)
    if key in _STATE_CACHE:
        return _STATE_CACHE[key]

    selected_group = None
    selected_ranks: tuple[int, ...] | None = None
    for start in range(0, world_size, degree):
        ranks = tuple(range(start, start + degree))
        group = dist.new_group(ranks=list(ranks), backend="nccl")
        if rank in ranks:
            selected_group = group
            selected_ranks = ranks
    if selected_group is None or selected_ranks is None:
        raise RuntimeError(f"Rank {rank} was not assigned to a MiniMax H3 context-parallel group.")

    state = MiniMaxH3ContextParallelState(
        degree=degree,
        rank=rank - selected_ranks[0],
        group=selected_group,
        group_ranks=selected_ranks,
        data_parallel_rank=rank // degree,
        data_parallel_size=world_size // degree,
    )
    _STATE_CACHE[key] = state
    return state


def sequence_to_heads(tensor: torch.Tensor, state: MiniMaxH3ContextParallelState) -> torch.Tensor:
    """Transform ``[B,S/P,H,D]`` into ``[B,S,H/P,D]`` with autograd."""

    if not state.enabled:
        return tensor
    batch, local_sequence, num_heads, head_dim = tensor.shape
    if num_heads % state.degree:
        raise ValueError(f"MiniMax H3 attention heads {num_heads} must be divisible by CP degree {state.degree}.")
    local_heads = num_heads // state.degree
    send = (
        tensor.reshape(batch, local_sequence, state.degree, local_heads, head_dim).permute(2, 0, 1, 3, 4).contiguous()
    )
    received = _AllToAllSingle.apply(state.group, send.flatten())
    return (
        received.reshape(state.degree, batch, local_sequence, local_heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .reshape(batch, state.degree * local_sequence, local_heads, head_dim)
    )


def heads_to_sequence(tensor: torch.Tensor, state: MiniMaxH3ContextParallelState) -> torch.Tensor:
    """Transform ``[B,S,H/P,D]`` back into ``[B,S/P,H,D]`` with autograd."""

    if not state.enabled:
        return tensor
    batch, sequence, local_heads, head_dim = tensor.shape
    if sequence % state.degree:
        raise ValueError(f"MiniMax H3 sequence length {sequence} must be divisible by CP degree {state.degree}.")
    local_sequence = sequence // state.degree
    send = (
        tensor.reshape(batch, state.degree, local_sequence, local_heads, head_dim).permute(1, 0, 2, 3, 4).contiguous()
    )
    received = _AllToAllSingle.apply(state.group, send.flatten())
    return (
        received.reshape(state.degree, batch, local_sequence, local_heads, head_dim)
        .permute(1, 2, 0, 3, 4)
        .reshape(batch, local_sequence, state.degree * local_heads, head_dim)
    )


def split_sequence(tensor: torch.Tensor, state: MiniMaxH3ContextParallelState, dim: int = 1) -> torch.Tensor:
    if not state.enabled:
        return tensor
    if tensor.shape[dim] % state.degree:
        raise ValueError(
            f"MiniMax H3 sequence dimension {tensor.shape[dim]} must be divisible by CP degree {state.degree}."
        )
    return tensor.chunk(state.degree, dim=dim)[state.rank].contiguous()


def gather_sequence(
    tensor: torch.Tensor,
    state: MiniMaxH3ContextParallelState,
    dim: int = 1,
    *,
    autograd: bool,
) -> torch.Tensor:
    if not state.enabled:
        return tensor
    if autograd:
        return _AllGatherSequence.apply(state.group, state.rank, state.degree, dim, tensor)
    gathered = [torch.empty_like(tensor) for _ in range(state.degree)]
    dist.all_gather(gathered, tensor.contiguous(), group=state.group)
    return torch.cat(gathered, dim=dim)
