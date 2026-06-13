#!/usr/bin/env python3
# Copyright (c) 2026.
#
# Standalone DeepEP MoE dispatch -> expert computation -> combine test.
#
# The DeepEP fused dispatch/combine wrapper logic and the grouped expert MLP
# call structure below are copied/adapted from Megatron-LM's
# megatron/core/transformer/moe/fused_a2a.py,
# megatron/core/transformer/moe/token_dispatcher.py, and
# megatron/core/transformer/moe/experts.py. This file intentionally does not
# import Megatron-LM so it keeps working if the Megatron-LM source tree is
# removed.

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


_DEEPEP_BUFFER = None


def _load_deepep():
    try:
        from deep_ep import Buffer
        from deep_ep.utils import EventHandle, EventOverlap
    except ImportError as exc:
        raise ImportError(
            "DeepEP is not importable. Install the deep_ep package on the "
            "server before running this test."
        ) from exc
    return Buffer, EventHandle, EventOverlap


def set_deepep_num_sms(num_sms: int) -> None:
    """Copied/adapted from Megatron-LM fused_a2a.py."""

    Buffer, _, _ = _load_deepep()
    Buffer.set_num_sms(num_sms)


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Copied/adapted from Megatron-LM fused_a2a.py."""

    return x.size(1) * max(x.element_size(), 2)


def get_deepep_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    """Copied/adapted from Megatron-LM fused_a2a.py."""

    global _DEEPEP_BUFFER
    Buffer, _, _ = _load_deepep()
    group_size = dist.get_world_size(group)
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group_size), Buffer.get_combine_config(group_size)):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group_size),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group_size),
            num_rdma_bytes,
        )

    if (
        _DEEPEP_BUFFER is None
        or _DEEPEP_BUFFER.group != group
        or _DEEPEP_BUFFER.num_nvl_bytes < num_nvl_bytes
        or _DEEPEP_BUFFER.num_rdma_bytes < num_rdma_bytes
    ):
        _DEEPEP_BUFFER = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _DEEPEP_BUFFER


class FusedDispatch(torch.autograd.Function):
    """Copied/adapted from Megatron-LM's DeepEP fused dispatch wrapper."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        _, EventHandle, EventOverlap = _load_deepep()
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        buffer = get_deepep_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event_overlap.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        if torch.is_tensor(num_recv_tokens_per_expert):
            tokens_per_expert = num_recv_tokens_per_expert.detach().cpu().to(torch.int64)
        else:
            tokens_per_expert = torch.tensor(num_recv_tokens_per_expert, dtype=torch.int64)
        return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):
        _, EventHandle, EventOverlap = _load_deepep()
        previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
        buffer = get_deepep_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            ctx.handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Copied/adapted from Megatron-LM's DeepEP fused combine wrapper."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        _, EventHandle, EventOverlap = _load_deepep()
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        buffer = get_deepep_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        _, EventHandle, EventOverlap = _load_deepep()
        previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
        buffer = get_deepep_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


def fused_dispatch(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup,
    async_finish: bool,
    allocate_on_comm_stream: bool,
):
    """Copied/adapted from Megatron-LM fused_a2a.py."""

    return FusedDispatch.apply(
        x.contiguous(),
        token_indices.contiguous(),
        token_probs.contiguous().float(),
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
    )


def fused_combine(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle,
    async_finish: bool,
    allocate_on_comm_stream: bool,
):
    """Copied/adapted from Megatron-LM fused_a2a.py."""

    return FusedCombine.apply(
        x.contiguous(),
        group,
        handle,
        async_finish,
        allocate_on_comm_stream,
    )


class DeepepDispatchCombineManager:
    """Minimal standalone version of Megatron-LM's _DeepepManager."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        local_expert_start: int,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.local_expert_start = local_expert_start
        self.async_finish = async_finish
        self.allocate_on_comm_stream = allocate_on_comm_stream
        self.handle = None
        self.dispatched_probs: Optional[torch.Tensor] = None
        self.tokens_per_expert: Optional[torch.Tensor] = None
        self.hidden_shape_before_local_permute: Optional[torch.Size] = None
        self.reversed_mapping_for_combine: Optional[torch.Tensor] = None

    def _indices_to_multihot(
        self,
        indices: torch.Tensor,
        probs: torch.Tensor,
        expected_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copied/adapted from Megatron-LM _DeepepManager._indices_to_multihot.

        Some DeepEP builds return the dispatched probability metadata as
        [num_dispatched_tokens, topk]. Megatron converts that matrix plus the
        returned indices into one local expert probability per expert-token row
        before calling the grouped expert MLP.
        """

        if indices.dim() == 1:
            indices = indices.unsqueeze(-1)
        if probs.dim() == 1:
            probs = probs.unsqueeze(-1)

        batch_size = indices.shape[0]
        local_end = self.local_expert_start + self.num_local_experts
        expected = expected_tokens_per_expert.detach().cpu().to(torch.int64)

        def build_candidate(local_indices: torch.Tensor, valid_mask: torch.Tensor):
            routing_map = torch.zeros(
                (batch_size, self.num_local_experts), dtype=torch.bool, device=indices.device
            )
            local_probs = torch.zeros(
                (batch_size, self.num_local_experts), dtype=torch.float32, device=indices.device
            )
            if valid_mask.any():
                rows = torch.arange(batch_size, device=indices.device).unsqueeze(1)
                rows = rows.expand_as(indices)
                routing_map[rows[valid_mask], local_indices[valid_mask].long()] = True
                local_probs[rows[valid_mask], local_indices[valid_mask].long()] = probs[
                    valid_mask
                ].float()
            counts = routing_map.sum(dim=0).detach().cpu().to(torch.int64)
            return routing_map, local_probs, counts

        local_mask = (indices >= 0) & (indices < self.num_local_experts)
        global_mask = (indices >= self.local_expert_start) & (indices < local_end)
        local_candidate = build_candidate(indices, local_mask)
        global_candidate = build_candidate(indices - self.local_expert_start, global_mask)

        local_matches = torch.equal(local_candidate[2], expected)
        global_matches = torch.equal(global_candidate[2], expected)
        if global_matches and not local_matches:
            routing_map, local_probs, _ = global_candidate
        elif local_matches:
            routing_map, local_probs, _ = local_candidate
        elif int(expected.sum().item()) != 0:
            sample = indices[: min(4, batch_size)].detach().cpu().tolist()
            raise RuntimeError(
                "DeepEP returned local tokens, but dispatched index metadata did not "
                f"match this rank's expected local expert counts. Expected "
                f"{expected.tolist()}, local-id counts {local_candidate[2].tolist()}, "
                f"global-id counts {global_candidate[2].tolist()}, local expert range "
                f"[{self.local_expert_start}, {local_end}). Sample indices: {sample}"
            )
        else:
            routing_map, local_probs, _ = local_candidate
        return routing_map, local_probs

    def _permute_to_local_experts(
        self,
        hidden_states: torch.Tensor,
        dispatched_indices: torch.Tensor,
        dispatched_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copied/adapted from Megatron-LM _DeepepManager.get_permuted_hidden_states_by_experts."""

        routing_map, local_probs = self._indices_to_multihot(
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert,
        )

        hidden_chunks = []
        prob_chunks = []
        reverse_chunks = []
        counts = []
        for expert_idx in range(self.num_local_experts):
            token_rows = torch.nonzero(routing_map[:, expert_idx], as_tuple=False).flatten()
            counts.append(token_rows.numel())
            if token_rows.numel() == 0:
                continue
            hidden_chunks.append(hidden_states.index_select(0, token_rows))
            prob_chunks.append(local_probs.index_select(0, token_rows)[:, expert_idx])
            reverse_chunks.append(token_rows)

        self.hidden_shape_before_local_permute = hidden_states.shape
        if hidden_chunks:
            permuted_hidden = torch.cat(hidden_chunks, dim=0).contiguous()
            permuted_probs = torch.cat(prob_chunks, dim=0).contiguous()
            self.reversed_mapping_for_combine = torch.cat(reverse_chunks, dim=0).contiguous()
        else:
            permuted_hidden = hidden_states.new_empty((0, hidden_states.shape[-1]))
            permuted_probs = torch.empty(0, dtype=torch.float32, device=hidden_states.device)
            self.reversed_mapping_for_combine = torch.empty(
                0, dtype=torch.long, device=hidden_states.device
            )

        recomputed_tokens_per_expert = torch.tensor(counts, dtype=torch.int64)
        expected = tokens_per_expert.detach().cpu().to(torch.int64)
        if not torch.equal(recomputed_tokens_per_expert, expected):
            raise RuntimeError(
                "Local DeepEP routing metadata is inconsistent: "
                f"computed tokens_per_expert={recomputed_tokens_per_expert.tolist()} "
                f"but DeepEP returned {expected.tolist()}."
            )
        return permuted_hidden, recomputed_tokens_per_expert, permuted_probs

    def _unpermute_from_local_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Copied/adapted from Megatron-LM _DeepepManager.get_restored_hidden_states_by_experts."""

        if self.hidden_shape_before_local_permute is None:
            raise RuntimeError("Missing shape metadata for local expert unpermute.")
        if self.reversed_mapping_for_combine is None:
            raise RuntimeError("Missing reverse mapping for local expert unpermute.")

        restored = hidden_states.new_zeros(self.hidden_shape_before_local_permute)
        if hidden_states.numel() != 0:
            restored.index_add_(0, self.reversed_mapping_for_combine, hidden_states)
        self.hidden_shape_before_local_permute = None
        self.reversed_mapping_for_combine = None
        return restored

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        token_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, dispatched_indices, dispatched_probs, tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states,
                token_indices,
                token_probs,
                self.num_experts,
                self.group,
                async_finish=self.async_finish,
                allocate_on_comm_stream=self.allocate_on_comm_stream,
            )
        )
        self.handle = handle
        self.tokens_per_expert = tokens_per_expert
        if tokens_per_expert.numel() != self.num_local_experts:
            raise RuntimeError(
                "DeepEP returned tokens_per_expert with length "
                f"{tokens_per_expert.numel()}, expected {self.num_local_experts}."
            )
        hidden_states, tokens_per_expert, dispatched_probs = self._permute_to_local_experts(
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert,
        )
        self.dispatched_probs = dispatched_probs
        self.tokens_per_expert = tokens_per_expert
        return hidden_states, tokens_per_expert, dispatched_probs

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.handle is None:
            raise RuntimeError("combine() called before dispatch().")
        hidden_states = self._unpermute_from_local_experts(hidden_states)
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=self.async_finish,
            allocate_on_comm_stream=self.allocate_on_comm_stream,
        )
        self.handle = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        return hidden_states


def _normal_init(tensor: torch.Tensor) -> None:
    torch.nn.init.normal_(tensor, mean=0.0, std=0.02)


def _call_grouped_linear(module, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
    out = module(x, splits)
    if isinstance(out, tuple):
        out = out[0]
    return out


class TransformerEngineGroupedExperts(torch.nn.Module):
    """Megatron-style TE grouped MLP, using the same grouped-linear kernel family."""

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        ffn_hidden_size: int,
        dtype: torch.dtype,
        activation: str,
    ):
        super().__init__()
        try:
            import transformer_engine.pytorch as te
        except ImportError as exc:
            raise ImportError("Transformer Engine is not importable.") from exc

        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation = activation
        self.gated = activation == "swiglu"
        fc1_out = 2 * ffn_hidden_size if self.gated else ffn_hidden_size
        self.linear_fc1 = self._make_grouped_linear(te, hidden_size, fc1_out, dtype)
        self.linear_fc2 = self._make_grouped_linear(te, ffn_hidden_size, hidden_size, dtype)

    def _make_grouped_linear(self, te, in_features: int, out_features: int, dtype: torch.dtype):
        cls = te.GroupedLinear
        kwargs = {
            "num_gemms": self.num_local_experts,
            "in_features": in_features,
            "out_features": out_features,
            "sequence_parallel": False,
            "fuse_wgrad_accumulation": False,
            "tp_group": None,
            "tp_size": 1,
            "get_rng_state_tracker": None,
            "init_method": _normal_init,
            "bias": False,
            "return_bias": False,
            "parallel_mode": None,
            "device": torch.cuda.current_device(),
            "params_dtype": dtype,
        }
        signature = inspect.signature(cls)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            kwargs = {key: val for key, val in kwargs.items() if key in signature.parameters}
        return cls(**kwargs)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate, linear = torch.chunk(x, 2, dim=-1)
            return F.silu(gate) * linear
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        routed_probs: torch.Tensor,
    ) -> torch.Tensor:
        # This mirrors Megatron-LM TEGroupedMLP.forward: fc1, activation,
        # multiply by routed probability, then fc2.
        splits = [int(x) for x in tokens_per_expert.cpu().tolist()]
        routed_probs = routed_probs.to(torch.float32).unsqueeze(-1)
        fc1_output = _call_grouped_linear(self.linear_fc1, hidden_states, splits)
        intermediate = self._activate(fc1_output)
        intermediate = (intermediate * routed_probs).to(hidden_states.dtype)
        return _call_grouped_linear(self.linear_fc2, intermediate, splits)


class TorchGroupedExperts(torch.nn.Module):
    """Small fallback adapted from Megatron-LM TEGroupedMLP math, without TE kernels."""

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        ffn_hidden_size: int,
        dtype: torch.dtype,
        activation: str,
    ):
        super().__init__()
        self.activation = activation
        self.gated = activation == "swiglu"
        fc1_out = 2 * ffn_hidden_size if self.gated else ffn_hidden_size
        self.weight1 = torch.nn.Parameter(
            torch.empty(num_local_experts, fc1_out, hidden_size, dtype=dtype, device="cuda")
        )
        self.weight2 = torch.nn.Parameter(
            torch.empty(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device="cuda")
        )
        _normal_init(self.weight1)
        _normal_init(self.weight2)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate, linear = torch.chunk(x, 2, dim=-1)
            return F.silu(gate) * linear
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        routed_probs: torch.Tensor,
    ) -> torch.Tensor:
        splits = [int(x) for x in tokens_per_expert.cpu().tolist()]
        if sum(splits) == 0:
            return hidden_states.new_empty((0, self.weight2.shape[1]))

        hidden_chunks = torch.split(hidden_states, splits)
        prob_chunks = torch.split(routed_probs.to(torch.float32), splits)
        outputs = []
        for expert_idx, (chunk, probs) in enumerate(zip(hidden_chunks, prob_chunks)):
            if chunk.numel() == 0:
                continue
            fc1 = F.linear(chunk, self.weight1[expert_idx])
            intermediate = self._activate(fc1)
            intermediate = (intermediate * probs.unsqueeze(-1)).to(chunk.dtype)
            outputs.append(F.linear(intermediate, self.weight2[expert_idx]))
        return torch.cat(outputs, dim=0) if outputs else hidden_states.new_empty((0, self.weight2.shape[1]))


def build_experts(args, num_local_experts: int, dtype: torch.dtype) -> tuple[torch.nn.Module, str]:
    if args.expert_backend in ("te", "auto"):
        try:
            experts = TransformerEngineGroupedExperts(
                num_local_experts,
                args.token_hidden,
                args.moe_ffn_hidden_size,
                dtype,
                args.activation,
            ).cuda()
            return experts, "transformer_engine_grouped_linear"
        except Exception as exc:
            if args.expert_backend == "te":
                raise
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"[warning] TE grouped experts unavailable; using torch fallback: {exc}")

    experts = TorchGroupedExperts(
        num_local_experts,
        args.token_hidden,
        args.moe_ffn_hidden_size,
        dtype,
        args.activation,
    ).cuda()
    return experts, "torch_fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone DeepEP MoE dispatch -> expert -> combine test."
    )
    parser.add_argument("--num-local-tokens", type=int, required=True)
    parser.add_argument("--token-hidden", type=int, required=True)
    parser.add_argument("--num-of-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--moe-ffn-hidden-size", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--activation", choices=("swiglu", "gelu", "silu", "relu"), default="swiglu")
    parser.add_argument("--expert-backend", choices=("auto", "te", "torch"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--deepep-num-sms", type=int, default=20)
    parser.add_argument("--print-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-correctness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-finish", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allocate-on-comm-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rerandomize-routing-each-iter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(dtype)


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    checks = [
        (args.num_local_tokens > 0, "--num-local-tokens must be positive."),
        (args.token_hidden > 0, "--token-hidden must be positive."),
        (args.num_of_experts > 0, "--num-of-experts must be positive."),
        (args.topk > 0, "--topk must be positive."),
        (args.ep > 1, "DeepEP dispatcher follows Megatron-LM and requires EP > 1."),
        (world_size % args.ep == 0, "--ep must divide torchrun WORLD_SIZE."),
        (
            args.num_of_experts % args.ep == 0,
            "--num-of-experts must be divisible by --ep for even expert partitioning.",
        ),
        (args.topk <= args.num_of_experts, "--topk cannot exceed --num-of-experts."),
        (args.moe_ffn_hidden_size > 0, "--moe-ffn-hidden-size must be positive."),
        (args.warmup_iters >= 0, "--warmup-iters cannot be negative."),
        (args.benchmark_iters >= 1, "--benchmark-iters must be at least 1."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def init_distributed() -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepEP.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK is not set. Run this script with torchrun.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def create_ep_group(ep_size: int) -> tuple[dist.ProcessGroup, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group_count = world_size // ep_size
    ep_group = None
    ep_group_id = rank // ep_size
    ep_rank = rank % ep_size

    if ep_size == world_size:
        ep_group = dist.group.WORLD
    else:
        for group_id in range(group_count):
            ranks = list(range(group_id * ep_size, (group_id + 1) * ep_size))
            group = dist.new_group(ranks=ranks, backend="nccl")
            if group_id == ep_group_id:
                ep_group = group
    if ep_group is None:
        raise RuntimeError("Failed to create EP process group.")
    return ep_group, ep_group_id, ep_rank


def make_random_routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Randomized top-k router output. This replaces Megatron-LM's real router.
    scores = torch.rand((num_tokens, num_experts), device=device, dtype=torch.float32)
    token_indices = torch.topk(scores, k=topk, dim=-1).indices.to(torch.int64)
    raw_probs = torch.rand((num_tokens, topk), device=device, dtype=torch.float32)
    token_probs = raw_probs / raw_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    return token_indices, token_probs


@dataclass
class IterationResult:
    output: torch.Tensor
    input_shape: tuple[int, ...]
    token_indices_shape: tuple[int, ...]
    token_probs_shape: tuple[int, ...]
    dispatch_shape: tuple[int, ...]
    expert_shape: tuple[int, ...]
    combine_shape: tuple[int, ...]
    tokens_per_expert_shape: tuple[int, ...]
    dispatched_probs_shape: tuple[int, ...]
    dispatch_ms: float
    expert_ms: float
    combine_ms: float
    total_ms: float


def elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


@torch.no_grad()
def run_one_iteration(
    args: argparse.Namespace,
    manager: DeepepDispatchCombineManager,
    experts: torch.nn.Module,
    input_tokens: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
) -> IterationResult:
    if args.rerandomize_routing_each_iter:
        token_indices, token_probs = make_random_routing(
            args.num_local_tokens,
            args.num_of_experts,
            args.topk,
            input_tokens.device,
        )

    start = torch.cuda.Event(enable_timing=True)
    after_dispatch = torch.cuda.Event(enable_timing=True)
    after_expert = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    dispatched, tokens_per_expert, dispatched_probs = manager.dispatch(
        input_tokens,
        token_indices,
        token_probs,
    )
    after_dispatch.record()
    expert_output = experts(dispatched, tokens_per_expert, dispatched_probs)
    after_expert.record()
    combined = manager.combine(expert_output)
    end.record()
    torch.cuda.synchronize()

    return IterationResult(
        output=combined,
        input_shape=tuple(input_tokens.shape),
        token_indices_shape=tuple(token_indices.shape),
        token_probs_shape=tuple(token_probs.shape),
        dispatch_shape=tuple(dispatched.shape),
        expert_shape=tuple(expert_output.shape),
        combine_shape=tuple(combined.shape),
        tokens_per_expert_shape=tuple(tokens_per_expert.shape),
        dispatched_probs_shape=tuple(dispatched_probs.shape),
        dispatch_ms=elapsed_ms(start, after_dispatch),
        expert_ms=elapsed_ms(after_dispatch, after_expert),
        combine_ms=elapsed_ms(after_expert, end),
        total_ms=elapsed_ms(start, end),
    )


def ordered_print(rank: int, world_size: int, message: str) -> None:
    for current in range(world_size):
        dist.barrier()
        if rank == current:
            print(message, flush=True)
    dist.barrier()


def all_ranks_boolean(ok: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    validate_args(args, world_size)
    ep_group, ep_group_id, ep_rank = create_ep_group(args.ep)
    dtype = dtype_from_arg(args.dtype)
    device = torch.device("cuda", local_rank)

    set_deepep_num_sms(args.deepep_num_sms)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    num_local_experts = args.num_of_experts // args.ep
    manager = DeepepDispatchCombineManager(
        ep_group,
        num_local_experts,
        args.num_of_experts,
        ep_rank * num_local_experts,
        args.async_finish,
        args.allocate_on_comm_stream,
    )
    experts, expert_backend = build_experts(args, num_local_experts, dtype)
    input_tokens = torch.randn(
        (args.num_local_tokens, args.token_hidden),
        device=device,
        dtype=dtype,
    )
    token_indices, token_probs = make_random_routing(
        args.num_local_tokens,
        args.num_of_experts,
        args.topk,
        device,
    )

    ordered_print(
        rank,
        world_size,
        (
            f"[rank {rank}/{world_size}] local_rank={local_rank} cuda={torch.cuda.current_device()} "
            f"ep_group={ep_group_id} ep_rank={ep_rank}/{args.ep} "
            f"tokens={args.num_local_tokens} hidden={args.token_hidden} "
            f"experts={args.num_of_experts} local_experts={num_local_experts} "
            f"topk={args.topk} moe_ffn_hidden={args.moe_ffn_hidden_size} "
            f"dtype={args.dtype} activation={args.activation} expert_backend={expert_backend}"
        ),
    )

    dist.barrier(group=ep_group)
    for _ in range(args.warmup_iters):
        run_one_iteration(args, manager, experts, input_tokens, token_indices, token_probs)
    dist.barrier(group=ep_group)

    results = []
    for _ in range(args.benchmark_iters):
        results.append(run_one_iteration(args, manager, experts, input_tokens, token_indices, token_probs))
    dist.barrier(group=ep_group)

    last = results[-1]
    local_ok = True
    if args.check_correctness:
        routing_ok = (
            last.token_indices_shape == (args.num_local_tokens, args.topk)
            and last.token_probs_shape == (args.num_local_tokens, args.topk)
            and token_indices.min().item() >= 0
            and token_indices.max().item() < args.num_of_experts
            and torch.allclose(
                token_probs.sum(dim=-1),
                torch.ones(args.num_local_tokens, device=device),
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        local_ok = (
            last.output.shape == input_tokens.shape
            and routing_ok
            and last.tokens_per_expert_shape == (num_local_experts,)
            and last.dispatched_probs_shape == (last.dispatch_shape[0],)
            and torch.isfinite(last.output).all().item()
        )
    global_ok = all_ranks_boolean(local_ok, device)

    avg_dispatch = sum(r.dispatch_ms for r in results) / len(results)
    avg_expert = sum(r.expert_ms for r in results) / len(results)
    avg_combine = sum(r.combine_ms for r in results) / len(results)
    avg_total = sum(r.total_ms for r in results) / len(results)

    ordered_print(
        rank,
        world_size,
        (
            f"[rank {rank}] shapes: input={last.input_shape} "
            f"topk_indices={last.token_indices_shape} topk_probs={last.token_probs_shape} "
            f"dispatch={last.dispatch_shape} expert_output={last.expert_shape} "
            f"combine={last.combine_shape} tokens_per_expert={last.tokens_per_expert_shape} "
            f"dispatched_probs={last.dispatched_probs_shape} sanity_ok={local_ok}"
        ),
    )

    if args.print_timing:
        ordered_print(
            rank,
            world_size,
            (
                f"[rank {rank}] avg over {args.benchmark_iters} iters: "
                f"dispatch={avg_dispatch:.3f} ms expert={avg_expert:.3f} ms "
                f"combine={avg_combine:.3f} ms total={avg_total:.3f} ms"
            ),
        )

    dist.barrier()
    if rank == 0:
        print(
            "All distributed ranks completed successfully."
            if global_ok
            else "At least one distributed rank failed sanity checks.",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if global_ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if dist.is_available() and dist.is_initialized():
            rank_text = f"rank {dist.get_rank()}"
        else:
            rank_text = "uninitialized rank"
        print(f"[{rank_text}] ERROR: {exc}", file=sys.stderr, flush=True)
        time.sleep(0.2)
        raise
