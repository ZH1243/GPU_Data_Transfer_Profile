"""Six-stage CuTe DSL routing metadata implementation for the two-node test.

Contract: contiguous CUDA int32 R[T, 8], distinct expert IDs per row in [0,128).
Topology: two nodes, eight GPUs/node, eight experts/GPU. All outputs match the
CUDA C++ node-mask API, including stable ordering and the direct-input exception.
This implementation uses bitmap ranks instead of CUB radix sorts. No QuACK
checkout is needed. Compile/allocate once; launch repeatedly on the chosen stream.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack


@cute.jit
def _splitmix64(value: cutlass.Uint64):
    value += cutlass.Uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> 30)) * cutlass.Uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> 27)) * cutlass.Uint64(0x94D049BB133111EB)
    return value ^ (value >> 31)


class RouteGenerator:
    """GPU input generator; seed is a runtime scalar, not a JIT specialization.

    Matches generate_r_kernel for 128 experts/top-k 8. A stride coprime to
    128 is odd, so rounding an even candidate up replaces the GCD loop.
    """

    @cute.jit
    def __call__(self, r: cute.Tensor, seed: cutlass.Uint64, stream: cuda.CUstream):
        self.generate_routes(r, seed).launch(
            grid=(cute.ceil_div(r.shape[0], 256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def generate_routes(self, r: cute.Tensor, seed: cutlass.Uint64):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        token = block * 256 + thread
        if token < r.shape[0]:
            first = _splitmix64(seed ^ cutlass.Uint64(token))
            second = _splitmix64(first)
            start = cutlass.Int32(first % cutlass.Uint64(128))
            stride = cutlass.Int32(second % cutlass.Uint64(127)) + 1
            if stride % 2 == 0:
                stride += 1
            for k in cutlass.range_constexpr(8):
                r[token, k] = (start + k * stride) % 128


@cute.jit
def _warp_scan(value: cutlass.Int32, lane: cutlass.Int32):
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(value, offset=1 << shift, mask_and_clamp=0)
        if lane >= (1 << shift):
            value += other
    return value


@cute.jit
def _block_scan(value: cutlass.Int32, thread: cutlass.Int32, sums: cute.Tensor):
    """Inclusive sum of 256 values; return this thread's prefix and block total."""
    lane = thread % 32
    warp = thread // 32
    value = _warp_scan(value, lane)
    if lane == 31:
        sums[warp] = value
    cute.arch.sync_threads()
    before = cutlass.Int32(0)
    total = cutlass.Int32(0)
    for w in cutlass.range_constexpr(8):
        part = sums[w]
        total += part
        if w < warp:
            before += part
    # Every warp must finish reading before the next invocation overwrites sums.
    cute.arch.sync_threads()
    return value + before, total


@cute.jit
def _bitmap_rank(bits: cute.Tensor, group: cutlass.Int32, thread: cutlass.Int32):
    """Number of set bits for this group at token positions below thread."""
    warp = thread // 32
    lane = thread % 32
    lower = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
    rank = cutlass.Int32(cute.arch.popc(cutlass.Uint32(bits[group, warp]) & lower))
    for w in cutlass.range_constexpr(8):
        if w < warp:
            rank += cutlass.Int32(cute.arch.popc(cutlass.Uint32(bits[group, w])))
    return rank


class MetadataKernels:
    def __init__(self, tokens, local_gpu_id):
        self.tokens = tokens
        self.chunks = (tokens + 255) // 256
        self.local_gpu_id = local_gpu_id

    @cute.jit
    def __call__(self, r: cute.Tensor, b: tuple, stream: cuda.CUstream):
        (masks, bits, counts, prefixes, mask_offsets, expert_offsets,
         x3, x4, node_offsets, reordered_bits, reordered_counts,
         reordered_prefixes, values) = b
        self.count_input(r, masks, bits, counts).launch(
            grid=(self.chunks, 1, 1), block=(256, 1, 1), stream=stream)
        self.scan_input(counts, prefixes, mask_offsets, expert_offsets, node_offsets).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
        self.build_node_indices(masks, bits, prefixes, mask_offsets, x3, x4).launch(
            grid=(self.chunks, 1, 1), block=(256, 1, 1), stream=stream)
        self.count_reordered(r, x3, x4, node_offsets, reordered_bits, reordered_counts).launch(
            grid=(self.chunks, 2, 1), block=(256, 1, 1), stream=stream)
        self.scan_reordered(reordered_counts, reordered_prefixes).launch(
            grid=(18, 1, 1), block=(256, 1, 1), stream=stream)
        self.scatter_experts(r, x3, node_offsets, reordered_bits, reordered_prefixes,
                             expert_offsets, values).launch(
            grid=(self.chunks, 2, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def count_input(self, r: cute.Tensor, masks: cute.Tensor, bits: cute.Tensor,
                    counts: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        chunk, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        bitmap = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4096,)))
        histogram = smem.allocate_tensor(cutlass.Int32, cute.make_layout((128,)))
        for i in cutlass.range(thread, 4096, 256):
            bitmap[i] = 0
        if thread < 128:
            histogram[thread] = 0
        cute.arch.sync_threads()
        token = chunk * 256 + thread
        mask0 = cutlass.Int32(0)
        mask1 = cutlass.Int32(0)
        if token < self.tokens:
            for k in cutlass.range_constexpr(8):
                expert = r[token, k]
                cute.arch.atomic_add(histogram.iterator + expert, cutlass.Int32(1))
                bit = cutlass.Int32(1) << ((self.local_gpu_id - (expert // 8) % 8 + 8) % 8)
                if expert < 64:
                    mask0 |= bit
                else:
                    mask1 |= bit
            masks[token, 0] = cutlass.Uint8(mask0)
            masks[token, 1] = cutlass.Uint8(mask1)
            lane_bit = cutlass.Int32(cutlass.Uint32(1) << (thread % 32))
            if mask0 != 0:
                cute.arch.atomic_or(bitmap.iterator + mask0 * 8 + thread // 32, lane_bit)
            if mask1 != 0:
                cute.arch.atomic_or(bitmap.iterator + (256 + mask1) * 8 + thread // 32, lane_bit)
        cute.arch.sync_threads()
        for i in cutlass.range(thread, 4096, 256):
            bits[chunk, i // 8, i % 8] = bitmap[i]
        if thread < 128:
            counts[chunk, thread] = histogram[thread]
        for group in cutlass.range(thread, 512, 256):
            total = cutlass.Int32(0)
            for w in cutlass.range_constexpr(8):
                total += cutlass.Int32(cute.arch.popc(cutlass.Uint32(bitmap[group * 8 + w])))
            counts[chunk, 128 + group] = total

    @cute.kernel
    def scan_input(self, counts: cute.Tensor, prefixes: cute.Tensor,
                   mask_offsets: cute.Tensor, expert_offsets: cute.Tensor,
                   node_offsets: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        totals = smem.allocate_tensor(cutlass.Int32, cute.make_layout((640,)))
        sums = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8,)))
        for group in cutlass.range(thread, 640, 256):
            total = cutlass.Int32(0)
            for chunk in cutlass.range(self.chunks):
                prefixes[chunk, group] = total
                total += counts[chunk, group]
            totals[group] = total
        cute.arch.sync_threads()
        value = cutlass.Int32(0)
        if thread < 128:
            value = totals[thread]
        prefix, _ = _block_scan(value, thread, sums)
        if thread == 0:
            expert_offsets[0] = 0
            node_offsets[0] = 0
        if thread < 128:
            expert_offsets[thread + 1] = prefix
        node_base = cutlass.Int32(0)
        for node in cutlass.range_constexpr(2):
            mask = 255 - thread
            value = totals[128 + node * 256 + mask]
            prefix, total = _block_scan(value, thread, sums)
            mask_offsets[node * 256 + mask] = node_base + prefix - value
            node_base += total
            if thread == 0:
                node_offsets[node + 1] = node_base

    @cute.kernel
    def build_node_indices(self, masks: cute.Tensor, bits: cute.Tensor,
                           prefixes: cute.Tensor, mask_offsets: cute.Tensor,
                           x3: cute.Tensor, x4: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        chunk, _, _ = cute.arch.block_idx()
        token = chunk * 256 + thread
        if token < self.tokens:
            for node in cutlass.range_constexpr(2):
                mask = cutlass.Int32(masks[token, node])
                if mask != 0:
                    group = node * 256 + mask
                    rank = _bitmap_rank(bits[chunk, None, None], group, thread)
                    dst = mask_offsets[group] + prefixes[chunk, 128 + group] + rank
                    x3[dst] = token
                    x4[dst] = cutlass.Uint8(mask)

    @cute.kernel
    def count_reordered(self, r: cute.Tensor, x3: cute.Tensor, x4: cute.Tensor,
                        node_offsets: cute.Tensor, bits: cute.Tensor, counts: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        chunk, node, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        bitmap = smem.allocate_tensor(cutlass.Int32, cute.make_layout((576,)))
        for i in cutlass.range(thread, 576, 256):
            bitmap[i] = 0
        cute.arch.sync_threads()
        position = node_offsets[node] + chunk * 256 + thread
        if position < node_offsets[node + 1]:
            token = x3[position]
            lane_bit = cutlass.Int32(cutlass.Uint32(1) << (thread % 32))
            for k in cutlass.range_constexpr(8):
                expert = r[token, k]
                if expert // 64 == node:
                    group = expert % 64
                    cute.arch.atomic_or(bitmap.iterator + group * 8 + thread // 32, lane_bit)
            mask = cutlass.Int32(x4[position])
            for gpu in cutlass.range_constexpr(8):
                if (mask & (1 << ((self.local_gpu_id - gpu + 8) % 8))) != 0:
                    cute.arch.atomic_or(bitmap.iterator + (64 + gpu) * 8 + thread // 32, lane_bit)
        cute.arch.sync_threads()
        for i in cutlass.range(thread, 576, 256):
            bits[node, chunk, i // 8, i % 8] = bitmap[i]
        if thread < 72:
            total = cutlass.Int32(0)
            for w in cutlass.range_constexpr(8):
                total += cutlass.Int32(cute.arch.popc(cutlass.Uint32(bitmap[thread * 8 + w])))
            counts[node, chunk, thread] = total

    @cute.kernel
    def scan_reordered(self, counts: cute.Tensor, prefixes: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        # One warp per (node, expert/GPU) bin; 18 CTAs cover 144 bins.
        group = block * 8 + thread // 32
        node = group // 72
        bin_id = group % 72
        lane = thread % 32
        carry = cutlass.Int32(0)
        for first in cutlass.range(0, self.chunks, 32):
            chunk = first + lane
            value = cutlass.Int32(0)
            if chunk < self.chunks:
                value = counts[node, chunk, bin_id]
            prefix = _warp_scan(value, lane)
            if chunk < self.chunks:
                prefixes[node, chunk, bin_id] = carry + prefix - value
            carry += cute.arch.shuffle_sync(prefix, offset=31)

    @cute.kernel
    def scatter_experts(self, r: cute.Tensor, x3: cute.Tensor, node_offsets: cute.Tensor,
                        bits: cute.Tensor, prefixes: cute.Tensor,
                        expert_offsets: cute.Tensor, values: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        chunk, node, _ = cute.arch.block_idx()
        position = chunk * 256 + thread
        absolute = node_offsets[node] + position
        if absolute < node_offsets[node + 1]:
            token = x3[absolute]
            for k in cutlass.range_constexpr(8):
                expert = r[token, k]
                if expert // 64 == node:
                    local_expert = expert % 64
                    gpu = local_expert // 8
                    rank = _bitmap_rank(bits[node, chunk, None, None], local_expert, thread)
                    dst = expert_offsets[expert] + prefixes[node, chunk, local_expert] + rank
                    index = position
                    if gpu != self.local_gpu_id:
                        index = prefixes[node, chunk, 64 + gpu] + _bitmap_rank(
                            bits[node, chunk, None, None], 64 + gpu, thread)
                    values[dst] = index


class MetadataPlan:
    """Own fixed-shape buffers and a compiled six-kernel launch callable.

    Construction validates R (including copying routes to CPU); launch() does
    not allocate, compile, synchronize, or validate. R may be updated in place
    between launches provided the distinct/in-range route contract is preserved.
    Calls sharing a plan must be ordered on one stream or externally synchronized.
    """

    def __init__(self, r, local_gpu_id=0):
        import torch

        if (not r.is_cuda or r.dtype != torch.int32 or r.ndim != 2
                or r.shape[1] != 8 or not r.is_contiguous()):
            raise ValueError("R must be a contiguous CUDA int32 tensor of shape [T,8]")
        if not 1 <= r.shape[0] <= 10_000_000 or not 0 <= local_gpu_id < 8:
            raise ValueError("require 1 <= T <= 10000000 and 0 <= local_gpu_id < 8")
        routes = r.cpu()
        ordered = routes.sort(dim=1).values
        if (ordered[:, 0] < 0).any() or (ordered[:, -1] >= 128).any():
            raise ValueError("expert IDs must be in [0,128)")
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("each token must select eight distinct experts")
        self.r = r
        self.local_gpu_id = local_gpu_id
        tokens = r.shape[0]
        chunks = (tokens + 255) // 256

        def allocate(shape, dtype=torch.int32):
            return torch.empty(shape, dtype=dtype, device=r.device)

        self.buffers = (
            allocate((tokens, 2), torch.uint8), allocate((chunks, 512, 8)),
            allocate((chunks, 640)), allocate((chunks, 640)), allocate((512,)),
            allocate((129,)), allocate((tokens * 2,)), allocate((tokens * 2,), torch.uint8),
            allocate((3,)), allocate((2, chunks, 72, 8)), allocate((2, chunks, 72)),
            allocate((2, chunks, 72)), allocate((tokens * 8,)),
        )
        self.expert_offsets = self.buffers[5]
        self.x3, self.x4, self.node_offsets = self.buffers[6:9]
        self.expert_token_indices = self.buffers[12]
        with torch.cuda.device(r.device):
            self.arguments = (from_dlpack(r), tuple(from_dlpack(b) for b in self.buffers))
            stream = cuda.CUstream(torch.cuda.current_stream(r.device).cuda_stream)
            self.compiled = cute.compile(MetadataKernels(tokens, local_gpu_id),
                                         *self.arguments, stream)

    def launch(self, stream=None):
        import torch

        if stream is None:
            stream = torch.cuda.current_stream(self.r.device)
        if stream.device != self.r.device:
            raise ValueError("launch stream and routing tensor must use the same GPU")
        with torch.cuda.device(self.r.device):
            self.compiled(*self.arguments, cuda.CUstream(stream.cuda_stream))

    def check(self):
        """Validate against the existing independent CPU reference, outside timing."""
        import torch
        from run import check_node_mask_result

        torch.cuda.synchronize(self.r.device)
        check_node_mask_result(self.r, self.expert_token_indices, self.expert_offsets,
                               self.x3, self.x4, self.node_offsets, 128, 8, 8,
                               self.local_gpu_id)
