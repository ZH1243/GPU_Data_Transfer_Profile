"""Register-prefetch reduction adapted from QuACK run/final_combine.py.

Preserves its striped 128-bit loads and FP32 peer-order accumulation. The proxy
publishes groups of N tiles (not necessarily one tile) and owns the HBM flag.
"""
from typing import Optional
import cutlass
import cutlass.cute as cute
from run.final_combine import _CombineKernel


class RegisterIntraNodeKernel(_CombineKernel):
    """Stripe 128-bit vectors across lanes; load peer batches before consuming them.

    Unlike contiguous per-thread vectors, every copy instruction covers adjacent
    lane stripes. Stage values in received precision, then reduce in peer order.
    Prefetch depth is a compiler scheduling opportunity, not an async-copy fence.
    """

    wait_for_inputs = True
    publish_ready = False
    publish_rdma_ready = False
    source_tokens = None

    @cute.kernel
    def kernel(self, received: Optional[tuple], rows: cute.Tensor, output: cute.Tensor,
               ready_groups: Optional[cute.Tensor]):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        vec = self.elements_per_thread
        lanes = self.lanes_per_group
        tokens = rows.shape[1]
        if cutlass.const_expr(self.source_tokens is not None):
            tokens = self.source_tokens
        total_vectors = tokens * (self.hidden // vec)
        peer_count = len(self.counts)
        depth = self.prefetch_peers
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), output.element_type, num_bits_per_copy=128,
        )
        staged = cute.make_rmem_tensor((8, vec // 8, depth), output.element_type)
        values = cute.make_rmem_tensor((8, vec // 8), output.element_type)
        total = cute.make_rmem_tensor((8, vec // 8), cutlass.Float32)
        peer_rows = cute.make_rmem_tensor((max(1, peer_count),), cutlass.Int32)
        rounds = 1
        if cutlass.const_expr(self.publish_ready):
            rounds = cute.ceil_div(self.hidden // self.tile_n, self.group_tiles)
        for completed_group in cutlass.range(rounds):
            first_vector = cutlass.Int32(0)
            end_vector = cutlass.Int32(total_vectors)
            if cutlass.const_expr(self.publish_ready):
                vectors_per_tile = tokens * (self.tile_n // vec)
                first_vector = completed_group * self.group_tiles * vectors_per_tile
                end_vector = cutlass.min(first_vector + self.group_tiles * vectors_per_tile,
                                        total_vectors)
            round_chunks = cute.ceil_div(end_vector - first_vector, self.threads_per_cta)
            for chunk in cutlass.range(block, round_chunks, self.num_ctas):
                vector = first_vector + chunk * self.threads_per_cta + thread
                if vector < end_vector:
                    group = vector // lanes
                    token = group // (self.hidden // (lanes * vec))
                    column = group % (self.hidden // (lanes * vec)) * (lanes * vec)
                    if cutlass.const_expr(self.column_group_major):
                        groups_per_tile_row = self.tile_n // (lanes * vec)
                        groups_per_tile = max(1, tokens) * groups_per_tile_row
                        tile = group // groups_per_tile
                        within_tile = group % groups_per_tile
                        token = within_tile // groups_per_tile_row
                        column = (tile * self.tile_n
                                  + within_tile % groups_per_tile_row * (lanes * vec))
                    column += vector % lanes * 8
                    cancelled = cutlass.Int32(0)
                    if cutlass.const_expr(self.wait_for_inputs):
                        ready_ptr = cute.make_ptr(cutlass.Int32, ready_groups[0],
                                                  cute.AddressSpace.gmem, assumed_align=4)
                        cancel_ptr = cute.make_ptr(cutlass.Int32, ready_groups[1],
                                                   cute.AddressSpace.gmem, assumed_align=4)
                        required = column // (self.tile_n * self.group_tiles) + 1
                        ready = cute.arch.load(ready_ptr.llvm_ptr, cutlass.Int32,
                                               sem="acquire", scope="sys")
                        cancelled = cute.arch.load(cancel_ptr.llvm_ptr, cutlass.Int32,
                                                   sem="acquire", scope="sys")
                        while ready < required and cancelled == 0:
                            ready = cute.arch.load(ready_ptr.llvm_ptr, cutlass.Int32,
                                                   sem="acquire", scope="sys")
                            cancelled = cute.arch.load(cancel_ptr.llvm_ptr, cutlass.Int32,
                                                       sem="acquire", scope="sys")
                    if cancelled == 0:
                        for peer in cutlass.range_constexpr(peer_count):
                            peer_rows[peer] = rows[peer, token]
                        total.fill(0.0)
                        for first in cutlass.range_constexpr(0, peer_count, depth):
                            for slot in cutlass.range_constexpr(min(depth, peer_count - first)):
                                peer = first + slot
                                row = peer_rows[peer]
                                if row >= 0:
                                    for stripe in cutlass.range_constexpr(vec // 8):
                                        col = column + stripe * lanes * 8
                                        offset = row * self.hidden + col
                                        if cutlass.const_expr(self.tile_n > 0):
                                            offset = ((col // self.tile_n * self.counts[peer] + row)
                                                      * self.tile_n + col % self.tile_n)
                                        source = cute.make_tensor(
                                            received[peer].iterator + cute.assume(offset, divby=8),
                                            cute.make_layout((8,)),
                                        )
                                        cute.copy(copy_atom, source, staged[None, stripe, slot])
                            for slot in cutlass.range_constexpr(min(depth, peer_count - first)):
                                if peer_rows[first + slot] >= 0:
                                    total.store(total.load()
                                                + staged[None, None, slot].load().to(cutlass.Float32))
                        values.store(total.load().to(output.element_type))
                        if cutlass.const_expr(self.output_tile_n > 0):
                            # Each original-source domain has its own packed tiled
                            # span. A global tile stride would interleave domains.
                            output_base = cutlass.Int32(0)
                            output_row = token
                            output_rows = cutlass.Int32(rows.shape[1])
                            for domain in cutlass.range_constexpr(len(self.output_domains)):
                                start, count = self.output_domains[domain]
                                if token >= start and token < start + count:
                                    output_base = cutlass.Int32(start * self.hidden)
                                    output_row = token - start
                                    output_rows = cutlass.Int32(count)
                            for stripe in cutlass.range_constexpr(vec // 8):
                                col = column + stripe * lanes * 8
                                output_offset = (output_base
                                                 + (col // self.output_tile_n * output_rows + output_row)
                                                 * self.output_tile_n + col % self.output_tile_n)
                                destination = cute.make_tensor(
                                    output.iterator + cute.assume(output_offset, divby=8),
                                    cute.make_layout((8,)),
                                )
                                cute.copy(copy_atom, values[None, stripe], destination)
                        else:
                            destination = cute.make_tensor(
                                output.iterator + cute.assume(token * self.hidden + column, divby=8),
                                cute.make_layout((8, vec // 8), stride=(1, lanes * 8)),
                            )
                            cute.copy(copy_atom, values, destination)
            if cutlass.const_expr(self.publish_ready):
                # All ordinary global stores in this CTA happen before its
                # release slot. CTA 0 acquires every slot before system release.
                cute.arch.sync_threads()
                if thread == 0:
                    slots = cute.make_ptr(cutlass.Int32, ready_groups[2],
                                          cute.AddressSpace.gmem, assumed_align=4)
                    done = cute.make_ptr(cutlass.Int32, ready_groups[3],
                                         cute.AddressSpace.gmem, assumed_align=4)
                    cancel = cute.make_ptr(cutlass.Int32, ready_groups[1],
                                           cute.AddressSpace.gmem, assumed_align=4)
                    cancelled = cute.arch.load(cancel.llvm_ptr, cutlass.Int32,
                                               sem="acquire", scope="sys")
                    if cancelled == 0:
                        cute.arch.store((slots + completed_group * self.num_ctas + block).llvm_ptr,
                                        cutlass.Int32(1), sem="release", scope="gpu")
                    if block == 0:
                        for peer_cta in cutlass.range(self.num_ctas):
                            arrived = cute.arch.load(
                                (slots + completed_group * self.num_ctas + peer_cta).llvm_ptr,
                                cutlass.Int32, sem="acquire", scope="gpu")
                            while arrived == 0 and cancelled == 0:
                                arrived = cute.arch.load(
                                    (slots + completed_group * self.num_ctas + peer_cta).llvm_ptr,
                                    cutlass.Int32, sem="acquire", scope="gpu")
                                cancelled = cute.arch.load(cancel.llvm_ptr, cutlass.Int32,
                                                           sem="acquire", scope="sys")
                        if cancelled == 0:
                            cute.arch.store(done.llvm_ptr, completed_group + 1,
                                            sem="release", scope="sys")
                            if cutlass.const_expr(self.publish_rdma_ready):
                                host_done = cute.make_ptr(cutlass.Int32, ready_groups[4],
                                                          cute.AddressSpace.gmem, assumed_align=4)
                                cute.arch.store(host_done.llvm_ptr, completed_group + 1,
                                                sem="release", scope="sys")
                # Keep CTA 0's next group behind publication of this one.
                cute.arch.sync_threads()


class RegisterInterNodeKernel(RegisterIntraNodeKernel):
    """Post-RDMA reduction: no polling or publication; rows index original tokens."""

    wait_for_inputs = False
