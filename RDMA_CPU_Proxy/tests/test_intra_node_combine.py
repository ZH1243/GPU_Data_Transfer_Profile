"""CPU mapping checks; opt-in Hopper numerical/readiness checks.

RDMA_PROXY_TEST_CUDA=1 QUACK_ROOT=/path/to/quack_for_communication_overlap \
python -m unittest discover -s RDMA_CPU_Proxy/tests -p test_intra_node_combine.py -v
"""
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from intra_node_combine import contribution_indices


class MappingTests(unittest.TestCase):
    def test_sparse_domains_and_self_compaction(self):
        # Both domains have different gaps. Peer 2 is empty in the first one.
        domains = [[1, 2, 3, 8], [4, 12, 1]]
        for masks in domains:
            indices = contribution_indices(masks, 4)
            actual = [0] * len(masks)
            for peer, rows in enumerate(indices):
                packed = [(peer + 1) * 10 + row for row in rows]
                for row, value in zip(rows, packed):
                    actual[row] += value
            expected = [sum((peer + 1) * 10 + row for peer in range(4)
                            if mask & (1 << peer)) for row, mask in enumerate(masks)]
            self.assertEqual(actual, expected)
        self.assertEqual(contribution_indices(domains[0], 4)[2], [])
        self.assertEqual(contribution_indices(domains[0], 4)[0], [0, 2])
        self.assertEqual(contribution_indices([], 8), [[] for _ in range(8)])
        for masks in ([0], [256], [-1]):
            with self.assertRaises(ValueError):
                contribution_indices(masks, 8)


@unittest.skipUnless(os.environ.get("RDMA_PROXY_TEST_CUDA") == "1", "requires Hopper opt-in")
class KernelTests(unittest.TestCase):
    def test_register_prefetch_group_readiness_and_reuse(self):
        import torch
        root = os.environ.get("QUACK_ROOT", str(Path(__file__).resolve().parents[3] /
                                                "quack_for_communication_overlap"))
        sys.path.insert(0, root)
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from intra_node_combine_kernel import RegisterIntraNodeKernel
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("requires SM90")
        width, tile, tokens = 512, 128, 56
        for group_tiles in (1, 2):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(group_tiles=group_tiles, dtype=dtype):
                    # Two original-source domains, each reduced over eight GPUs.
                    masks = [[1 + ((row * 37) % 127) for row in range(37)],
                             [128 if row % 2 else 3 for row in range(19)]]
                    selected = []
                    offset = 0
                    for domain in masks:
                        selected += [[r + offset for r in rows]
                                     for rows in contribution_indices(domain, 8) if rows]
                        offset += len(domain)
                    rows = torch.full((len(selected), tokens), -1, dtype=torch.int32, device="cuda")
                    indices = [torch.tensor(x, dtype=torch.int64, device="cuda") for x in selected]
                    counts = tuple(len(x) for x in selected)
                    buffers = [torch.empty(width * n, dtype=dtype, device="cuda") for n in counts]
                    for p, idx in enumerate(indices):
                        rows[p, idx] = torch.arange(idx.numel(), dtype=torch.int32, device="cuda")
                    output = torch.empty((tokens, width), dtype=dtype, device="cuda")
                    ready = torch.zeros(1, dtype=torch.int32, device="cuda")
                    cancel = torch.zeros_like(ready)
                    addresses = torch.tensor([ready.data_ptr(), cancel.data_ptr()], dtype=torch.int64, device="cuda")
                    args = (tuple(from_dlpack(b, assumed_align=16) for b in buffers),
                            from_dlpack(rows), from_dlpack(output, assumed_align=16), from_dlpack(addresses))
                    stream = torch.cuda.Stream()
                    kernel = RegisterIntraNodeKernel(counts, width, tile, 4, 32, 512, True, 0)
                    kernel.group_tiles, kernel.prefetch_peers, kernel.lanes_per_group = group_tiles, 4, 4
                    compiled = cute.compile(kernel, *args, cuda.CUstream(stream.cuda_stream))
                    for iteration in range(2):
                        ready.zero_()
                        output.fill_(-100)
                        for b in buffers:
                            b.fill_(float('nan'))
                        values = [torch.randn((n, width), device="cuda", dtype=dtype) for n in counts]
                        expected = torch.zeros((tokens, width), device="cuda", dtype=torch.float32)
                        for idx, val in zip(indices, values):
                            expected[idx] += val.float()
                        torch.cuda.synchronize()
                        compiled(*args, cuda.CUstream(stream.cuda_stream))
                        try:
                            time.sleep(0.05)
                            torch.testing.assert_close(output, torch.full_like(output, -100),
                                                       rtol=0, atol=0)
                            # Publish real data group by group while combine polls.
                            for group in range(width // (tile * group_tiles)):
                                lo, hi = group * group_tiles, (group + 1) * group_tiles
                                for b, val, n in zip(buffers, values, counts):
                                    b.view(width // tile, n, tile)[lo:hi].copy_(
                                        val.view(n, width // tile, tile).permute(1, 0, 2)[lo:hi])
                                ready.fill_(group + 1)
                            done = torch.cuda.Event()
                            done.record(stream)
                            # Bounded host wait; failure cancels kernel polling.
                            deadline = time.monotonic() + 20
                            while not done.query() and time.monotonic() < deadline:
                                time.sleep(0.01)
                            self.assertTrue(done.query(), "combine timed out")
                            torch.testing.assert_close(output, expected.to(dtype), rtol=0, atol=0)
                        finally:
                            cancel.fill_(1)
                            torch.cuda.current_stream().synchronize()
                            stream.synchronize()
                            cancel.zero_()
                    # Cancellation also releases a launch with no ready groups.
                    ready.zero_()
                    torch.cuda.synchronize()
                    compiled(*args, cuda.CUstream(stream.cuda_stream))
                    cancel.fill_(1)
                    torch.cuda.current_stream().synchronize()
                    stream.synchronize()


if __name__ == "__main__":
    unittest.main()
