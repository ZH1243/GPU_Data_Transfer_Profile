"""Remote Hopper check: validate producer output BEFORE any exchange can repack it.

Run with python check_direct_x3_x4_gpu.py on either remote node. Requires one GPU.
Checks both destination slices, eager/graph launch, partial chunks, changing
lengths, and byte guards around the borrowed output buffer.
"""
import struct
import sys
from pathlib import Path

from reference import make_routes, metadata_two


def main():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "get_expert_token_idx_list_gpu"))
    from metadata_cute import MetadataPlan

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    source = 3  # Test nonzero source-relative mask ordering on physical GPU 0.
    guard, sentinel = 64, 0xA5
    for tokens in (1, 33, 257, 8193):
        for remote in (0, 1):
            for mode in ("eager", "graph"):
                with torch.cuda.stream(stream):
                    r = torch.arange(8, dtype=torch.int32, device="cuda").expand(tokens, 8).clone()
                    storage = torch.full((2 * guard + 5 * tokens,), sentinel,
                                         dtype=torch.uint8, device="cuda")
                    plan = MetadataPlan(r, source, rdma_x3_x4_payload=storage.data_ptr() + guard,
                                        remote_output_node=remote)
                    plan.launch(stream)
                stream.synchronize()
                graph = None
                if mode == "graph":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        plan.launch(stream)
                for iteration, pattern in enumerate(("affine", "hot-low", "hot-high", "affine")):
                    routes = make_routes(tokens, 1234 + iteration, pattern)
                    offsets, values, x3, x4, node_offsets = metadata_two(routes, source)
                    with torch.cuda.stream(stream):
                        storage.fill_(sentinel)
                        r.copy_(torch.tensor(routes, dtype=torch.int32, device="cpu"))
                        if graph is None:
                            plan.launch(stream)
                        else:
                            graph.replay()
                    stream.synchronize()
                    first, last = node_offsets[remote:remote + 2]
                    payload = b"".join(struct.pack("<i", t) for t in x3[first:last]) + bytes(x4[first:last])
                    expected = bytes([sentinel]) * guard + payload
                    expected += bytes([sentinel]) * (guard + 5 * tokens - len(payload))
                    context = f"tokens={tokens} remote={remote} mode={mode} pattern={pattern}"
                    if bytes(storage.cpu().tolist()) != expected:
                        raise AssertionError(f"direct payload or guard mismatch: {context}")
                    actual = (plan.expert_offsets.cpu().tolist(),
                              plan.expert_token_indices[:len(values)].cpu().tolist(),
                              plan.x3[:len(x3)].cpu().tolist(), plan.x4[:len(x4)].cpu().tolist(),
                              plan.node_offsets.cpu().tolist())
                    if actual != (offsets, values, x3, x4, node_offsets):
                        raise AssertionError(f"canonical metadata changed: {context}")
                print(f"passed tokens={tokens} remote={remote} mode={mode}", flush=True)
                # Release captured pointers before their allocation owners.
                del graph, plan, storage


if __name__ == "__main__":
    main()
