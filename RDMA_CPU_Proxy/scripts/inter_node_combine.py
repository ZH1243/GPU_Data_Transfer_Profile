"""Post-RDMA node reduction, writing directly in original source-token order."""
import ctypes


def inverse_token_rows(token_lists, num_tokens):
    """Validate x3 domains and invert them independently; absent rows stay -1."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be nonnegative")
    result = []
    for tokens in token_lists:
        inverse = [-1] * num_tokens
        for row, token in enumerate(tokens):
            if token < 0 or token >= num_tokens or inverse[token] != -1:
                raise ValueError("x3 contains an out-of-range or duplicate token")
            inverse[token] = row
        result.append(inverse)
    return result


class TokenOrderCombine:
    """All inputs must be complete before run(); run joins before buffer reuse.

    Inputs are in ascending node order. Empty domains are omitted from the
    compiled tuple, but tokens without any contribution are still written zero.
    """

    def __init__(self, inputs, token_lists, num_tokens, check=False, ctas=None):
        import torch
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from intra_node_combine_kernel import RegisterInterNodeKernel

        if not inputs or len(inputs) != len(token_lists):
            raise ValueError("inter-node combine requires one x3 list per input")
        if ctas is not None and ctas <= 0:
            raise ValueError("inter-node combine CTA count must be positive")
        first = inputs[0]
        if first.ndim != 3 or first.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("inter-node combine requires tiled BF16/FP16 inputs")
        tiles, _, tile = first.shape
        width = tiles * tile
        if not width or tile % 8 or num_tokens * width > 2**31 - 1:
            raise ValueError("invalid inter-node width/alignment or int32 index overflow")
        for tensor, tokens in zip(inputs, token_lists):
            if (tensor.shape != (tiles, len(tokens), tile) or not tensor.is_contiguous()
                    or tensor.dtype != first.dtype or tensor.device != first.device
                    or tensor.numel() > 2**31 - 1):
                raise ValueError("inter-node input layout/rows/dtype/device mismatch")
        inverse = inverse_token_rows(token_lists, num_tokens)
        self.torch, self.cuda = torch, cuda
        self.inputs, self.token_lists = tuple(inputs), token_lists
        self.num_tokens, self.width = num_tokens, width
        self.check_enabled = check
        self.output = torch.empty((num_tokens, width), dtype=first.dtype, device=first.device)
        self.stream = torch.cuda.Stream(device=first.device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.compiled = None
        self.num_ctas = (ctas if ctas is not None else
                         torch.cuda.get_device_properties(first.device).multi_processor_count)
        active = [i for i, tokens in enumerate(token_lists) if tokens]
        self.rows = torch.tensor([inverse[i] for i in active], dtype=torch.int32,
                                 device=first.device) if active and num_tokens else torch.full(
                                     (max(1, len(active)), max(1, num_tokens)), -1,
                                     dtype=torch.int32, device=first.device)
        self.buffers = tuple(inputs[i].view(-1) for i in active)
        if num_tokens:
            self.arguments = (
                tuple(from_dlpack(b, assumed_align=16) for b in self.buffers) if active else None,
                from_dlpack(self.rows), from_dlpack(self.output, assumed_align=16), None,
            )
            vec = 32
            while tile % vec:
                vec //= 2
            lanes = 32
            while (tile // vec) % lanes:
                lanes //= 2
            kernel = RegisterInterNodeKernel(
                tuple(len(token_lists[i]) for i in active), width, tile,
                self.num_ctas,
                vec, 256, False, 0)
            kernel.source_tokens = num_tokens
            kernel.lanes_per_group = lanes
            kernel.prefetch_peers = min(4, max(1, len(active)))
            self.compiled = cute.compile(kernel, *self.arguments,
                                         cuda.CUstream(self.stream.cuda_stream))
        # Setup uploads must precede the first launch on the private stream.
        torch.cuda.synchronize(first.device)

    def run(self):
        """Caller has joined native RDMA and local combine, including visibility."""
        self.start.record(self.stream)
        try:
            if self.compiled is not None:
                self.compiled(*self.arguments, self.cuda.CUstream(self.stream.cuda_stream))
            self.end.record(self.stream)
        finally:
            # Also join a partially successful launch before exceptional teardown.
            self.stream.synchronize()
        self.elapsed_ms = self.start.elapsed_time(self.end)
        if self.check_enabled:
            self.check()

    def check(self):
        """Independent scatter/add reference, bounded to 4096 output rows."""
        torch = self.torch
        for start in range(0, self.num_tokens, 4096):
            end = min(start + 4096, self.num_tokens)
            expected = torch.zeros((end - start, self.width), dtype=torch.float32,
                                   device=self.output.device)
            for tensor, tokens in zip(self.inputs, self.token_lists):
                selected = [(row, token - start) for row, token in enumerate(tokens)
                            if start <= token < end]
                if selected:
                    rows, destinations = zip(*selected)
                    rows = torch.tensor(rows, dtype=torch.int64, device=tensor.device)
                    destinations = torch.tensor(destinations, dtype=torch.int64, device=tensor.device)
                    values = tensor[:, rows, :].permute(1, 0, 2).reshape(len(selected), self.width)
                    expected.index_add_(0, destinations, values.float())
            torch.testing.assert_close(self.output[start:end], expected.to(self.output.dtype),
                                       rtol=0, atol=0)
        print("Inter-node combine reference check: PASSED", flush=True)


class InterNodeCombine(TokenOrderCombine):
    def __init__(self, combine, options):
        exchange = combine.exchange
        query = exchange.library.rdma_proxy_get_outgoing_token_indices
        query.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64),
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                          ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int
        inputs, token_lists = [], []
        num_tokens = None
        for node in combine.nodes:
            count, total = ctypes.c_uint64(), ctypes.c_uint64()
            exchange.require(query(exchange.handle, node, None, 0,
                                   ctypes.byref(count), ctypes.byref(total)))
            indices = (ctypes.c_uint64 * count.value)()
            exchange.require(query(exchange.handle, node, indices, count.value,
                                   ctypes.byref(count), ctypes.byref(total)))
            if num_tokens is not None and num_tokens != total.value:
                raise ValueError("inconsistent original token count")
            num_tokens = total.value
            token_lists.append(list(indices))
            inputs.append(combine.outputs[node] if node == exchange.node
                          else combine.rdma_return.received[node])
        super().__init__(inputs, token_lists, num_tokens, options.check_inter_node_combine,
                         ctas=options.inter_node_combine_ctas)
        print(f"Inter-node combine: CTAs={self.num_ctas}, output={tuple(self.output.shape)} row-major, "
              "original token order; runs after local combine and any RDMA returns", flush=True)
