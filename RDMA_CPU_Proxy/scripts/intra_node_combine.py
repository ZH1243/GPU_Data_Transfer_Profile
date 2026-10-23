"""Source-side intra-node reduction, concurrent with readiness-publishing down GEMM."""
import ctypes


def contribution_indices(masks, world):
    """Output row k of peer p corresponds to the k-th selected full-input row."""
    if not 2 <= world <= 8 or any(mask <= 0 or mask >= (1 << world) for mask in masks):
        raise ValueError("invalid intra-node routing masks")
    return [[row for row, mask in enumerate(masks) if mask & (1 << peer)]
            for peer in range(world)]


class IntraNodeCombine:
    def __init__(self, exchange, options):
        import torch
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from intra_node_combine_kernel import RegisterIntraNodeKernel

        self.exchange = exchange
        self.torch, self.cuda = torch, cuda
        self.device = exchange.down.up.output.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.control_stream = torch.cuda.Stream(device=self.device)
        self.mapping_ready = torch.cuda.Event()
        self.finished = torch.cuda.Event()
        self.launched = False
        self.check_enabled = options.check_intra_node_combine
        self.cancel = torch.zeros(1, dtype=torch.int32, device=self.device)
        width = int(exchange.down.up.x_buffers[0].shape[1])
        tile = options.down_tile_n
        vec = options.intra_node_combine_elements_per_thread
        threads = options.intra_node_combine_threads_per_cta
        ctas = options.intra_node_combine_ctas
        if width % tile or tile % vec:
            raise ValueError("intra-node combine requires tile_n dividing width and elements/thread dividing tile_n")
        properties = torch.cuda.get_device_properties(self.device)
        # A persistent down grid with wave barriers must fit alongside combine.
        # Reserve a whole cluster per combine CTA to allow for fragmentation.
        available = properties.multi_processor_count - ctas * options.down_cluster_m
        if options.down_sm_count is None or options.down_sm_count > available:
            raise ValueError(f"intra-node combine requires --down-sm-count <= {available} "
                             "to leave room for persistent combine CTAs")
        query = exchange.library.rdma_proxy_get_intra_node_source_masks
        query.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8),
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int
        publication = exchange.library.rdma_proxy_get_return_ready_publication
        publication.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                               ctypes.POINTER(ctypes.c_size_t)]
        publication.restype = ctypes.c_int
        host, device = ctypes.c_size_t(), ctypes.c_size_t()
        exchange.require(publication(exchange.handle, ctypes.byref(host), ctypes.byref(device)))
        self.addresses = torch.tensor([device.value, self.cancel.data_ptr()],
                                     dtype=torch.int64, device=self.device)
        self.nodes = sorted({node for node, _ in exchange.down.up.source_pairs})
        self.domains = []
        self.entries = []
        total_rows = 0
        dtype = exchange.outputs[0].dtype
        for node in self.nodes:
            count = ctypes.c_uint64()
            exchange.require(query(exchange.handle, node, None, 0, ctypes.byref(count)))
            masks = (ctypes.c_uint8 * count.value)()
            exchange.require(query(exchange.handle, node, masks, count.value, ctypes.byref(count)))
            indices = contribution_indices(list(masks), exchange.world)
            self.domains.append((node, total_rows, count.value))
            for peer, selected in enumerate(indices):
                if peer == exchange.rank:
                    source = exchange.down.up.source_pairs.index((node, peer))
                    buffer = exchange.outputs[source]
                else:
                    buffer = exchange.received[peer, node]
                index = torch.tensor(selected, dtype=torch.int64, device=self.device)
                self.entries.append((node, peer, total_rows, index, buffer))
            total_rows += count.value
        if total_rows * width > 2**31 - 1:
            raise ValueError("intra-node combine exceeds the kernel's int32 element-index range")
        active = [entry for entry in self.entries if entry[3].numel()]
        inverse = torch.full((max(1, len(active)), total_rows), -1,
                             dtype=torch.int32, device=self.device)
        for p, (_, _, offset, index, _) in enumerate(active):
            inverse[p, index + offset] = torch.arange(index.numel(), dtype=torch.int32, device=self.device)
        self.rows = inverse
        self.output_tile_n = tile if options.intra_node_combine_n_tile_major_output else 0
        self.total_rows = total_rows
        # Pack each domain independently so consecutive N tiles for a source
        # form one contiguous transfer span, including multi-tile groups.
        self.output = torch.empty((total_rows * width,) if self.output_tile_n else (total_rows, width),
                                  dtype=dtype, device=self.device)
        self.outputs = {
            node: (self.output.view(-1)[offset * width:(offset + count) * width]
                   .view(width // tile, count, tile)
                   if self.output_tile_n else self.output[offset:offset + count])
            for node, offset, count in self.domains
        }
        self.buffers = tuple(buffer.view(-1)[:index.numel() * width]
                             for _, _, _, index, buffer in active)
        self.compiled = None
        self.width, self.tile = width, tile
        if total_rows:
            self.arguments = (
                tuple(from_dlpack(buffer, assumed_align=16) for buffer in self.buffers)
                if self.buffers else None,
                from_dlpack(self.rows), from_dlpack(self.output, assumed_align=16),
                from_dlpack(self.addresses),
            )
            kernel = RegisterIntraNodeKernel(
                tuple(entry[3].numel() for entry in active), width, tile, ctas, vec,
                threads, True, self.output_tile_n)
            kernel.output_domains = tuple((offset, count) for _, offset, count in self.domains)
            kernel.group_tiles = min(options.down_max_swizzle_size, width // tile)
            kernel.prefetch_peers = options.intra_node_combine_prefetch_peers
            lanes = 32
            while (tile // vec) % lanes:
                lanes //= 2
            kernel.lanes_per_group = lanes
            self.compiled = cute.compile(kernel, *self.arguments,
                                         cuda.CUstream(self.stream.cuda_stream))
        # Compilation/metadata uploads finish before any readiness-gated launch.
        torch.cuda.synchronize(self.device)
        print(f"Intra-node combine: register-prefetch, rows_by_source="
              f"{[(n, c) for n, _, c in self.domains]} CTAs={ctas}, "
              f"outputs={'N-tile-major' if self.output_tile_n else 'row-major'} "
              "in staging/RDMA input-row order", flush=True)

    def launch(self):
        # Called on the down stream after mapping/reset and down launch. This
        # event was recorded BEFORE down launch, so it does not wait for down.
        self.stream.wait_event(self.mapping_ready)
        self.launched = True
        if self.compiled is not None:
            self.compiled(*self.arguments, self.cuda.CUstream(self.stream.cuda_stream))
        self.finished.record(self.stream)

    def finish(self):
        if self.launched:
            self.finished.synchronize()
            self.launched = False
        if self.check_enabled:
            self.check()

    def check(self):
        """Independent FP32 gather/add reference after all returns are complete."""
        torch = self.torch
        expected = torch.zeros((self.total_rows, self.width), dtype=torch.float32, device=self.device)
        for node, peer, offset, index, buffer in self.entries:
            count = index.numel()
            if peer == self.exchange.rank:
                source = self.exchange.down.up.source_pairs.index((node, peer))
                actual_count = int(self.exchange.down.combine_valid_rows[source].item())
                actual_index = self.exchange.down.combine_output_to_input[source, :count]
                torch.testing.assert_close(actual_index, index, rtol=0, atol=0)
            else:
                actual_count = self.exchange.received_valid_rows[peer, node]
            if count != actual_count:
                raise RuntimeError("routing masks disagree with combine valid-row count")
            if count:
                values = (buffer.view(-1)[:count * self.width]
                          .view(self.width // self.tile, count, self.tile)
                          .permute(1, 0, 2).reshape(count, self.width))
                expected[index + offset] += values.float()
        for node, offset, count in self.domains:
            actual = self.outputs[node]
            if self.output_tile_n:
                actual = actual.permute(1, 0, 2).reshape(count, self.width)
            torch.testing.assert_close(actual, expected[offset:offset + count].to(self.output.dtype),
                                       rtol=0, atol=0)
        print("Intra-node combine reference check: PASSED", flush=True)

    def abort(self):
        # Unblock polling if a CPU proxy fails before publishing the final group.
        if self.launched:
            with self.torch.cuda.stream(self.control_stream):
                self.cancel.fill_(1)
            self.control_stream.synchronize()
            self.stream.synchronize()
            self.launched = False
