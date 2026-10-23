"""Same-node return storage and IPC control plane for the embedded proxy."""

import ctypes
import os


class ReturnCopy(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_size_t), ("destination", ctypes.c_size_t),
        ("capacity_bytes", ctypes.c_uint64), ("row_index", ctypes.c_uint64),
        ("peer", ctypes.c_int32),
    ]


class ReturnExchange:
    """One receive slot per (computing peer, original source node).

    C++ owns the return stream. All exported tensors, pinned counts, and the down
    event stay alive until native shutdown and the imports-closed barrier.
    """

    def __init__(self, torch, library, handle, down, options, node, rank, device, iterations):
        from run.token_exchange import CudaCopies, SharedExchange
        from run.hopper_stream_gather_table_gemm import export_cuda_ipc_allocation

        self.torch, self.library, self.handle, self.down = torch, library, handle, down
        self.rank, self.device = rank, device
        self.world = int(os.environ["LOCAL_WORLD_SIZE"])
        self.control = SharedExchange(rank, self.world, options.return_token_timeout,
                                      namespace=f"rdma-return-node{node}")
        self.exports = []
        self.cuda = CudaCopies()
        self.outputs = down.inputs.output_buffers()
        self.check_enabled = options.check_return_token_results
        self.received = {}
        self.received_valid_rows = {}
        self.rows_host = torch.empty(len(self.outputs), dtype=torch.int64, pin_memory=True)
        self.down_done = torch.cuda.Event()
        self.down_done.record()
        torch.cuda.synchronize(device)
        self.closed = False
        try:
            if self.world < 2:
                raise ValueError("return-token-results requires at least two local GPUs")
            sources = list(down.up.source_pairs)
            nodes = sorted({source_node for source_node, _ in sources})
            if set(sources) != {(n, g) for n in nodes for g in range(self.world)}:
                raise ValueError("return-token-results requires all global source buffers")
            signature = [nodes, str(self.outputs[0].dtype), down.combine_rows,
                         int(down.up.x_buffers[0].shape[1]), options.down_n_tile_major_output,
                         options.down_tile_n, options.down_max_swizzle_size,
                         options.return_token_results_per_group, self.check_enabled, iterations,
                         options.intra_node_combine, options.check_intra_node_combine,
                         options.intra_node_combine_n_tile_major_output,
                         os.environ.get("CUDA_VISIBLE_DEVICES")]
            settings = self.control.gather("configuration", {
                "signature": signature, "device": device, "sources": sources,
            })
            if any(item["signature"] != signature for item in settings):
                raise ValueError("return layouts/options must match across local ranks")
            self.settings = settings
            for peer in range(self.world):
                if peer != rank:
                    for source_node in nodes:
                        self.received[peer, source_node] = torch.empty_like(self.outputs[0])
            allocations = {}
            for (peer, source_node), tensor in self.received.items():
                exported = export_cuda_ipc_allocation(tensor)
                self.exports.append(exported)
                allocations[f"{peer}:{source_node}"] = [
                    exported.handle_hex, exported.allocation_offset_bytes,
                    tensor.numel() * tensor.element_size(),
                ]
            # Optional validation imports producing outputs to compare exact bytes.
            output_allocations = {}
            if self.check_enabled:
                for index, tensor in enumerate(self.outputs):
                    exported = export_cuda_ipc_allocation(tensor)
                    self.exports.append(exported)
                    output_allocations[str(index)] = [
                        exported.handle_hex, exported.allocation_offset_bytes,
                    ]
            torch.cuda.synchronize(device)
            self.metadata = self.control.gather(
                "allocations", {"received": allocations, "outputs": output_allocations})
            copies = []
            for index, (source_node, peer) in enumerate(sources):
                if peer == rank:
                    continue
                entry = self.metadata[peer]["received"][f"{rank}:{source_node}"]
                destination = self.cuda.peer_pointer(entry[:2], device, settings[peer]["device"])
                tensor = self.outputs[index]
                capacity = min(entry[2], tensor.numel() * tensor.element_size())
                copies.append(ReturnCopy(tensor.data_ptr(), destination, capacity, index, peer))
            configure = library.rdma_proxy_configure_return_copies
            configure.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ReturnCopy), ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32,
                ctypes.c_size_t,
            ]
            configure.restype = ctypes.c_int
            self.arm_fn = library.rdma_proxy_arm_return_copies
            self.arm_fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            self.arm_fn.restype = ctypes.c_int
            width = int(down.up.x_buffers[0].shape[1])
            tile = options.down_tile_n if options.down_n_tile_major_output else 0
            group_tiles = min(options.down_max_swizzle_size,
                              (width + options.down_tile_n - 1) // options.down_tile_n)
            descriptors = (ReturnCopy * len(copies))(*copies)
            self.require(configure(
                handle, descriptors, len(copies),
                ctypes.cast(self.rows_host.data_ptr(), ctypes.POINTER(ctypes.c_int64)),
                len(self.outputs), width, tile, group_tiles, self.outputs[0].element_size(),
                int(options.return_token_results_per_group), self.down_done.cuda_event))
            down.return_exchange = self
            self.control.gather("configured")
            print(f"Return copies configured: receive_slots={len(self.received)} "
                  f"mode={'per-group' if options.return_token_results_per_group else 'whole-buffer'} "
                  "stream=dedicated-nonblocking; return completion/barrier excluded from GEMM timing",
                  flush=True)
        except BaseException as error:
            self.control.state("error", str(error))
            # Retain allocations/imports on failure until process exit.
            _failed_exchanges.append(self)
            raise

    def require(self, status):
        if status:
            message = self.library.rdma_proxy_last_error()
            raise RuntimeError(message.decode() if message else "return-copy operation failed")

    def publish_counts(self):
        # Called on the compute stream after mapping, before down GEMM. Seeing a
        # positive system-release ready prefix establishes that this D2H is done.
        self.rows_host.copy_(self.down.combine_valid_rows, non_blocking=True)

    def arm(self, iteration):
        # The event must be recorded before C++ may query this generation.
        self.down_done.record()
        self.require(self.arm_fn(self.handle, iteration))

    def finish_iteration(self, iteration):
        # Caller already waited for native outgoing-return completion. The
        # all-rank barrier establishes incoming completion before reuse.
        counts = self.rows_host.tolist()
        all_counts = self.control.gather(f"done-{iteration}", counts)
        for peer, source_node in self.received:
            index = self.settings[peer]["sources"].index([source_node, self.rank])
            self.received_valid_rows[peer, source_node] = all_counts[peer][index]
        if self.check_enabled:
            self.check()
            self.control.gather(f"checked-{iteration}")

    def check(self):
        """Exact physical-prefix comparison; GEMM arithmetic has separate tests."""
        stream = self.torch.cuda.current_stream(self.device).cuda_stream
        width = int(self.down.up.x_buffers[0].shape[1])
        tile = self.down.combine_tile_n
        physical_width = ((width + tile - 1) // tile) * tile if tile else width
        for (peer, source_node), actual in self.received.items():
            count = self.received_valid_rows[peer, source_node]
            elements = count * physical_width
            if not elements:
                continue
            index = self.settings[peer]["sources"].index([source_node, self.rank])
            pointer = self.cuda.peer_pointer(
                self.metadata[peer]["outputs"][str(index)], self.device,
                self.settings[peer]["device"])
            expected = self.torch.empty(elements, dtype=actual.dtype, device=actual.device)
            self.cuda.call("cudaMemcpyPeerAsync", expected.data_ptr(), self.device, pointer,
                           self.settings[peer]["device"],
                           elements * actual.element_size(), stream)
            self.cuda.call("cudaStreamSynchronize", stream)
            self.torch.testing.assert_close(
                actual.view(-1)[:elements].view(self.torch.uint8),
                expected.view(self.torch.uint8), rtol=0, atol=0)
        print("Return-copy reference check: PASSED", flush=True)

    def close(self, success):
        """Called only after native shutdown has stopped and drained submissions."""
        if self.closed:
            return
        self.closed = True
        if not success:
            self.control.state("error", "return exchange aborted")
            _failed_exchanges.append(self)
            return
        try:
            self.control.gather("workers_stopped")
            self.cuda.close()
            self.control.gather("imports_closed")
            for exported in self.exports:
                exported.release()
            self.control.close()
            # Counter release alone does not destroy the exported storages.
            # Break down -> exchange -> down so they are reclaimed before
            # PyTorch's CUDA IPC globals are destroyed at interpreter shutdown.
            self.down.intra_node_combine = None
            self.down.return_exchange = None
            self.down = None
            self.exports.clear()
            self.outputs = ()
            self.received.clear()
            self.received_valid_rows.clear()
        except BaseException:
            _failed_exchanges.append(self)
            raise


# Exceptional teardown must not free storage that another process may still use.
_failed_exchanges = []
