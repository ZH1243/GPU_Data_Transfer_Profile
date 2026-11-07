"""ctypes ABI; no PyTorch dependency and no metadata copies in the timed path."""
import ctypes as C


class Stats(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        "metadata_launch_barrier_ms", "metadata_gpu_ms", "metadata_completion_wait_ms", "metadata_barrier_wait_ms",
        "exchange_gpu_ms", "sender_completion_wait_ms", "peer_completion_wait_ms",
        "headers_d2h_ms", "payload_d2h_ms", "payload_checks_ms", "payload_submit_ms",
        "payload_event_record_ms", "payload_completion_ms", "payload_other_ms", "install_ms", "native_ms",
        "rdma_post_ms", "rdma_poll_ms", "rdma_post_recv_ms", "rdma_work_complete_ms",
        "local_staging_gpu_ms", "local_staging_wait_ms", "staging_before_d2h_wait_ms")]
    _fields_ += [(name, C.c_uint64) for name in (
        "received_index_bytes", "peer_received_index_bytes", "header_bytes", "rdma_sent_bytes", "rdma_received_bytes",
        "x3_x4_sent_bytes", "x3_x4_received_bytes")]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


class Proxy:
    def __init__(self, library, rank, device, tokens, shm_name, timeout, validate, core,
                 per_source_readback=False, metadata_barrier=False, batch_payload_copy=False,
                 nodes=1, node=0, d2h_after_local_staging=False, local_staging_after_d2h=False):
        self.experts_per_gpu = 128 // (nodes * 8)
        self.nodes = nodes
        self.lib = C.CDLL(str(library))  # CDLL releases the GIL in native calls.
        self.handle = C.c_void_p()
        self.lib.rme2_abi_version.restype = C.c_int
        if self.lib.rme2_abi_version() != 11:
            raise RuntimeError("native ABI mismatch; rebuild the library")
        signatures = {
            "rme2_abi_version": ([], C.c_int),
            "rme2_error": ([], C.c_char_p),
            "rme2_create": ([C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_double,
                              C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p)], C.c_int),
            "rme2_rdma_init": ([C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_void_p, C.c_int], C.c_int),
            "rme2_rdma_connect": ([C.c_void_p, C.c_char_p], C.c_int),
            "rme2_set_batch_payload_copy": ([C.c_void_p, C.c_int], C.c_int),
            "rme2_set_d2h_after_local_staging": ([C.c_void_p, C.c_int], C.c_int),
            "rme2_set_local_staging_after_d2h": ([C.c_void_p, C.c_int], C.c_int),
            "rme2_export": ([C.c_void_p, C.c_void_p], C.c_int),
            "rme2_connect": ([C.c_void_p, C.c_void_p], C.c_int),
            "rme2_unlink": ([C.c_void_p], C.c_int),
            "rme2_disconnect": ([C.c_void_p], C.c_int),
            "rme2_capture_begin": ([C.c_void_p, C.c_void_p], C.c_int),
            "rme2_capture_end": ([C.c_void_p, C.c_int], C.c_int),
            "rme2_staging_capture_begin": ([C.c_void_p, C.c_void_p], C.c_int),
            "rme2_staging_capture_end": ([C.c_void_p], C.c_int),
            "rme2_submit": ([C.c_void_p, C.c_uint64] + [C.c_void_p] * 6, C.c_int),
            "rme2_wait": ([C.c_void_p, C.POINTER(Stats)], C.c_int),
            "rme2_view": ([C.c_void_p, C.c_int, C.POINTER(C.POINTER(C.c_int32)),
                            C.POINTER(C.POINTER(C.c_int32)), C.POINTER(C.c_int)], C.c_int),
            "rme2_x3_x4_view": ([C.c_void_p, C.POINTER(C.POINTER(C.c_int32)),
                                 C.POINTER(C.POINTER(C.c_uint8)), C.POINTER(C.c_int)], C.c_int),
            "rme2_destroy": ([C.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        if self.lib.rme2_abi_version() != 11:
            raise RuntimeError("native ABI mismatch; rebuild the library")
        self.check(self.lib.rme2_create(rank, device, tokens, shm_name.encode(), timeout,
                                       int(validate), core, int(per_source_readback), int(metadata_barrier),
                                       nodes, node, C.byref(self.handle)))
        try:
            self.check(self.lib.rme2_set_batch_payload_copy(self.handle, int(batch_payload_copy)))
            self.check(self.lib.rme2_set_d2h_after_local_staging(self.handle, int(d2h_after_local_staging)))
            self.check(self.lib.rme2_set_local_staging_after_d2h(self.handle, int(local_staging_after_d2h)))
        except Exception:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.rme2_error().decode())

    def export(self):
        handle = C.create_string_buffer(64)
        self.check(self.lib.rme2_export(self.handle, handle))
        return handle.raw

    def rdma_init(self, device, port, gid):
        descriptor = C.create_string_buffer(4096)
        self.check(self.lib.rme2_rdma_init(self.handle, device.encode(), port, gid, descriptor, len(descriptor)))
        return descriptor.value

    def rdma_connect(self, descriptor):
        self.check(self.lib.rme2_rdma_connect(self.handle, descriptor))

    def connect(self, handles):
        if len(handles) != 8 or any(len(h) != 64 for h in handles):
            raise ValueError("expected eight CUDA IPC handles")
        data = C.create_string_buffer(b"".join(handles))
        self.check(self.lib.rme2_connect(self.handle, data))

    def unlink(self):
        self.check(self.lib.rme2_unlink(self.handle))

    def capture_metadata(self, plan, stream, launch_barrier=False):
        # Native owns graph executable; retain explicit owners of captured pointers.
        self._metadata_plan, self._metadata_stream = plan, stream
        self.check(self.lib.rme2_capture_begin(self.handle, stream.cuda_stream))
        plan.launch(stream)
        self.check(self.lib.rme2_capture_end(self.handle, int(launch_barrier)))

    def capture_staging(self, plan, stream):
        # Separate graph: native records exchange readiness before launching it.
        # Keep the plan (and hence its source/destination tensors) alive.
        self._staging_plan, self._staging_stream = plan, stream
        self.check(self.lib.rme2_staging_capture_begin(self.handle, stream.cuda_stream))
        plan.launch(stream)
        self.check(self.lib.rme2_staging_capture_end(self.handle))

    def submit(self, iteration, plan, stream):
        route_pointers = ((plan.node_offsets.data_ptr(), plan.x3.data_ptr(), plan.x4.data_ptr())
                          if self.nodes == 2 else (None,) * 3)
        self.check(self.lib.rme2_submit(self.handle, iteration, plan.expert_offsets.data_ptr(),
                                       plan.expert_token_indices.data_ptr(), *route_pointers, stream.cuda_stream))

    def wait(self):
        stats = Stats()
        self.check(self.lib.rme2_wait(self.handle, C.byref(stats)))
        return stats.as_dict()

    def snapshot(self, source):
        # Explicitly outside timing, for exact-content verification only.
        offsets, indices, count = C.POINTER(C.c_int32)(), C.POINTER(C.c_int32)(), C.c_int()
        self.check(self.lib.rme2_view(self.handle, source, C.byref(offsets),
                                     C.byref(indices), C.byref(count)))
        return list(offsets[:self.experts_per_gpu + 1]), list(indices[:count.value])

    def snapshot_x3_x4(self):
        # Remote same-index source's slice for this node; outside timing only.
        x3, x4, count = C.POINTER(C.c_int32)(), C.POINTER(C.c_uint8)(), C.c_int()
        self.check(self.lib.rme2_x3_x4_view(self.handle, C.byref(x3), C.byref(x4), C.byref(count)))
        return list(x3[:count.value]), list(x4[:count.value])

    def disconnect(self):
        self.check(self.lib.rme2_disconnect(self.handle))

    def close(self):
        if self.handle:
            self.lib.rme2_destroy(self.handle)
            self.handle = C.c_void_p()
