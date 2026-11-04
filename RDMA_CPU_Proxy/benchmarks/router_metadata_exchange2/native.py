"""ctypes ABI; no PyTorch dependency and no metadata copies in the timed path."""
import ctypes as C


class Stats(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        "exchange_gpu_ms", "sender_completion_wait_ms", "peer_completion_wait_ms",
        "headers_d2h_ms", "payload_d2h_ms", "install_ms", "native_ms")]
    _fields_ += [(name, C.c_uint64) for name in (
        "received_index_bytes", "peer_received_index_bytes", "header_bytes")]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


class Proxy:
    def __init__(self, library, rank, device, tokens, shm_name, timeout, validate, core):
        self.lib = C.CDLL(str(library))  # CDLL releases the GIL in native calls.
        self.handle = C.c_void_p()
        signatures = {
            "rme2_abi_version": ([], C.c_int),
            "rme2_error": ([], C.c_char_p),
            "rme2_create": ([C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_double,
                              C.c_int, C.c_int, C.POINTER(C.c_void_p)], C.c_int),
            "rme2_export": ([C.c_void_p, C.c_void_p], C.c_int),
            "rme2_connect": ([C.c_void_p, C.c_void_p], C.c_int),
            "rme2_unlink": ([C.c_void_p], C.c_int),
            "rme2_disconnect": ([C.c_void_p], C.c_int),
            "rme2_submit": ([C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p, C.c_void_p], C.c_int),
            "rme2_wait": ([C.c_void_p, C.POINTER(Stats)], C.c_int),
            "rme2_view": ([C.c_void_p, C.c_int, C.POINTER(C.POINTER(C.c_int32)),
                            C.POINTER(C.POINTER(C.c_int32)), C.POINTER(C.c_int)], C.c_int),
            "rme2_destroy": ([C.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        if self.lib.rme2_abi_version() != 1:
            raise RuntimeError("native ABI mismatch; rebuild the library")
        self.check(self.lib.rme2_create(rank, device, tokens, shm_name.encode(), timeout,
                                       int(validate), core, C.byref(self.handle)))

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.rme2_error().decode())

    def export(self):
        handle = C.create_string_buffer(64)
        self.check(self.lib.rme2_export(self.handle, handle))
        return handle.raw

    def connect(self, handles):
        if len(handles) != 8 or any(len(h) != 64 for h in handles):
            raise ValueError("expected eight CUDA IPC handles")
        data = C.create_string_buffer(b"".join(handles))
        self.check(self.lib.rme2_connect(self.handle, data))

    def unlink(self):
        self.check(self.lib.rme2_unlink(self.handle))

    def submit(self, iteration, plan, stream):
        self.check(self.lib.rme2_submit(self.handle, iteration, plan.expert_offsets.data_ptr(),
                                       plan.expert_token_indices.data_ptr(), stream.cuda_stream))

    def wait(self):
        stats = Stats()
        self.check(self.lib.rme2_wait(self.handle, C.byref(stats)))
        return stats.as_dict()

    def snapshot(self, source):
        # Explicitly outside timing, for exact-content verification only.
        offsets, indices, count = C.POINTER(C.c_int32)(), C.POINTER(C.c_int32)(), C.c_int()
        self.check(self.lib.rme2_view(self.handle, source, C.byref(offsets),
                                     C.byref(indices), C.byref(count)))
        return list(offsets[:17]), list(indices[:count.value])

    def disconnect(self):
        self.check(self.lib.rme2_disconnect(self.handle))

    def close(self):
        if self.handle:
            self.lib.rme2_destroy(self.handle)
            self.handle = C.c_void_p()
