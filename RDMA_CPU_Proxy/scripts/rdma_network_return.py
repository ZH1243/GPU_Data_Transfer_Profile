"""Registered reverse-RDMA storage; native coordinator releases ready groups."""
import ctypes
import hashlib


class NetworkReturnBuffer(ctypes.Structure):
    _fields_ = [("peer", ctypes.c_int32), ("source", ctypes.c_size_t),
                ("destination", ctypes.c_size_t), ("source_rows", ctypes.c_uint64),
                ("destination_rows", ctypes.c_uint64)]


class NetworkReturn:
    def __init__(self, combine, options):
        exchange = combine.exchange
        self.exchange = exchange
        # Retain registered storage even if later kernel compilation/setup fails.
        # ReturnExchange releases this only after native shutdown.
        exchange.network_return = self
        self.check_enabled = options.check_rdma_return_token_results
        self.sources, self.received, self.storage = {}, {}, []
        torch = exchange.torch
        query = exchange.library.rdma_proxy_get_rdma_return_receive_rows
        query.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int
        configure = exchange.library.rdma_proxy_configure_rdma_returns
        configure.argtypes = [ctypes.c_void_p, ctypes.POINTER(NetworkReturnBuffer)] + [ctypes.c_uint64] * 7 + [ctypes.POINTER(ctypes.c_size_t)]
        configure.restype = ctypes.c_int
        check = exchange.library.rdma_proxy_check_rdma_return
        check.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p]
        check.restype = ctypes.c_int
        descriptors = []
        for peer in combine.nodes:
            if peer == exchange.node:
                continue
            source = combine.outputs[peer]
            if not source.is_contiguous():
                raise ValueError("RDMA return source must have packed N-tile-major layout")
            rows = ctypes.c_uint64()
            exchange.require(query(exchange.handle, peer, ctypes.byref(rows)))
            storage = torch.empty(max(1, rows.value * combine.width), dtype=source.dtype,
                                  device=combine.device)
            destination = storage[:rows.value * combine.width].view(
                combine.width // combine.tile, rows.value, combine.tile)
            # Empty domains still participate in group markers, and need valid MRs.
            source_storage = source if source.numel() else torch.empty(1, dtype=source.dtype, device=combine.device)
            self.storage.extend((source_storage, storage))
            self.sources[peer], self.received[peer] = source, destination
            descriptors.append(NetworkReturnBuffer(peer, source_storage.data_ptr(), storage.data_ptr(),
                                                   source.shape[1], rows.value))
        if not descriptors:
            raise ValueError("RDMA returning requires a remote source node")
        self.descriptors = (NetworkReturnBuffer * len(descriptors))(*descriptors)
        prefix = ctypes.c_size_t()
        torch.cuda.synchronize(combine.device)
        exchange.require(configure(exchange.handle, self.descriptors, len(descriptors),
                                   combine.width, combine.tile, combine.group_tiles,
                                   combine.output.element_size(), options.rdma_return_chunk_bytes,
                                   max(1, int(options.return_token_timeout * 1000)), ctypes.byref(prefix)))
        self.device_prefix = prefix.value
        print(f"RDMA returns: registered combine outputs, peers={list(self.received)}, "
              f"groups={combine.num_groups}, chunk_bytes={options.rdma_return_chunk_bytes}", flush=True)

    @staticmethod
    def digest(tensor):
        # Bound temporary CPU storage independently of total return size.
        import torch
        flat = tensor.view(-1).view(dtype=torch.uint8)
        digest = hashlib.sha256()
        for start in range(0, flat.numel(), 16 * 1024 * 1024):
            digest.update(flat[start:start + 16 * 1024 * 1024].cpu().numpy().tobytes())
        return digest.hexdigest().encode('ascii')

    def check(self):
        if not self.check_enabled:
            return
        for peer in sorted(self.sources):
            self.exchange.require(self.exchange.library.rdma_proxy_check_rdma_return(
                self.exchange.handle, peer, self.digest(self.sources[peer]), self.digest(self.received[peer])))
        print("RDMA return transport check: PASSED", flush=True)
