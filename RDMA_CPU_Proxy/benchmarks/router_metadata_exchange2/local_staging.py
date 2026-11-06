"""Reusable activation buffers and out-of-band setup/checks for the CuTe gather."""


class StagingWorkload:
    def __init__(self, metadata, node, rank, dimension):
        import torch
        from metadata_cute import LocalStagingPlan

        self.rank = rank
        shape = (metadata.r.shape[0], dimension)
        self.source = torch.empty(shape, dtype=torch.bfloat16, device=metadata.r.device)
        self.destination = torch.empty_like(self.source)
        self.source_bits = self.source.view(torch.int16)
        # Bit patterns distinguish rows, columns, ranks and iterations. Integer
        # views allow exact copies of every BF16 bit pattern, including NaNs.
        self.rows = (torch.arange(shape[0], dtype=torch.int64, device=metadata.r.device) * 17
                     + rank * 131).to(torch.int16)
        self.columns = (torch.arange(dimension, dtype=torch.int64, device=metadata.r.device)
                        * 257).to(torch.int16)
        self.plan = LocalStagingPlan(metadata, self.source, self.destination, node)

    def prepare(self, iteration, verify):
        """Refresh activations outside timing on the caller's current stream."""
        import torch

        self.source_bits.copy_(self.rows[:, None])
        self.source_bits.add_(self.columns[None, :])
        salt = ((iteration * 257 + 32768) % 65536) - 32768
        self.source_bits.bitwise_xor_(salt)
        if verify:
            # Catch writes past the current valid prefix, including empty slices.
            self.destination.view(torch.uint8).fill_(0xA5)

    def verify(self, expected_rows):
        """Compare all bytes with CPU-reference x3, in bounded-size GPU chunks.

        This runs after native wait and outside timing. The indices come from
        the independent routing reference, not the GPU-produced x3 buffer.
        """
        import torch

        indices = torch.tensor(expected_rows, dtype=torch.int64, device=self.source.device)
        actual = self.destination.view(torch.int16)
        rows = len(expected_rows)
        for first in range(0, rows, 256):
            last = min(first + 256, rows)
            expected = self.source_bits.index_select(0, indices[first:last])
            if not torch.equal(actual[first:last], expected):
                raise AssertionError(f"rank={self.rank}: local staging payload mismatch at rows {first}:{last}")
        for first in range(rows, self.destination.shape[0], 256):
            tail = self.destination[first:first + 256].view(torch.uint8)
            if not torch.all(tail == 0xA5).item():
                raise AssertionError(f"rank={self.rank}: local staging modified unused tail at row {first}")
