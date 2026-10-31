"""Host-only rank-contract tests; GPU validation is benchmark --check-suite.

These tests need neither PyTorch nor CUDA and do not claim to compile/test DSL.
"""

import random
import unittest

from benchmark_metadata_cute import make_routes


def bitmap_positions(groups):
    """CPU model of per-warp bitmap ranks, independently checked against lists."""
    words = {}
    for position, group in enumerate(groups):
        bits = words.setdefault(group, [0] * 8)
        bits[position // 32] |= 1 << (position % 32)
    ranks = []
    for position, group in enumerate(groups):
        bits = words[group]
        warp, lane = divmod(position, 32)
        ranks.append(sum(bin(word).count("1") for word in bits[:warp])
                     + bin(bits[warp] & ((1 << lane) - 1)).count("1"))
    return ranks


class MetadataContracts(unittest.TestCase):
    def test_bitmap_stability_at_warp_and_chunk_boundaries(self):
        rng = random.Random(1234)
        for length in (1, 31, 32, 33, 255, 256):
            for groups in ([7] * length, list(range(length)),
                           [rng.randrange(13) for _ in range(length)]):
                expected = [groups[:i].count(group) for i, group in enumerate(groups)]
                self.assertEqual(bitmap_positions(groups), expected)

    def test_generator_contract(self):
        for pattern in ("affine", "node0", "node1", "spread", "random"):
            for row in make_routes(257, pattern=pattern):
                self.assertEqual(len(row), 8)
                self.assertEqual(len(set(row)), 8)
                self.assertTrue(all(0 <= expert < 128 for expert in row))
        self.assertEqual(make_routes(20), make_routes(20))
        self.assertNotEqual(make_routes(20, seed=1234), make_routes(20, seed=4321))

    def test_chunk_prefix_and_bitmap_rank_preserve_global_token_order(self):
        groups = [random.Random(i).randrange(17) for i in range(1025)]
        totals = [groups.count(group) for group in range(17)]
        offsets = [sum(totals[group + 1:]) for group in range(17)]
        prefix = [0] * 17
        result = [None] * len(groups)
        for start in range(0, len(groups), 256):
            tile = groups[start:start + 256]
            for i, rank in enumerate(bitmap_positions(tile)):
                group = tile[i]
                result[offsets[group] + prefix[group] + rank] = start + i
            for group in range(17):
                prefix[group] += tile.count(group)
        expected = sorted(range(len(groups)), key=lambda i: -groups[i])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
