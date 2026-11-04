import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from reference import make_routes, metadata, metadata_two, destination_slice, sample_seed, sample_pattern


class ReferenceTests(unittest.TestCase):
    def test_two_node_empty_slices_and_seed(self):
        self.assertEqual(sample_seed((1 << 64) - 2, 15, 2, 2), 29)
        for source in range(8):
            for pattern, target in (("hot-low", 0), ("hot-high", 15)):
                offsets, values, x3, x4, nodes = metadata_two(make_routes(33, 0, pattern), source)
                self.assertEqual(nodes, [0, 33, 33] if target == 0 else [0, 0, 33])
                self.assertEqual(x3, list(range(33)))
                self.assertEqual(x4, [1 << ((source - target % 8) % 8)] * 33)
                for dest in range(16):
                    off, val = destination_slice(offsets, values, dest, 2)
                    self.assertEqual(len(off), 9)
                    self.assertEqual(len(val), 264 if dest == target else 0)

    def test_remote_same_index_uses_direct_positions(self):
        # Both destination nodes retain gaps for source-local GPU 0; GPU 1
        # uses compact positions. Relaying through GPU 1 must not reindex.
        routes = [list(range(e, e + 8)) for e in (0, 8, 64, 72, 0, 64)]
        offsets, values, _, _, nodes = metadata_two(routes, 0)
        self.assertEqual(nodes, [0, 3, 6])
        for expert in (0, 64):
            self.assertEqual(values[offsets[expert]:offsets[expert + 1]], [1, 2])
        for expert in (8, 72):
            self.assertEqual(values[offsets[expert]:offsets[expert + 1]], [0])

    @unittest.skipUnless(os.environ.get("RME2_REFERENCE_BIN"), "run through CTest for production C++ comparison")
    def test_two_node_matches_production(self):
        for tokens in (1, 33, 257):
            for source in range(16):
                seed = sample_seed((1 << 64) - 2, source, 2, 2)
                expected = json.loads(subprocess.check_output(
                    [os.environ["RME2_REFERENCE_BIN"], "--dump", str(tokens), str(seed), str(source), "2"], text=True))
                self.assertEqual(list(metadata_two(make_routes(tokens, seed), source % 8)), expected)

    def test_generator_matches_existing_generator(self):
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "get_expert_token_idx_list_gpu"))
        from benchmark_metadata_cute import make_routes as original
        for seed in (0, 1234, (1 << 64) - 1):
            self.assertEqual(make_routes(257, seed), original(257, seed))

    def test_seed_wrap_and_patterns(self):
        self.assertEqual(sample_seed((1 << 64) - 2, 3, 2), 9)
        self.assertEqual([sample_pattern("cycle", i) for i in range(1, 5)],
                         ["affine", "hot-low", "hot-high", "affine"])

    def test_hot_destinations_and_empty_lists(self):
        for source in range(8):
            for pattern, target in (("hot-low", 0), ("hot-high", 7)):
                offsets, values, x3, x4 = metadata(make_routes(33, 0, pattern), source)
                self.assertEqual(x3, list(range(33)))
                self.assertEqual(x4, [1 << ((source - target) % 8)] * 33)
                for dest in range(8):
                    off, val = destination_slice(offsets, values, dest)
                    self.assertEqual(off[-1], 264 if dest == target else 0)
                    self.assertEqual(len(val), off[-1])

    def test_direct_vs_compacted_positions(self):
        routes = [list(range(8)), list(range(16, 24)), list(range(8)), list(range(16, 24))]
        offsets, values, x3, _ = metadata(routes, 0)
        # GPU 1's mask sorts before GPU 0's: direct self indices retain gaps.
        self.assertEqual(x3, [1, 3, 0, 2])
        self.assertEqual(values[offsets[0]:offsets[1]], [2, 3])
        self.assertEqual(values[offsets[16]:offsets[17]], [0, 1])

    @unittest.skipUnless(os.environ.get("RME2_REFERENCE_BIN"), "run through CTest for production C++ comparison")
    def test_matches_production_cpp_routing(self):
        binary = os.environ["RME2_REFERENCE_BIN"]
        for tokens in (1, 33, 257):
            for source in range(8):
                seed = (1 << 64) - 1 - source
                expected = json.loads(subprocess.check_output(
                    [binary, "--dump", str(tokens), str(seed), str(source)], text=True))
                self.assertEqual(list(metadata(make_routes(tokens, seed), source)), expected)


if __name__ == "__main__":
    unittest.main()
