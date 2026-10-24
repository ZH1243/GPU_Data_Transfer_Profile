"""CPU map tests and opt-in Hopper post-RDMA reduction tests."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inter_node_combine import inverse_token_rows


class MappingTests(unittest.TestCase):
    def test_independent_unsorted_domains(self):
        domains = [[5, 0, 2], [2, 4], [], [0, 5, 3]]
        inverse = inverse_token_rows(domains, 7)
        self.assertEqual(inverse[0], [1, -1, 2, -1, -1, 0, -1])
        # Independent scatter reference versus gather in original token order.
        expected = [0] * 7
        packed = [[100 * (node + 1) + token for token in ids]
                  for node, ids in enumerate(domains)]
        for ids, values in zip(domains, packed):
            for token, value in zip(ids, values):
                expected[token] += value
        actual = [sum(values[rows[token]] for rows, values in zip(inverse, packed)
                      if rows[token] >= 0) for token in range(7)]
        self.assertEqual(actual, expected)
        self.assertEqual(actual[6], 0)
        self.assertEqual(inverse_token_rows([[], []], 0), [[], []])

    def test_reject_invalid_routes(self):
        for ids in ([1, 1], [-1], [3]):
            with self.assertRaises(ValueError):
                inverse_token_rows([ids], 3)
        with self.assertRaises(ValueError):
            inverse_token_rows([], -1)


@unittest.skipUnless(os.environ.get('RDMA_PROXY_TEST_CUDA') == '1', 'requires Hopper opt-in')
class KernelTests(unittest.TestCase):
    def test_token_order_empty_domains_and_iteration_reuse(self):
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            self.skipTest('requires SM90')
        sys.path.insert(0, os.environ.get('QUACK_ROOT', str(
            Path(__file__).resolve().parents[3] / 'quack_for_communication_overlap')))
        from inter_node_combine import TokenOrderCombine
        for dtype in (torch.bfloat16, torch.float16):
            for domains, tokens in (([[5, 0, 2], [2, 4]], 7),
                                    ([[5, 0, 2], [], [4, 2], [0, 5, 3]], 7),
                                    ([[], []], 7), ([[], []], 0)):
                for tile in (8, 128):
                    with self.subTest(dtype=dtype, domains=domains, tokens=tokens, tile=tile):
                        width = tile * 3
                        inputs = [torch.empty((3, len(ids), tile), dtype=dtype, device='cuda')
                                  for ids in domains]
                        combine = TokenOrderCombine(inputs, domains, tokens, check=True)
                        for iteration in range(2):
                            expected = torch.zeros((tokens, width), dtype=torch.float32, device='cuda')
                            for tensor, ids in zip(inputs, domains):
                                values = torch.randn((len(ids), width), dtype=dtype, device='cuda') + iteration
                                tensor.copy_(values.view(len(ids), 3, tile).permute(1, 0, 2))
                                for row, token in enumerate(ids):
                                    expected[token] += values[row].float()
                            combine.output.fill_(float('nan'))
                            torch.cuda.synchronize()
                            combine.run()
                            torch.testing.assert_close(combine.output, expected.to(dtype), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
