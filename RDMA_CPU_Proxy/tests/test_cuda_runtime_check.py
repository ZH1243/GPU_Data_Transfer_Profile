"""CPU regressions for the mixed-runtime failure in embedded CUDA workers."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


WORKER = Path(__file__).resolve().parents[1] / "scripts" / "rdma_proxy_worker.py"
SPEC = importlib.util.spec_from_file_location("runtime_check_worker", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def maps(*paths):
    return "\n".join(f"1000-2000 r-xp 00000000 00:01 123 {path}" for path in paths)


class CudaRuntimeTests(unittest.TestCase):
    def test_matching_runtime_and_repeated_segments(self):
        for version, path in (("12.9", "/env/nvidia/lib/libcudart.so.12"),
                              ("13.0", "/usr/local/cuda/lib/libcudart.so.13.0.96")):
            with self.subTest(version=version):
                worker.check_cuda_runtime_compatibility(
                    version, "/proxy.so", mappings=maps(path, path, "/usr/lib/libcuda.so.1"))

    def test_reported_cuda12_cuda13_conflict(self):
        cuda12 = "/env/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
        cuda13 = "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so.13.0.96"
        with self.assertRaises(RuntimeError) as failure:
            worker.check_cuda_runtime_compatibility(
                "12.9", "/build-blackwell/librdma_cpu_proxy.so", mappings=maps(cuda12, cuda13))
        message = str(failure.exception)
        for detail in (cuda12, cuda13, "before proxy initialization", "CUDA 12.9",
                       "CMAKE_CUDA_COMPILER", "CUDAToolkit_ROOT", "RDMA_PROXY_BUILD_DIR"):
            self.assertIn(detail, message)

    def test_single_wrong_runtime_and_duplicate_same_major(self):
        for paths in (("/cuda/libcudart.so.13.0.96",),
                      ("/torch/libcudart.so.12", "/other/libcudart.so.12"),
                      ("/torch/libcudart.so.12 (deleted)", "/torch/libcudart.so.12"),
                      ("/cuda/libcudart.so",), ()):
            with self.subTest(paths=paths), self.assertRaises(RuntimeError):
                worker.check_cuda_runtime_compatibility("12.9", "/proxy.so", mappings=maps(*paths))

    def test_loader_checks_before_any_native_api_call(self):
        library = Mock()
        with tempfile.NamedTemporaryFile() as file, \
             patch.object(worker.ctypes, "CDLL", return_value=library), \
             patch.object(worker, "check_cuda_runtime_compatibility",
                          side_effect=RuntimeError("mixed runtimes")) as check:
            with self.assertRaisesRegex(RuntimeError, "mixed runtimes"):
                worker.load_proxy_library(Path(file.name), torch_cuda_version="12.9")
            check.assert_called_once_with("12.9", Path(file.name))
            self.assertEqual(library.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
