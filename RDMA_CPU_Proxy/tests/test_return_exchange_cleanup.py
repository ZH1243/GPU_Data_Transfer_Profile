"""CPU checks for successful CUDA IPC ownership teardown."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref


PATH = Path(__file__).resolve().parents[1] / "scripts" / "rdma_return_exchange.py"
SPEC = importlib.util.spec_from_file_location("rdma_return_exchange_cleanup", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Storage:
    pass


class CleanupTests(unittest.TestCase):
    def test_success_releases_storage_after_imports_closed(self):
        exchange = MODULE.ReturnExchange.__new__(MODULE.ReturnExchange)
        events = []
        exchange.closed = False
        exchange.control = SimpleNamespace(
            gather=lambda phase: events.append(phase),
            close=lambda: events.append("control_closed"))
        exchange.cuda = SimpleNamespace(close=lambda: events.append("imports_closed_local"))
        storage = Storage()
        reference = weakref.ref(storage)
        export = SimpleNamespace(storage=storage, release=lambda: events.append("counter_released"))
        exchange.exports = [export]
        exchange.outputs = (storage,)
        exchange.received = {0: storage}
        exchange.received_valid_rows = {0: 1}
        down = SimpleNamespace(return_exchange=exchange)
        exchange.down = down
        del storage, export

        exchange.close(True)
        self.assertEqual(events, ["workers_stopped", "imports_closed_local",
                                  "imports_closed", "counter_released", "control_closed"])
        self.assertIsNone(reference())
        self.assertIsNone(down.return_exchange)
        self.assertIsNone(exchange.down)
        exchange.close(True)
        self.assertEqual(len(events), 5)

    def test_failure_retains_resources(self):
        exchange = MODULE.ReturnExchange.__new__(MODULE.ReturnExchange)
        exchange.closed = False
        exchange.control = Mock()
        exchange.cuda = Mock()
        exchange.exports = [Mock()]
        try:
            exchange.close(False)
            self.assertIn(exchange, MODULE._failed_exchanges)
            exchange.cuda.close.assert_not_called()
            exchange.exports[0].release.assert_not_called()
        finally:
            MODULE._failed_exchanges.remove(exchange)


if __name__ == "__main__":
    unittest.main()
