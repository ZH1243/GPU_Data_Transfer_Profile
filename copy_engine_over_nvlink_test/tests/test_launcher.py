#!/usr/bin/env python3
"""Exercise shell supervision without CUDA, using a temporary fake proxy."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

launcher = Path(__file__).resolve().parents[1] / "run.sh"
with tempfile.TemporaryDirectory(prefix="nvlink-launcher-test-") as tmp:
    root = Path(tmp)
    fake = root / "proxy"
    fake.write_text('''#!/usr/bin/env python3
import os, pathlib, sys, time
args = sys.argv[1:]
rank = args[args.index('--rank') + 1]
shared = args[args.index('--shared-file') + 1]
pathlib.Path(os.environ['LOG_DIR'], rank).write_text(str(os.getpid()) + '\\n' + shared + '\\n' + repr(args))
if rank == '0':
    pathlib.Path(shared).touch()
mode = os.environ.get('MODE', 'success')
if mode == 'fail' and rank == '1':
    time.sleep(0.1)
    sys.exit(7)
if mode != 'success':
    time.sleep(30)
time.sleep(0.1)
''')
    fake.chmod(0o755)
    for mode in ("success", "fail", "signal"):
        logs = root / mode
        logs.mkdir()
        env = dict(os.environ, NVLINK_PROXY_BIN=str(fake), NVLINK_SHM_DIR=tmp,
                   LOG_DIR=str(logs), MODE=mode)
        proc = subprocess.Popen([str(launcher), "--devices", "3,1", "--sizes", "1MiB"],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if mode == "signal":
            deadline = time.monotonic() + 5
            while len(list(logs.iterdir())) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=5)
        expected = {"success": 0, "fail": 7, "signal": 143}[mode]
        assert proc.returncode == expected, (mode, proc.returncode, out, err)
        assert {p.name for p in logs.iterdir()} == {"0", "1"}
        paths = set()
        for log in logs.iterdir():
            pid, shared, args = log.read_text().splitlines()
            paths.add(shared)
            assert "'3,1'" in args and "'1MiB'" in args
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(f"proxy {pid} left running")
        assert len(paths) == 1
        assert not list(root.glob("nvlink-a2a.*")), "shared directory left behind"
    # An empty proxy-option list must also work.
    env["MODE"] = "success"
    subprocess.run([str(launcher), "--devices", "0,1"], env=env, check=True, timeout=5)
print("PASS: rank launch, argument forwarding, failure/signal cleanup, empty options")
