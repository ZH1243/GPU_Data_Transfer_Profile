#!/usr/bin/env python3
"""CPU-only router metadata benchmark. torchrun supplies environment only."""
import argparse
import copy
import json
import math
import os
import selectors
import socket
import statistics
import struct
import time
from pathlib import Path


U64 = struct.Struct("!Q")
FRAME = struct.Struct("!QQ")  # payload bytes, iteration (benchmark envelope)
MASK64 = (1 << 64) - 1
STAGE_FIELDS = ("serialize_ms", "self_install_ms", "communication_prepare_ms",
                "tcp_exchange_ms", "receive_process_ms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def prepare(a, source, dataset):
    """CPU equivalent of node-mask routing, including compact buffer indices."""
    local = source % a.gpus_per_node
    epg = a.num_experts // a.world
    epn = epg * a.gpus_per_node
    seed = (a.seed + source + dataset * a.world) & MASK64
    routes, buckets = [], [[] for _ in a.hosts]
    for token in range(a.num_tokens):
        r0 = splitmix64(seed ^ token)
        r1 = splitmix64(r0)
        stride = 1 if a.num_experts == 1 else 1 + r1 % (a.num_experts - 1)
        while math.gcd(stride, a.num_experts) != 1:
            stride = stride + 1 if stride + 1 < a.num_experts else 1
        experts = [(r0 % a.num_experts + k * stride) % a.num_experts
                   for k in range(a.top_k)]
        routes.append(experts)
        masks = {}
        for expert in experts:
            node, gpu = expert // epn, (expert // epg) % a.gpus_per_node
            masks[node] = masks.get(node, 0) | (1 << ((local - gpu) % a.gpus_per_node))
        for node, mask in masks.items():
            buckets[node].append((mask, token))

    lists = [[] for _ in range(a.num_experts)]
    node_data = {}
    for node, entries in enumerate(buckets):
        entries.sort(key=lambda entry: -entry[0])  # stable descending mask
        positions = [0] * a.gpus_per_node
        for pos, (mask, token) in enumerate(entries):
            for expert in routes[token]:
                if expert // epn == node:
                    gpu = (expert // epg) % a.gpus_per_node
                    lists[expert].append(pos if gpu == local else positions[gpu])
            for gpu in range(a.gpus_per_node):
                if mask & (1 << ((local - gpu) % a.gpus_per_node)):
                    positions[gpu] += 1
        node_data[node] = ([t for _, t in entries], [m for m, _ in entries])

    metadata = {}
    for dest in range(a.world):
        indices, offsets = [], [0]
        for expert in range(dest * epg, (dest + 1) * epg):
            indices.extend(lists[expert])
            offsets.append(len(indices))
        metadata[(dest, "expert")] = (
            "router_expert_metadata_v1",
            [source // a.gpus_per_node, local, dest // a.gpus_per_node,
             dest % a.gpus_per_node, len(a.hosts), a.gpus_per_node,
             a.num_experts, epg, dest * epg, a.num_tokens, len(indices)],
            offsets, indices)
    for node, (indices, masks) in node_data.items():
        if node != source // a.gpus_per_node:
            dest = node * a.gpus_per_node + local
            metadata[(dest, "x3x4")] = (
                "router_x3_x4_v2",
                [source // a.gpus_per_node, node, local, len(a.hosts),
                 a.gpus_per_node, a.num_experts, a.top_k, a.num_tokens,
                 a.token_dimension, a.element_bytes, a.tokens_per_chunk, len(indices)],
                indices, masks)
    return metadata


def encode(metadata):
    version, header, first, second = metadata
    return (version + " " + " ".join(map(str, header + first + second))).encode("ascii")


def install(a, source, kind, payload, installed):
    key = (source, kind)
    require(key not in installed, "duplicate metadata")
    words = payload.split()
    version = words[0]
    values = list(map(int, words[1:]))
    epg = a.num_experts // a.world
    if kind == "expert":
        require(version == b"router_expert_metadata_v1", "expert version mismatch")
        expected = [source // a.gpus_per_node, source % a.gpus_per_node,
                    a.rank // a.gpus_per_node, a.rank % a.gpus_per_node,
                    len(a.hosts), a.gpus_per_node, a.num_experts, epg,
                    a.rank * epg, a.num_tokens]
        require(values[:10] == expected and len(values) >= 11, "expert header mismatch")
        count = values[10]
        offsets = values[11:12 + epg]
        indices = values[12 + epg:]
        require(0 <= count <= a.num_tokens * a.top_k and len(indices) == count,
                "expert index count mismatch")
        require(len(offsets) == epg + 1 and offsets[0] == 0 and offsets[-1] == count
                and all(0 <= x <= y <= count for x, y in zip(offsets, offsets[1:])),
                "invalid expert offsets")
        require(all(0 <= x < a.num_tokens for x in indices), "expert index out of range")
        for start, end in zip(offsets, offsets[1:]):
            require(all(indices[i] <= indices[i + 1] for i in range(start, end - 1)),
                    "expert indices not monotonic")
        installed[key] = {"offsets": offsets, "indices": indices,
                          "heads": [0] * epg, "tails": [0] * epg}
    else:
        require(version == b"router_x3_x4_v2", "x3/x4 version mismatch")
        expected = [source // a.gpus_per_node, a.rank // a.gpus_per_node,
                    a.rank % a.gpus_per_node, len(a.hosts), a.gpus_per_node,
                    a.num_experts, a.top_k, a.num_tokens, a.token_dimension,
                    a.element_bytes, a.tokens_per_chunk]
        require(values[:11] == expected and len(values) >= 12, "x3/x4 header mismatch")
        count = values[11]
        require(0 <= count <= a.num_tokens and len(values) == 12 + 2 * count,
                "x3/x4 count mismatch")
        indices, masks = values[12:12 + count], values[12 + count:]
        require(all(0 <= x < a.num_tokens for x in indices), "x3 out of range")
        require(all(0 < m < (1 << a.gpus_per_node) for m in masks), "x4 out of range")
        chunks = []
        token_bytes = a.token_dimension * a.element_bytes
        for chunk, start in enumerate(range(0, count, a.tokens_per_chunk)):
            rows = indices[start:start + a.tokens_per_chunk]
            chunks.append(dict(chunk_index=chunk, start_token=start, num_tokens=len(rows),
                               src_offset_bytes=0, dst_offset_bytes=start * token_bytes,
                               length_bytes=len(rows) * token_bytes, qp_index=-1,
                               imm_data=chunk, source_token_indices=rows))
        installed[key] = {"indices": indices, "masks": masks, "chunks": chunks}


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("peer closed connection")
        data.extend(part)
    return bytes(data)


def send_json(sock, value):
    data = json.dumps(value).encode()
    sock.sendall(U64.pack(len(data)) + data)


def recv_json(sock):
    size = U64.unpack(recv_exact(sock, U64.size))[0]
    require(size <= 64 * 1024 * 1024, "control message too large")
    return json.loads(recv_exact(sock, size))


def kinds(a, left, right):
    result = ["expert"]
    if left % a.gpus_per_node == right % a.gpus_per_node:
        result.append("x3x4")
    if left == 0:
        result.append("control")
    return result


def connect_mesh(a):
    """Lower ranks accept higher ranks; every pair owns full-duplex channels."""
    channels, controls = {}, {}
    def port_for(rank):
        return a.port_base + (rank // a.gpus_per_node) * a.node_port_stride + rank % a.gpus_per_node

    listener = socket.create_server((a.bind_host, port_for(a.rank)),
                                    backlog=max(128, 3 * a.world))
    listener.settimeout(a.timeout)
    try:
        for peer in range(a.rank):
            host = "127.0.0.1" if peer // a.gpus_per_node == a.rank // a.gpus_per_node else a.hosts[peer // a.gpus_per_node]
            for kind in kinds(a, peer, a.rank):
                deadline = time.monotonic() + a.timeout
                while True:
                    try:
                        sock = socket.create_connection((host, port_for(peer)),
                                                        timeout=min(1, a.timeout))
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.02)
                sock.settimeout(a.timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                send_json(sock, [a.rank, kind])
                (controls if kind == "control" else channels)[peer if kind == "control" else (peer, kind)] = sock
        expected = {(peer, kind) for peer in range(a.rank + 1, a.world)
                    for kind in kinds(a, a.rank, peer)}
        while expected:
            sock, _ = listener.accept()
            sock.settimeout(a.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            peer, kind = recv_json(sock)
            require((peer, kind) in expected, "unexpected/duplicate connection")
            expected.remove((peer, kind))
            (controls if kind == "control" else channels)[peer if kind == "control" else (peer, kind)] = sock
    finally:
        listener.close()
    for sock in channels.values():
        sock.setblocking(False)
    return channels, controls


def collective(a, controls, tag, value):
    """Control-only gather and release, always outside the measured interval."""
    if a.rank:
        send_json(controls[0], [tag, value])
        result_tag, result = recv_json(controls[0])
        require(result_tag == tag, "control phase mismatch")
        return result
    values = [value]
    for peer in range(1, a.world):
        received_tag, received = recv_json(controls[peer])
        require(received_tag == tag, "control phase mismatch")
        values.append(received)
    for peer in range(1, a.world):
        send_json(controls[peer], [tag, values])
    return values


def exchange(a, channels, metadata, cached, iteration):
    started = time.perf_counter()
    payloads = cached if cached is not None else {key: encode(metadata[key]) for key in channels}
    serialized = time.perf_counter()
    # Self installation is a local copy, without wire serialization/parsing.
    own = copy.deepcopy(metadata[(a.rank, "expert")])
    epg = a.num_experts // a.world
    installed = {(a.rank, "expert"): {"offsets": own[2], "indices": own[3],
                                      "heads": [0] * epg, "tails": [0] * epg}}
    self_installed = time.perf_counter()
    states = {}
    receive_seconds = 0.0
    max_payload = 4096 + (a.num_tokens * a.top_k + epg + 1) * 32
    with selectors.DefaultSelector() as selector:
        for key, sock in channels.items():
            payload = payloads[key]
            state = dict(key=key, output=memoryview(FRAME.pack(len(payload), iteration) + payload),
                         sent=0, header=bytearray(), body=bytearray(), length=None, done=False)
            states[key] = state
            selector.register(sock, selectors.EVENT_READ | selectors.EVENT_WRITE, state)
        deadline = time.monotonic() + a.timeout
        prepared = time.perf_counter()
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("metadata exchange timed out")
            for entry, events in selector.select(remaining):
                sock, state = entry.fileobj, entry.data
                if events & selectors.EVENT_WRITE:
                    try:
                        n = sock.send(state["output"][state["sent"]:])
                        if n == 0:
                            raise ConnectionError("zero-byte send")
                        state["sent"] += n
                    except BlockingIOError:
                        pass
                if events & selectors.EVENT_READ:
                    target = state["header"] if state["length"] is None else state["body"]
                    needed = FRAME.size if state["length"] is None else state["length"]
                    try:
                        part = sock.recv(min(262144, needed - len(target)))
                    except BlockingIOError:
                        part = None
                    if part == b"":
                        raise ConnectionError("peer closed during exchange")
                    if part:
                        target.extend(part)
                    if state["length"] is None and len(state["header"]) == FRAME.size:
                        size, received_iteration = FRAME.unpack(state["header"])
                        require(received_iteration == iteration, "stale/mixed iteration")
                        require(0 < size <= max_payload, "invalid payload length")
                        state["length"] = size
                    if state["length"] is not None and len(state["body"]) == state["length"]:
                        before = time.perf_counter()
                        install(a, *state["key"], state["body"], installed)
                        receive_seconds += time.perf_counter() - before
                        state["done"] = True
                mask = (0 if state["done"] else selectors.EVENT_READ)
                if state["sent"] < len(state["output"]):
                    mask |= selectors.EVENT_WRITE
                if mask:
                    selector.modify(sock, mask, state)
                else:
                    selector.unregister(sock)
    require(len(installed) == 1 + len(channels), "missing installed metadata")
    finished = time.perf_counter()
    elapsed = finished - started
    self_seconds = self_installed - serialized
    # Non-overlapping local wall-time accounting. Socket progress can overlap
    # remote CPU work; this residual is not pure network latency.
    tcp_seconds = finished - prepared - receive_seconds
    return dict(rank=a.rank, elapsed_ms=elapsed * 1000,
                serialize_ms=(serialized - started) * 1000,
                self_install_ms=self_seconds * 1000,
                communication_prepare_ms=(prepared - self_installed) * 1000,
                tcp_exchange_ms=tcp_seconds * 1000,
                receive_process_ms=receive_seconds * 1000,
                install_ms=(self_seconds + receive_seconds) * 1000,
                installed_expert_sources=sum(kind == "expert" for _, kind in installed),
                installed_x3x4_sources=sum(kind == "x3x4" for _, kind in installed),
                sent_bytes=sum(len(p) + FRAME.size for p in payloads.values()),
                received_bytes=sum(s["length"] + FRAME.size for s in states.values())), installed


def summarize(values):
    ordered = sorted(values)
    return dict(min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), p95=ordered[math.ceil(len(values) * .95) - 1],
                p99=ordered[math.ceil(len(values) * .99) - 1], max=max(values))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", required=True, help="comma-separated TCP node addresses, in node-rank order")
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("LOCAL_WORLD_SIZE", "8")))
    p.add_argument("--port-base", type=int, default=20515)
    p.add_argument("--node-port-stride", type=int, default=0,
                   help="port offset per node; use GPUs-per-node to emulate nodes on one host")
    p.add_argument("--bind-host", default="0.0.0.0")
    p.add_argument("--num-tokens", type=int, default=20480)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--token-dimension", type=int, default=4096)
    p.add_argument("--element-bytes", type=int, default=2)
    p.add_argument("--tokens-per-chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--datasets", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--preserialized", action="store_true")
    p.add_argument("--output", type=Path, help="rank-0 JSON report path")
    a = p.parse_args()
    a.hosts = a.hosts.split(",")
    a.rank = int(os.environ["RANK"])
    a.world = int(os.environ["WORLD_SIZE"])
    require(all(a.hosts), "empty node host")
    require(1 <= a.gpus_per_node <= 8 and a.world == len(a.hosts) * a.gpus_per_node,
            "world size must equal nodes * GPUs per node (1..8)")
    require(0 <= a.rank < a.world and int(os.environ["LOCAL_RANK"]) == a.rank % a.gpus_per_node,
            "invalid rank mapping")
    for name in ("num_tokens", "num_experts", "top_k", "token_dimension", "element_bytes",
                 "tokens_per_chunk", "datasets", "iterations", "timeout"):
        require(getattr(a, name) > 0, name + " must be positive")
    require(a.warmup >= 0 and a.top_k <= a.num_experts and a.num_experts % a.world == 0,
            "invalid warmup/top-k/expert partition")
    require(a.node_port_stride >= 0 and 0 < a.port_base and
            a.port_base + (len(a.hosts) - 1) * a.node_port_stride + a.gpus_per_node <= 65536,
            "invalid port range")
    return a


def main():
    a = arguments()
    channels, controls = connect_mesh(a)
    try:
        signature = {k: v for k, v in vars(a).items() if k not in ("rank", "output", "bind_host")}
        signatures = collective(a, controls, "configuration", signature)
        require(all(s == signature for s in signatures), "ranks have different benchmark configurations")
        if a.rank == 0:
            print(f"Connected {a.world} CPU proxies; preparing {a.datasets} routing datasets per rank", flush=True)
        datasets = [prepare(a, a.rank, d) for d in range(a.datasets)]
        cached = [{key: encode(data[key]) for key in channels} for data in datasets] if a.preserialized else None
        records = []
        for iteration in range(a.warmup + a.iterations):
            d = iteration % a.datasets
            collective(a, controls, f"start:{iteration}", None)
            result, installed = exchange(a, channels, datasets[d], cached[d] if cached else None, iteration)
            results = collective(a, controls, f"complete:{iteration}", result)
            del installed
            if iteration >= a.warmup:
                records.append(results)
        if a.rank == 0:
            critical_records = [max(rows, key=lambda r: r["elapsed_ms"]) for rows in records]
            maxima = [r["elapsed_ms"] for r in critical_records]
            report = dict(configuration=signature, critical_rank_ms=summarize(maxima),
                          critical_rank_stage_ms={field: summarize([r[field] for r in critical_records])
                                                  for field in STAGE_FIELDS},
                          rank_summaries=[dict(rank=rank, **{field: summarize([row[rank][field] for row in records])
                                          for field in ("elapsed_ms", "install_ms") + STAGE_FIELDS})
                                          for rank in range(a.world)], iterations=records)
            print(json.dumps({"critical_rank_ms": report["critical_rank_ms"],
                              "critical_rank_stage_ms": report["critical_rank_stage_ms"],
                              "mode": "preserialized" if a.preserialized else "serialize-in-timer"}, indent=2), flush=True)
            if a.output:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"Saved {a.output}", flush=True)
    finally:
        for sock in list(channels.values()) + list(controls.values()):
            sock.close()


if __name__ == "__main__":
    main()
