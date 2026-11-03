"""Independent Python metadata oracle for unit tests; never used by the benchmark."""
import math

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

