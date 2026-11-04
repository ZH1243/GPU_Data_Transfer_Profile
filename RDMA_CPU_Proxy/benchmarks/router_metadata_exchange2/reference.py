"""Independent, host-only single-node routing reference (no Torch/CUDA imports)."""
MASK64 = (1 << 64) - 1


def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def sample_seed(base, source, iteration, nodes=1):
    return (base + source + (iteration - 1) * 8 * nodes) & MASK64


def sample_pattern(pattern, iteration):
    return ("affine", "hot-low", "hot-high")[(iteration - 1) % 3] if pattern == "cycle" else pattern


def make_routes(tokens, seed, pattern="affine"):
    if pattern in ("hot-low", "hot-high"):
        first = 0 if pattern == "hot-low" else 120
        return [list(range(first, first + 8)) for _ in range(tokens)]
    if pattern != "affine":
        raise ValueError("unknown route pattern")
    routes = []
    for token in range(tokens):
        first = splitmix64(seed ^ token)
        stride = (1 + splitmix64(first) % 127) | 1
        routes.append([(first % 128 + k * stride) % 128 for k in range(8)])
    return routes


def metadata(routes, source):
    if not 0 <= source < 8:
        raise ValueError("source GPU must be 0..7")
    masks = []
    for row in routes:
        if len(row) != 8 or len(set(row)) != 8 or any(e < 0 or e >= 128 for e in row):
            raise ValueError("routes require eight distinct experts in [0,128)")
        mask = 0
        for e in row:
            mask |= 1 << ((source - e // 16) % 8)
        masks.append(mask)
    order = sorted(range(len(routes)), key=lambda token: -masks[token])
    positions = [0] * 8
    lists = [[] for _ in range(128)]
    for position, token in enumerate(order):
        for e in routes[token]:
            dest = e // 16
            lists[e].append(position if dest == source else positions[dest])
        for dest in range(8):
            if masks[token] & (1 << ((source - dest) % 8)):
                positions[dest] += 1
    offsets, values = [0], []
    for indices in lists:
        values.extend(indices)
        offsets.append(len(values))
    return offsets, values, order, [masks[t] for t in order]


def destination_slice(offsets, values, destination, nodes=1):
    epg = 128 // (nodes * 8)
    first = destination * epg
    base, end = offsets[first], offsets[first + epg]
    return [x - base for x in offsets[first:first + epg + 1]], values[base:end]


def metadata_two(routes, source):
    """Two destination-node orders; source is the original local GPU index.

    Return only meaningful x3/x4 entries (GPU allocations have a 2*T tail).
    """
    if not 0 <= source < 8:
        raise ValueError("source GPU must be 0..7")
    entries = [[], []]
    for token, row in enumerate(routes):
        if len(row) != 8 or len(set(row)) != 8 or any(e < 0 or e >= 128 for e in row):
            raise ValueError("routes require eight distinct experts in [0,128)")
        masks = [0, 0]
        for e in row:
            masks[e // 64] |= 1 << ((source - (e // 8) % 8) % 8)
        for node, mask in enumerate(masks):
            if mask:
                entries[node].append((token, mask))
    lists = [[] for _ in range(128)]
    x3, x4, node_offsets = [], [], [0]
    for node in range(2):
        positions = [0] * 8
        for position, (token, mask) in enumerate(sorted(entries[node], key=lambda x: -x[1])):
            x3.append(token)
            x4.append(mask)
            for e in routes[token]:
                if e // 64 == node:
                    gpu = (e // 8) % 8
                    lists[e].append(position if gpu == source else positions[gpu])
            for gpu in range(8):
                if mask & (1 << ((source - gpu) % 8)):
                    positions[gpu] += 1
        node_offsets.append(len(x3))
    offsets, values = [0], []
    for indices in lists:
        values.extend(indices)
        offsets.append(len(values))
    return offsets, values, x3, x4, node_offsets
