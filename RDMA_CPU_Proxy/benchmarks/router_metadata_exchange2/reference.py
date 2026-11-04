"""Independent, host-only single-node routing reference (no Torch/CUDA imports)."""
MASK64 = (1 << 64) - 1


def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def sample_seed(base, source, iteration):
    return (base + source + (iteration - 1) * 8) & MASK64


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


def destination_slice(offsets, values, destination):
    first = destination * 16
    base, end = offsets[first], offsets[first + 16]
    return [x - base for x in offsets[first:first + 17]], values[base:end]
