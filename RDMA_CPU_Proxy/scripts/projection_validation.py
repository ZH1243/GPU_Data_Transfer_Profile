"""Untimed numerical checks for the actual embedded up/down GEMM composition.

Reference matrices are bounded by ``column_chunk``; weights are converted to
FP32 one column block at a time. Call only after the iteration's GPU work and
return/combine consumers have completed, before reusing any input buffers.
"""

from collections import Counter
import ctypes
import math
import os
import tempfile


def route_positions(up, routes):
    """Return exact (expert, source, A_idx position), ignoring repeated N groups."""
    indices = [index.cpu().tolist() for index in up.a_idx]
    result = [None] * routes
    for descriptor in up.work_table.cpu().tolist():
        expert, group, *ranges = descriptor
        if group != 0:
            continue
        offset = sum(ranges[::2])
        for source, index in enumerate(indices):
            begin, end = ranges[2 * source:2 * source + 2]
            if not 0 <= begin <= end <= len(index):
                raise AssertionError("gather descriptor references an invalid index range")
            for position in range(begin, end):
                if not 0 <= offset < routes or result[offset] is not None:
                    raise AssertionError("gather descriptors overlap or exceed the routed output")
                token = index[position]
                if not 0 <= token < up.x_buffers[source].shape[0]:
                    raise AssertionError("gather descriptor references a padded/invalid input row")
                result[offset] = (expert, source, position)
                offset += 1
    if any(row is None for row in result):
        raise AssertionError("gather descriptors leave routed output rows uncovered")
    return result


def route_metadata(up, routes):
    """Decode N-group zero independently of the GPU combine mapping helper."""
    indices = [index.cpu().tolist() for index in up.a_idx]
    return [(expert, source, indices[source][position])
            for expert, source, position in route_positions(up, routes)]


def route_weight_reference(up, down, routes):
    """Decode immutable source/list weights, never trusting packed GPU mapping."""
    source_weights = getattr(up, "route_weights", None)
    packed = getattr(down, "route_weights", None) if down is not None else None
    if source_weights is None:
        if packed is not None:
            raise AssertionError("packed routing weights have no immutable source reference")
        return [1.0] * routes
    weights = [weight.cpu().tolist() for weight in source_weights]
    result = [weights[source][position]
              for _, source, position in route_positions(up, routes)]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise AssertionError("routing weights must be finite and nonnegative")
    if down is not None:
        if packed is None or packed.cpu().tolist() != result:
            raise AssertionError("packed routing weights disagree with source/list positions")
    return result


def original_source_routes(experts, weights, *, source_gpu, destination_node,
                           destination_gpu, num_nodes, gpus_per_node, num_experts):
    """Independently reconstruct x3/x4 from original [token, top-k] assignments.

    Return (local expert, destination input row, FP32 weight, original token).
    The source-local GPU consumes the full node x3; other GPUs consume compact
    x4-selected rows. x3 sorts descending source-relative GPU mask, stably by
    original token. This uses neither A_idx nor native destination metadata.
    """
    if len(experts) != len(weights) or num_experts % (num_nodes * gpus_per_node):
        raise AssertionError("invalid original assignment dimensions")
    per_gpu = num_experts // (num_nodes * gpus_per_node)
    per_node = per_gpu * gpus_per_node
    first = (destination_node * gpus_per_node + destination_gpu) * per_gpu
    node_tokens = []
    for token, (selected, values) in enumerate(zip(experts, weights)):
        if len(selected) != len(values) or len(set(selected)) != len(selected):
            raise AssertionError("invalid original selected expert set")
        mask = 0
        for expert, value in zip(selected, values):
            if not 0 <= expert < num_experts or not math.isfinite(value) or value < 0:
                raise AssertionError("invalid original expert/weight assignment")
            if expert // per_node == destination_node:
                gpu = expert // per_gpu % gpus_per_node
                mask |= 1 << ((source_gpu - gpu) % gpus_per_node)
        if mask:
            node_tokens.append((mask, token))
    node_tokens.sort(key=lambda pair: -pair[0])
    if source_gpu != destination_gpu:
        bit = 1 << ((source_gpu - destination_gpu) % gpus_per_node)
        node_tokens = [pair for pair in node_tokens if pair[0] & bit]
    result = []
    for row, (_, token) in enumerate(node_tokens):
        for expert, value in zip(experts[token], weights[token]):
            if first <= expert < first + per_gpu:
                result.append((expert - first, row, value, token))
    return result


def configure_original_reference(up, down, library, handle, return_exchange, *, num_experts):
    """Capture original local-node assignments once, outside measured iterations.

    SharedExchange is node-local and has bounded JSON storage. Only small file
    descriptors travel through it; temporary binary assignment files are read by
    every local worker and unlinked after a collective read barrier. Remote-node
    sources still receive descriptor/source-weight checks. No distributed torch
    process group, additional GPU kernels, or data-plane exchange is introduced.
    """
    query = library.rdma_proxy_get_local_router_assignments
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                      ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                      ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                      ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    count, tokens, top_k = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
    return_exchange.require(query(handle, None, None, 0, ctypes.byref(count),
                                  ctypes.byref(tokens), ctypes.byref(top_k)))
    selected, weights = (ctypes.c_int32 * count.value)(), (ctypes.c_float * count.value)()
    return_exchange.require(query(handle, selected, weights, count.value, ctypes.byref(count),
                                  ctypes.byref(tokens), ctypes.byref(top_k)))
    if count.value != tokens.value * top_k.value:
        raise AssertionError("native original assignment dimensions disagree")
    descriptor, path = tempfile.mkstemp(prefix="rdma-route-reference-", dir="/dev/shm")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(bytes(selected))
            stream.write(bytes(weights))
        entries = return_exchange.control.gather("original-route-reference-files", {
            "path": path, "tokens": tokens.value, "top_k": top_k.value,
        })
        up.original_reference = {}
        nodes = sorted({node for node, _ in up.source_pairs})
        for source, (node, gpu) in enumerate(up.source_pairs):
            if node != return_exchange.node:
                continue
            entry = entries[gpu]
            size = entry["tokens"] * entry["top_k"]
            with open(entry["path"], "rb") as stream:
                payload = stream.read()
            if len(payload) != 8 * size:
                raise AssertionError("original assignment reference payload has invalid size")
            selected = list((ctypes.c_int32 * size).from_buffer_copy(payload[:4 * size]))
            weights = list((ctypes.c_float * size).from_buffer_copy(payload[4 * size:]))
            width = entry["top_k"]
            up.original_reference[source] = original_source_routes(
                [selected[start:start + width] for start in range(0, size, width)],
                [weights[start:start + width] for start in range(0, size, width)],
                source_gpu=gpu, destination_node=return_exchange.node,
                destination_gpu=return_exchange.rank, num_nodes=len(nodes),
                gpus_per_node=return_exchange.world, num_experts=num_experts)
        return_exchange.control.gather("original-route-reference-read")
    finally:
        os.unlink(path)


def _validate_original_routes(up, metadata, weights):
    for source, original in getattr(up, "original_reference", {}).items():
        actual = Counter((expert, token, weights[row])
                         for row, (expert, src, token) in enumerate(metadata) if src == source)
        expected = Counter((expert, token, value) for expert, token, value, _ in original)
        if actual != expected:
            raise AssertionError(f"source {source}: original token/expert/weight assignments disagree")


def _activation(torch, value, name, other=None):
    if name is None:
        return value
    name = name.removesuffix("-tanh")
    if name in ("silu", "swiglu"):
        result = torch.nn.functional.silu(value)
    elif name == "swiglu_oai":
        return value * torch.sigmoid(1.702 * value) * (other + 1)
    elif name in ("relu", "reglu"):
        result = torch.relu(value)
    elif name == "relu_sq":
        result = torch.relu(value).square()
    elif name in ("gelu_tanh_approx", "geglu"):
        result = torch.nn.functional.gelu(value, approximate="tanh")
    elif name == "glu":
        result = torch.sigmoid(value)
    elif name == "tanh":
        result = torch.tanh(value)
    else:
        raise ValueError(f"unsupported projection reference activation: {name}")
    return result if other is None else result * other


def _assert_values(torch, actual, expected, label, *, reduction=False):
    """Scale tolerances to payload magnitude, so tiny inputs cannot pass as zero."""
    actual, expected = actual.float(), expected.float()
    if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
        raise AssertionError(f"{label}: nonfinite numerical output/reference")
    if not expected.numel():
        return
    scale = max(expected.square().mean().sqrt().item(), 1e-12)
    error = actual - expected
    relative_rms = error.square().mean().sqrt().item() / scale
    limit = 0.04 if reduction else 0.015
    if relative_rms > limit:
        raise AssertionError(f"{label}: relative RMS error {relative_rms:.6g} exceeds {limit}")
    # A small number of missing routes/column bands must not hide in a global RMS.
    torch.testing.assert_close(actual, expected, rtol=0.04 if reduction else 0.015,
                               atol=scale * (0.12 if reduction else 0.04), msg=label)


def check_projections(torch, up, down, routes, *, column_chunk=256, row_chunk=512):
    """Validate every valid up row, and every down destination, against FP32 GEMMs.

    Down scales FP32 accumulators using immutable source/list weights, then
    rounds contributions to output dtype before FP32 reference scatter-add.
    Actual output-dtype atomic reductions can round again in arrival order, so
    reduction comparisons use a larger tolerance. Compact destinations are
    reconstructed without the worker's scatter-address table. Available
    original source assignments independently validate native input-row mapping
    and weights; the synthetic fixture also recomputes the entire up projection.
    """
    if column_chunk <= 0 or row_chunk <= 0:
        raise ValueError("reference chunk sizes must be positive")
    metadata = route_metadata(up, routes)
    route_weights = route_weight_reference(up, down, routes)
    _validate_original_routes(up, metadata, route_weights)
    device = up.output.device
    groups = {}
    for row, (expert, source, token) in enumerate(metadata):
        groups.setdefault((expert, source), []).append((row, token))
    # Explicitly disable TF32 for the FP32 reference, restoring the caller's policy.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            width = up.output.shape[1]
            for col in range(0, width, column_chunk):
                end = min(col + column_chunk, width)
                reference = torch.empty((routes, end - col), device=device, dtype=torch.float32)
                for (expert, source), rows in groups.items():
                    weight = up.weights[expert, :, col:end].float()
                    other_weight = (up.weights[expert, :, col + width:end + width].float()
                                    if up.gated else None)
                    for start in range(0, len(rows), row_chunk):
                        chunk = rows[start:start + row_chunk]
                        positions = torch.tensor([row for row, _ in chunk], device=device)
                        tokens = torch.tensor([token for _, token in chunk], device=device)
                        x = up.x_buffers[source][tokens].float()
                        reference[positions] = _activation(
                            torch, x @ weight, up.activation,
                            x @ other_weight if other_weight is not None else None)
                _assert_values(torch, up.output[:routes, col:end],
                               reference.to(up.output.dtype), f"up columns {col}:{end}")
            if down is None:
                print(f"Projection reference check: PASSED (up, {routes} routes)", flush=True)
                return
            outputs = down.inputs.output_buffers()
            tile = down.inputs.n_tile_major_tile_n
            if down.combine:
                used = [sorted({token for _, src, token in metadata if src == source})
                        for source in range(len(outputs))]
                sizes = [len(tokens) for tokens in used]
                compact = [{token: row for row, token in enumerate(tokens)} for tokens in used]
                offsets = [0]
                for size in sizes:
                    offsets.append(offsets[-1] + size)
                destinations = [offsets[src] + compact[src][token] for _, src, token in metadata]
                if down.combine_valid_rows.cpu().tolist() != sizes:
                    raise AssertionError("compact output counts disagree with input routing")
                for source, tokens in enumerate(used):
                    inverse = down.combine_output_to_input[source].cpu().tolist()
                    if inverse != tokens + [-1] * (len(inverse) - len(tokens)):
                        raise AssertionError("compact output-to-input map disagrees with routing")
            else:
                sizes = (list(down.inputs.scatter_output_rows)
                         if down.inputs.scatter_output_rows is not None else [routes])
                offsets = [0]
                for size in sizes:
                    offsets.append(offsets[-1] + size)
                destinations = (down.inputs.scatter_table.cpu().tolist()
                                if down.inputs.scatter_table is not None else list(range(routes)))
            destination = torch.tensor(destinations, dtype=torch.int64, device=device)
            expert_rows = {}
            originals = getattr(up, "original_reference", {})
            checked_rows = {(expert, source, token): row
                            for row, (expert, source, token) in enumerate(metadata)}
            for row, (expert, source, _) in enumerate(metadata):
                # Original assignments drive this source's contribution set;
                # the packed mapping does not select its weights/destinations.
                if source in originals and down.combine:
                    continue
                expert_rows.setdefault(expert, []).append(row)
            scale = torch.tensor(route_weights, device=device, dtype=torch.float32)
            width = down.weights.shape[-1]
            for col in range(0, width, column_chunk):
                end = min(col + column_chunk, width)
                reference = torch.zeros((sum(sizes), end - col), dtype=torch.float32, device=device)
                for expert, rows in expert_rows.items():
                    weight = down.weights[expert, :, col:end].float()
                    for start in range(0, len(rows), row_chunk):
                        positions = torch.tensor(rows[start:start + row_chunk], device=device)
                        accumulator = up.output[positions].float() @ weight
                        contribution = (accumulator * scale[positions, None]).to(outputs[0].dtype).float()
                        reference.index_add_(0, destination[positions], contribution)
                if down.combine:
                    for source, original in originals.items():
                        for expert in range(up.weights.shape[0]):
                            selected = [(token, value) for exp, token, value, _ in original
                                        if exp == expert]
                            for start in range(0, len(selected), row_chunk):
                                chunk = selected[start:start + row_chunk]
                                tokens = torch.tensor([token for token, _ in chunk], device=device)
                                values = torch.tensor([value for _, value in chunk], device=device,
                                                      dtype=torch.float32)
                                up_width = up.output.shape[1]
                                if getattr(up, "original_reference_recompute", False):
                                    # Synthetic composition tests independently compute
                                    # the entire composition. The large production check
                                    # reuses rows already checked against FP32 up GEMMs.
                                    x = up.x_buffers[source][tokens].float()
                                    intermediate = torch.empty((len(chunk), up_width), device=device,
                                                               dtype=up.output.dtype)
                                    for begin in range(0, up_width, column_chunk):
                                        stop = min(begin + column_chunk, up_width)
                                        gate = x @ up.weights[expert, :, begin:stop].float()
                                        other = (x @ up.weights[expert, :, begin + up_width:stop + up_width].float()
                                                 if up.gated else None)
                                        intermediate[:, begin:stop] = _activation(
                                            torch, gate, up.activation, other).to(up.output.dtype)
                                else:
                                    positions = torch.tensor([checked_rows[expert, source, token]
                                                              for token, _ in chunk], device=device)
                                    intermediate = up.output[positions]
                                accumulator = intermediate.float() @ down.weights[expert, :, col:end].float()
                                contribution = (accumulator * values[:, None]).to(outputs[0].dtype).float()
                                target = torch.tensor([offsets[source] + compact[source][token]
                                                       for token, _ in chunk], device=device)
                                reference.index_add_(0, target, contribution)
                columns = torch.arange(col, end, device=device)
                for source, (output, size) in enumerate(zip(outputs, sizes)):
                    if tile is None:
                        actual = output[:size, col:end]
                    else:
                        rows = torch.arange(size, device=device)[:, None]
                        flat = (columns[None, :] // tile * size + rows) * tile + columns[None, :] % tile
                        actual = output.flatten()[flat]
                    _assert_values(torch, actual, reference[offsets[source]:offsets[source + 1]],
                                   f"down source {source} columns {col}:{end}", reduction=True)
            if down.inputs.ready_groups is not None:
                expected_ready = int(down.inputs.wave_ready_prefix[-1].item())
                if down.inputs.ready_groups.item() != expected_ready:
                    raise AssertionError("down ready prefix did not cover every output group")
            print(f"Projection reference check: PASSED (up/down, {routes} routes)", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
