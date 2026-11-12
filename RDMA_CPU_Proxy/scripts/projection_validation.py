"""Untimed numerical checks for the actual embedded up/down GEMM composition.

Reference matrices are bounded by ``column_chunk``; weights are converted to
FP32 one column block at a time. Call only after the iteration's GPU work and
return/combine consumers have completed, before reusing any input buffers.
"""

def route_metadata(up, routes):
    """Decode N-group zero independently of the GPU combine mapping helper."""
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
                result[offset] = (expert, source, token)
                offset += 1
    if any(row is None for row in result):
        raise AssertionError("gather descriptors leave routed output rows uncovered")
    return result


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

    Down uses the already-checked BF16/FP16 up output, and rounds individual down
    contributions to output dtype before FP32 scatter-add, matching the bulk
    reduction contract. Compact destinations are reconstructed from source token
    IDs rather than trusting the worker's GPU mapping or scatter-address table.
    """
    if column_chunk <= 0 or row_chunk <= 0:
        raise ValueError("reference chunk sizes must be positive")
    metadata = route_metadata(up, routes)
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
            for row, (expert, _, _) in enumerate(metadata):
                expert_rows.setdefault(expert, []).append(row)
            width = down.weights.shape[-1]
            for col in range(0, width, column_chunk):
                end = min(col + column_chunk, width)
                reference = torch.zeros((sum(sizes), end - col), dtype=torch.float32, device=device)
                for expert, rows in expert_rows.items():
                    weight = down.weights[expert, :, col:end].float()
                    for start in range(0, len(rows), row_chunk):
                        positions = torch.tensor(rows[start:start + row_chunk], device=device)
                        contribution = (up.output[positions].float() @ weight).to(outputs[0].dtype).float()
                        reference.index_add_(0, destination[positions], contribution)
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
