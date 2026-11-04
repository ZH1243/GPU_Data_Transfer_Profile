"""Host-only launch and setup validation shared with tests."""
def rank_topology(env):
    world = int(env.get("WORLD_SIZE", "0"))
    rank = int(env.get("RANK", "-1"))
    local = int(env.get("LOCAL_RANK", "-1"))
    if (world not in (8, 16) or int(env.get("LOCAL_WORLD_SIZE", "0")) != 8
            or not 0 <= rank < world or local != rank % 8):
        raise ValueError("launch with one or two nodes and eight processes per node")
    return world, rank, local, world // 8, rank // 8


def validate_hosts(configs, nodes):
    if len(configs) != nodes * 8:
        raise ValueError("incorrect setup participant count")
    for node in range(nodes):
        if len({configs[node * 8 + g][0] for g in range(8)}) != 1:
            raise ValueError("each eight-rank node group must share a host")
    if nodes == 2 and configs[0][0] == configs[8][0]:
        raise ValueError("two-node mode requires two distinct hosts")
    # Device/core maps, library locations and output paths may differ by node.
    local_options = {"rdma_device_map", "cpu_core_map", "cores", "library", "output"}
    comparable = [{k: v for k, v in config.items() if k not in local_options}
                  for _, config in configs]
    if any(c != comparable[0] for c in comparable):
        raise ValueError("all ranks must use identical workload/protocol options")
    for node in range(nodes):
        if any(c != configs[node * 8][1] for _, c in configs[node * 8:node * 8 + 8]):
            raise ValueError("all ranks on a node must use identical options")


def rdma_device(mapping, local):
    devices = mapping.split(",")
    if len(devices) not in (1, 8) or any(not d.strip() for d in devices):
        raise ValueError("two-node mode requires --rdma-device-map with one or eight NIC names")
    return devices[0 if len(devices) == 1 else local].strip()
