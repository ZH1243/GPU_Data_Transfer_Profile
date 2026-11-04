import unittest
from topology import rank_topology, validate_hosts, rdma_device


class TopologyTests(unittest.TestCase):
    def test_rank_mapping_and_invalid_launches(self):
        for world in (8, 16):
            for rank in range(world):
                env = dict(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank % 8), LOCAL_WORLD_SIZE="8")
                self.assertEqual(rank_topology(env), (world, rank, rank % 8, world // 8, rank // 8))
        for env in ({}, dict(WORLD_SIZE="16", RANK="8", LOCAL_RANK="8", LOCAL_WORLD_SIZE="8"),
                    dict(WORLD_SIZE="16", RANK="16", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8")):
            with self.assertRaises(ValueError):
                rank_topology(env)

    def test_local_configuration_and_host_groups(self):
        configs = [(f"host{r // 8}", dict(tokens=257, rdma_device_map=f"nic{r // 8}",
                                         cpu_core_map=str(r // 8), output=f"node{r // 8}.json")) for r in range(16)]
        validate_hosts(configs, 2)
        wrong = list(configs)
        wrong[9] = ("host0", configs[9][1])
        with self.assertRaises(ValueError):
            validate_hosts(wrong, 2)
        wrong = list(configs)
        wrong[8] = ("host1", dict(configs[8][1], tokens=258))
        with self.assertRaises(ValueError):
            validate_hosts(wrong, 2)

    def test_two_nodes_may_report_the_same_hostname(self):
        configs = [("container", dict(tokens=257, rdma_device_map=f"nic{r // 8}"))
                   for r in range(16)]
        validate_hosts(configs, 2)
        # Shared hostname must not bypass the workload consistency checks.
        configs[8] = ("container", dict(tokens=258, rdma_device_map="nic1"))
        with self.assertRaises(ValueError):
            validate_hosts(configs, 2)

    def test_nic_mapping(self):
        self.assertEqual(rdma_device("mlx5_0", 7), "mlx5_0")
        self.assertEqual(rdma_device(",".join(f"nic{i}" for i in range(8)), 6), "nic6")
        for value in ("", "one,two", "one,", ",".join([" "] * 8)):
            with self.assertRaises(ValueError):
                rdma_device(value, 0)


if __name__ == "__main__":
    unittest.main()
