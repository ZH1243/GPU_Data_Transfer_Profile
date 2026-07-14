# System Architecture Graph 01: CPU Proxy, RDMA, and NVLink

```mermaid
flowchart LR
    %% High-level GPU server architecture for RDMA_CPU_Proxy.
    %% Solid arrows show data movement actuated by hardware engines.
    %% Dashed arrows show CPU-side control/command paths.

    subgraph Server["GPU server"]
        direction TB

        CPU["Multi-core CPU"]

        subgraph P0["CPU proxy for GPU 0"]
            direction TB
            P0_RDMA["RDMA management\nQP setup, WR posting,\ndoorbells, CQ polling"]
            P0_NV["NVLink forwarding\nCUDA API calls,\nrouting/batch logic"]
        end

        subgraph P1["CPU proxy for GPU 1"]
            direction TB
            P1_RDMA["RDMA management"]
            P1_NV["NVLink forwarding"]
        end

        subgraph PK["CPU proxy for GPU N"]
            direction TB
            PK_RDMA["RDMA management"]
            PK_NV["NVLink forwarding"]
        end

        subgraph G0["GPU 0"]
            direction TB
            G0_MEM["GPU memory\nsend/recv token buffers"]
            G0_CE["GPU copy engine"]
        end

        subgraph G1["GPU 1"]
            direction TB
            G1_MEM["GPU memory\nsend/recv token buffers"]
            G1_CE["GPU copy engine"]
        end

        subgraph GK["GPU N"]
            direction TB
            GK_MEM["GPU memory\nsend/recv token buffers"]
            GK_CE["GPU copy engine"]
        end

        NIC0["Closest NIC for GPU 0"]
        NIC1["Closest NIC for GPU 1"]
        NICK["Closest NIC for GPU N"]
    end

    Remote["Remote GPU server(s)\nmatching GPU-index proxies"]

    CPU -.->|pins/hosts one proxy process per GPU| P0
    CPU -.->|pins/hosts one proxy process per GPU| P1
    CPU -.->|pins/hosts one proxy process per GPU| PK

    P0_RDMA -.->|posts RDMA writes,<br/>rings NIC doorbells| NIC0
    P1_RDMA -.->|posts RDMA writes,<br/>rings NIC doorbells| NIC1
    PK_RDMA -.->|posts RDMA writes,<br/>rings NIC doorbells| NICK

    NIC0 <==>|GPUDirect RDMA DMA<br/>fetch/write GPU memory| G0_MEM
    NIC1 <==>|GPUDirect RDMA DMA<br/>fetch/write GPU memory| G1_MEM
    NICK <==>|GPUDirect RDMA DMA<br/>fetch/write GPU memory| GK_MEM

    NIC0 <==>|RDMA network| Remote
    NIC1 <==>|RDMA network| Remote
    NICK <==>|RDMA network| Remote

    P0_NV -.->|invokes CUDA D2D copy| G0_CE
    P1_NV -.->|invokes CUDA D2D copy| G1_CE
    PK_NV -.->|invokes CUDA D2D copy| GK_CE

    G0_CE ==>|NVLink data movement| G1_MEM
    G0_CE ==>|NVLink data movement| GK_MEM
    G1_CE ==>|NVLink data movement| G0_MEM
    G1_CE ==>|NVLink data movement| GK_MEM
    GK_CE ==>|NVLink data movement| G0_MEM
    GK_CE ==>|NVLink data movement| G1_MEM

    classDef cpu fill:#f3f5f7,stroke:#48515a,color:#111;
    classDef proxy fill:#e7f1ff,stroke:#2d6cdf,color:#111;
    classDef gpu fill:#e9f8ef,stroke:#21824b,color:#111;
    classDef nic fill:#fff4dd,stroke:#b26b00,color:#111;
    classDef remote fill:#f6eafe,stroke:#7c3fb2,color:#111;

    class CPU cpu;
    class P0,P1,PK,P0_RDMA,P0_NV,P1_RDMA,P1_NV,PK_RDMA,PK_NV proxy;
    class G0,G1,GK,G0_MEM,G0_CE,G1_MEM,G1_CE,GK_MEM,GK_CE gpu;
    class NIC0,NIC1,NICK nic;
    class Remote remote;
```

Legend:

- Dashed arrows: CPU proxy control path.
- Thick arrows: hardware-actuated data movement.
- Each proxy owns one local GPU and uses that GPU's closest NIC for RDMA.
- RDMA payload bytes move directly between the NIC and GPU memory through GPUDirect RDMA.
- NVLink forwarding is submitted by the CPU proxy through CUDA, then performed by GPU copy engines.
