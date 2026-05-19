# Module Interactions — `code/src` Architecture

> Comprehensive documentation of how all modules in `code/src` collaborate to deliver CPU and GPU-accelerated genetic algorithm benchmarking for TSP instances.

---

## Table of Contents

1. [Overview — Layered Architecture](#1-overview--layered-architecture)
2. [Module Dependency Diagram](#2-module-dependency-diagram)
3. [Data Flow Sequence Diagram](#3-data-flow-sequence-diagram)
4. [Strategy Pattern Sequence Diagram](#4-strategy-pattern-sequence-diagram)
5. [ProblemContext Lifecycle Sequence Diagram](#5-problemcontext-lifecycle-sequence-diagram)
6. [Protocol Architecture Diagram](#6-protocol-architecture-diagram)
7. [GPU Helper Utilities Flowchart](#7-gpu-helper-utilities-flowchart)

---

## 1. Overview — Layered Architecture

The codebase is organized into **six distinct layers**, each with a clear responsibility boundary. Dependencies flow **downward only** — higher layers depend on lower layers, never the reverse.

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| **Benchmarking** | `benchmarking_v2/` | Orchestration, config management, statistical analysis, reporting |
| **Algorithm** | `algorithms/metaheuristics/` | GA variants (CPU, Hybrid Naive, Hybrid Optimized, Full GPU) |
| **Strategy** | `algorithms/strategies/` | Pluggable selection, crossover, and mutation operators |
| **Protocol** | `protocols/` | Interface contracts (PEP 544 structural subtyping), context management |
| **Data & Distance** | `data_models/`, `distances/`, `loaders/` | Problem representation, distance computation, TSPLIB loading |
| **Utility** | `utils/` | GPU detection, kernel compilation, backend abstraction |

### Key Design Patterns

- **Template Method** — `GeneticAlgorithmBase` defines the `evolve()` skeleton; subclasses override `_improve_population()` and `_evaluate_population()`.
- **Strategy** — Selection, crossover, and mutation are injected as composable strategy objects.
- **Protocol (Structural Subtyping)** — `BackendModule` abstracts NumPy/CuPy; algorithm strategies define duck-typed interfaces via `typing.Protocol`.
- **Lazy Loading** — `ProblemContext` defers distance matrix computation and GPU transfers until first access.
- **Context Manager** — `DatabaseLoader` manages SQLite connections via `__enter__`/`__exit__`.

---

## 2. Module Dependency Diagram

```mermaid
flowchart TB
    subgraph Benchmarking["🔴 Benchmarking Layer"]
        style Benchmarking fill:#ffe0e0,stroke:#cc0000
        ORCH["orchestration.py<br/><i>run_comprehensive_benchmark()</i>"]
        CONF["config_loader.py<br/><i>GAParams, BenchmarkConfig</i>"]
        RUNNER["algorithm_runner.py<br/><i>run_single_algorithm()</i>"]
        CKPT["checkpoint_io.py<br/><i>CheckpointManager</i>"]
        STATS["statistics.py<br/><i>StatisticalAnalyzer</i>"]
        PSTATS["problem_statistics.py<br/><i>Friedman, Nemenyi</i>"]
        ASTATS["aggregate_statistics.py<br/><i>Cross-problem stratified</i>"]
        REPORT["report_generator.py<br/><i>Markdown & LaTeX</i>"]
        SUTILS["stats_utils.py<br/><i>R², curve fitting</i>"]
    end

    subgraph Algorithm["🟢 Algorithm Layer"]
        style Algorithm fill:#e0ffe0,stroke:#008800
        BASE["genetic_algorithm_base.py<br/><i>GeneticAlgorithmBase (ABC)</i>"]
        CPU["genetic_algorithm_cpu.py<br/><i>GeneticAlgorithmCPU</i>"]
        HN["genetic_algorithm_hybrid_naive.py<br/><i>GeneticAlgorithmHybridNaive</i>"]
        HO["genetic_algorithm_hybrid_optimized.py<br/><i>GeneticAlgorithmHybridOptimized</i>"]
        FG["genetic_algorithm_full_gpu_early_stop.py<br/><i>GeneticAlgorithmFullGPUEarlyStop</i>"]
    end

    subgraph Strategy["🔵 Strategy Layer"]
        style Strategy fill:#e0e0ff,stroke:#0000cc
        SEL["selection_strategies.py<br/><i>Tournament, Roulette, Crowding</i>"]
        CRO["crossover_strategies.py<br/><i>OrderCrossover, PMX</i>"]
        MUT["mutation_strategies.py<br/><i>Swap, Inversion, Insertion</i>"]
    end

    subgraph Protocol["🟣 Protocol Layer"]
        style Protocol fill:#f0e0ff,stroke:#8800aa
        ALGP["algorithm_strategies.py<br/><i>BinPacking, TspConstruction,<br/>TspImprovement, TspMetaheuristic</i>"]
        BACK["backend.py<br/><i>BackendModule protocol</i>"]
        PCTX["problem_context.py<br/><i>ProblemContext (lazy caching)</i>"]
    end

    subgraph DataLayer["🟠 Data & Distance Layer"]
        style DataLayer fill:#fff0e0,stroke:#cc8800
        PROB["problem.py<br/><i>Problem (frozen dataclass)</i>"]
        EXC["exceptions.py<br/><i>ProblemLoaderError hierarchy</i>"]
        MAT["matrix.py<br/><i>compute_distance_matrix()</i>"]
        PAIR["pairwise.py<br/><i>Pairwise distance functions</i>"]
        LOAD["database_loader.py<br/><i>DatabaseLoader (context mgr)</i>"]
    end

    subgraph Utility["⬜ Utility Layer"]
        style Utility fill:#f0f0f0,stroke:#888888
        GPU["gpu_helpers.py<br/><i>load_kernel, get_backend,<br/>is_gpu_available</i>"]
    end

    %% Benchmarking internal dependencies
    ORCH --> CONF
    ORCH --> RUNNER
    ORCH --> CKPT
    ORCH --> PSTATS
    ORCH --> ASTATS
    ORCH --> REPORT
    PSTATS --> STATS
    ASTATS --> STATS
    ASTATS --> SUTILS
    REPORT --> STATS

    %% Benchmarking → Algorithm
    RUNNER --> CPU
    RUNNER --> HN
    RUNNER --> HO
    RUNNER --> FG

    %% Benchmarking → Data
    RUNNER --> PCTX
    ORCH --> LOAD

    %% Algorithm inheritance
    CPU --> BASE
    HN --> BASE
    HO --> BASE
    FG --> BASE

    %% Algorithm → Strategy
    BASE --> SEL
    BASE --> CRO
    BASE --> MUT

    %% Algorithm → Protocol/Context
    BASE --> PCTX

    %% GPU variants → Utility
    HN --> GPU
    HO --> GPU
    FG --> GPU

    %% Protocol → Data
    PCTX --> PROB
    PCTX --> MAT
    PCTX --> BACK

    %% Data internal
    MAT --> PAIR
    LOAD --> PROB
    LOAD --> EXC

    %% External backends
    NP["numpy"]:::external
    CP["cupy"]:::external
    BACK -.->|"abstracts"| NP
    BACK -.->|"abstracts"| CP

    classDef external fill:#ffffff,stroke:#aaaaaa,stroke-dasharray: 5 5
```

### Dependency Summary Table

| Module | Depends On | Depended On By |
|--------|-----------|----------------|
| `GeneticAlgorithmBase` | `ProblemContext`, `TournamentSelection`, `OrderCrossover`, `SwapMutation` | All 4 GA variants |
| `GeneticAlgorithmCPU` | `GeneticAlgorithmBase`, `numpy` | `algorithm_runner` |
| `GeneticAlgorithmHybridNaive` | `GeneticAlgorithmBase`, `numpy`, `cupy`, `load_kernel` | `algorithm_runner` |
| `GeneticAlgorithmHybridOptimized` | `GeneticAlgorithmBase`, `numpy`, `cupy`, `load_kernel` | `algorithm_runner` |
| `GeneticAlgorithmFullGPUEarlyStop` | `GeneticAlgorithmBase`, `numpy`, `cupy`, `load_kernel` | `algorithm_runner` |
| `ProblemContext` | `Problem`, `compute_distance_matrix`, `BackendModule` | All GA variants, `algorithm_runner` |
| `DatabaseLoader` | `Problem`, exceptions | `orchestration` |
| `compute_distance_matrix` | `pairwise` distance functions | `ProblemContext` |

---

## 3. Data Flow Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant FS as TSPLIB File /<br/>SQLite Database
    participant DL as DatabaseLoader
    participant P as Problem<br/>(frozen dataclass)
    participant CTX as ProblemContext<br/>(lazy caching)
    participant MAT as compute_distance_<br/>matrix()
    participant PAIR as pairwise.py<br/>distance functions
    participant ALG as GeneticAlgorithm<br/>Variant
    participant GPU as GPU Kernels<br/>(CuPy)
    participant BENCH as orchestration.py<br/>+ statistics

    Note over FS,BENCH: === Phase 1: Problem Loading ===

    FS->>DL: Open SQLite DB (context manager)
    DL->>DL: SQL query: problems, nodes,<br/>edge_weight_matrices
    DL->>P: Construct Problem(name, dimension,<br/>coordinates, edge_type, ...)
    DL-->>FS: Close connection (__exit__)

    Note over FS,BENCH: === Phase 2: Context Creation (No Computation) ===

    P->>CTX: ProblemContext(problem, xp=numpy)
    Note right of CTX: __init__ stores references only.<br/>_cpu_distances = None<br/>_gpu_distances = None<br/>No computation yet!

    Note over FS,BENCH: === Phase 3: Algorithm Execution ===

    BENCH->>ALG: evolve(context, customers,<br/>max_generations, optimal_cost, patience)

    ALG->>CTX: get_cpu_distances()
    Note right of CTX: First call triggers lazy load

    alt Cache Miss (_cpu_distances is None)
        CTX->>MAT: compute_distance_matrix(<br/>coordinates, edge_type, xp=np)
        MAT->>PAIR: get_distance_function(edge_type)
        PAIR-->>MAT: e.g. compute_euclidean_2d
        MAT->>MAT: Vectorized broadcasting:<br/>diff = coords[:, None] - coords[None, :]<br/>distances = sqrt(sum(diff², axis=-1))
        MAT-->>CTX: np.ndarray (n×n, float32)
        CTX->>CTX: Cache: _cpu_distances = result
    else Cache Hit
        CTX-->>ALG: Return cached _cpu_distances
    end

    CTX-->>ALG: distances (np.ndarray)

    Note over ALG,GPU: === Phase 3a: GPU Variant — Additional Transfers ===

    alt GPU Variant (Hybrid or Full GPU)
        ALG->>CTX: get_gpu_distances()
        alt GPU Cache Miss
            CTX->>CTX: cp.asarray(_cpu_distances)<br/>Transfer CPU → GPU (H2D)
            CTX->>CTX: Cache: _gpu_distances = result
        end
        CTX-->>ALG: gpu_distances (cp.ndarray on VRAM)
    end

    Note over FS,BENCH: === Phase 4: Evolution Loop ===

    loop Each Generation (up to max_generations)
        ALG->>ALG: _select_parents(population, fitness)
        ALG->>ALG: _create_offspring(population, parent_indices)
        ALG->>ALG: _improve_population(offspring, distances)<br/>[CPU: numpy 2-opt | GPU: CUDA kernels]
        ALG->>ALG: _evaluate_population(offspring, distances)
        ALG->>ALG: _survival_selection(pop, offspring, ...)
        ALG->>ALG: Check early stopping:<br/>patience counter vs improvement
    end

    ALG-->>BENCH: (best_tour, stats_dict)

    Note over FS,BENCH: === Phase 5: Statistical Analysis & Reporting ===

    BENCH->>BENCH: Collect results across<br/>runs × algorithms × instances
    BENCH->>BENCH: StatisticalAnalyzer:<br/>Shapiro-Wilk, paired t-test,<br/>Wilcoxon, Cohen's d, bootstrap CI
    BENCH->>BENCH: Friedman test + Nemenyi post-hoc<br/>(per-problem ranking)
    BENCH->>BENCH: Cross-problem stratified analysis<br/>(small/medium/large instances)
    BENCH->>BENCH: Generate Markdown & LaTeX tables
```

### Backend Switching Flow

```mermaid
sequenceDiagram
    autonumber
    participant ORCH as orchestration.py
    participant GPU as gpu_helpers.py
    participant CTX as ProblemContext
    participant NP as numpy (CPU)
    participant CP as cupy (GPU)

    ORCH->>GPU: get_backend()
    GPU->>GPU: Check: is cupy importable?<br/>Check: cp.cuda.runtime.getDeviceCount() > 0
    alt CUDA Available
        GPU-->>ORCH: xp = cupy module
    else No CUDA
        GPU-->>ORCH: xp = numpy module
    end

    ORCH->>CTX: ProblemContext(problem, xp=xp)

    Note right of CTX: For CPU algorithm:
    CTX->>NP: compute_distance_matrix(..., xp=np)
    NP-->>CTX: np.ndarray on CPU RAM

    Note right of CTX: For GPU algorithm:
    CTX->>NP: get_cpu_distances() first
    NP-->>CTX: np.ndarray on CPU RAM
    CTX->>CP: cp.asarray(cpu_distances)
    CP-->>CTX: cp.ndarray on GPU VRAM
```

---

## 4. Strategy Pattern Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Client as algorithm_runner.py
    participant GA as GeneticAlgorithmBase<br/>__init__()
    participant TS as TournamentSelection<br/>(default)
    participant OX as OrderCrossover<br/>(default)
    participant SM as SwapMutation<br/>(default)

    Note over Client,SM: === Construction Phase: Strategy Injection ===

    Client->>GA: GeneticAlgorithmCPU(<br/>  population_size=256,<br/>  mutation_rate=0.02,<br/>  tournament_size=5<br/>)
    GA->>TS: TournamentSelection(tournament_size=5)
    GA->>OX: OrderCrossover()
    GA->>SM: SwapMutation()
    GA->>GA: self.selection_strategy = TS<br/>self.crossover_strategy = OX<br/>self.mutation_strategy = SM

    Note over Client,SM: === Execution Phase: Strategy Delegation ===

    Client->>GA: evolve(context, customers,<br/>max_generations=500, ...)

    GA->>GA: population = _initialize_population(n)<br/>[256 random permutations]

    loop Each Generation
        Note over GA,SM: Step 1: SELECTION (delegate to strategy)
        GA->>TS: select(population, fitness, n_select=256)
        Note right of TS: For each selection slot:<br/>  Pick k=5 random individuals<br/>  Return index of best (lowest cost)
        TS-->>GA: parent_indices (256,)

        Note over GA,SM: Step 2: CROSSOVER (delegate to strategy)
        GA->>GA: _create_offspring(population, parent_indices)
        loop For each pair of parents
            GA->>OX: crossover(parent1, parent2, problem)
            Note right of OX: 1. Select random segment [i:j]<br/>2. Copy segment to child<br/>3. Fill remaining from parent1<br/>   in order, skipping duplicates
            OX-->>GA: child tour (valid permutation)
        end

        Note over GA,SM: Step 3: MUTATION (delegate to strategy)
        loop For each offspring (with probability mutation_rate)
            alt random() < mutation_rate
                GA->>SM: mutate(tour, problem)
                Note right of SM: Pick two random indices i, j<br/>Swap: tour[i] ↔ tour[j]
                SM-->>GA: mutated tour
            end
        end

        Note over GA,SM: Step 4: IMPROVEMENT (subclass-specific)
        GA->>GA: _improve_population(offspring, distances)<br/>[Abstract — implemented by variant]

        Note over GA,SM: Step 5: EVALUATION (subclass-specific)
        GA->>GA: _evaluate_population(offspring, distances)<br/>[Abstract — implemented by variant]

        Note over GA,SM: Step 6: SURVIVAL SELECTION
        GA->>GA: _survival_selection(pop, offspring,<br/>pop_fitness, off_fitness)<br/>[(μ+λ): merge & keep best 256]
    end

    GA-->>Client: (best_tour, stats)
```

### Strategy Interchangeability

```mermaid
flowchart LR
    subgraph Selection["Selection Strategies"]
        TS["TournamentSelection<br/>select(pop, fit, n)"]
        RW["RouletteWheelSelection<br/>select(pop, fit, n)"]
        CS["CrowdingSelection<br/>select(pop, fit, n)"]
    end

    subgraph Crossover["Crossover Strategies"]
        OX["OrderCrossover<br/>crossover(p1, p2, prob)"]
        PMX["PartiallyMappedCrossover<br/>crossover(p1, p2, prob)"]
    end

    subgraph Mutation["Mutation Strategies"]
        SW["SwapMutation<br/>mutate(tour, prob)"]
        INV["InversionMutation<br/>mutate(tour, prob)"]
        INS["InsertionMutation<br/>mutate(tour, prob)"]
    end

    BASE["GeneticAlgorithmBase"]
    BASE -->|"self.selection_strategy"| TS
    BASE -.->|"alternative"| RW
    BASE -.->|"alternative"| CS
    BASE -->|"self.crossover_strategy"| OX
    BASE -.->|"alternative"| PMX
    BASE -->|"self.mutation_strategy"| SW
    BASE -.->|"alternative"| INV
    BASE -.->|"alternative"| INS

    style TS fill:#aaddff
    style OX fill:#aaddff
    style SW fill:#aaddff
```

> **Default strategies** (solid arrows) are `TournamentSelection`, `OrderCrossover`, and `SwapMutation`. Alternatives (dashed arrows) can be swapped in without modifying the base class, thanks to duck-typed protocol compatibility.

---

## 5. ProblemContext Lifecycle Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant ORCH as orchestration.py
    participant CTX as ProblemContext
    participant MAT as compute_distance_<br/>matrix()
    participant CPU_MEM as CPU Memory<br/>(NumPy)
    participant GPU_MEM as GPU VRAM<br/>(CuPy)

    Note over ORCH,GPU_MEM: === Phase 1: Creation (Zero Computation) ===

    ORCH->>CTX: ProblemContext(problem, xp=numpy, seed=42)
    Note right of CTX: Stores references only:<br/>self.problem = problem<br/>self.xp = xp<br/>self.dimension = problem.dimension<br/>self.capacity = problem.capacity<br/>self.seed = seed<br/>self.rng = numpy.random.default_rng(seed)<br/><br/>All caches initialized to None:<br/>self._cpu_distances = None<br/>self._gpu_distances = None<br/>self._cpu_demands = None<br/>self._gpu_demands = None<br/>self._cpu_coordinates = None<br/>self._gpu_coordinates = None

    Note over ORCH,GPU_MEM: === Phase 2: CPU Algorithm Usage ===

    ORCH->>CTX: get_cpu_distances()
    Note right of CTX: CACHE MISS: _cpu_distances is None

    CTX->>MAT: compute_distance_matrix(<br/>problem.coordinates,<br/>problem.edge_type,<br/>xp=numpy)
    MAT->>MAT: Vectorized computation:<br/>diff = coords[:, None, :] - coords[None, :, :]<br/>dist = sqrt((diff**2).sum(axis=-1))<br/>return dist.astype(float32)
    MAT-->>CTX: np.ndarray shape (n, n)
    CTX->>CPU_MEM: Store _cpu_distances = result
    CTX-->>ORCH: np.ndarray (n×n)

    ORCH->>CTX: get_cpu_distances()
    Note right of CTX: CACHE HIT: return _cpu_distances
    CTX-->>ORCH: Same np.ndarray (zero recomputation)

    Note over ORCH,GPU_MEM: === Phase 3: GPU Algorithm Usage ===

    ORCH->>CTX: get_gpu_distances()
    Note right of CTX: CACHE MISS: _gpu_distances is None

    alt CPU distances not yet computed
        CTX->>CTX: get_cpu_distances() [triggers Phase 2]
    end

    CTX->>GPU_MEM: cp.asarray(_cpu_distances)<br/>Host-to-Device transfer<br/>Size: n² × 4 bytes (float32)
    GPU_MEM-->>CTX: cp.ndarray on GPU VRAM
    CTX->>CTX: _gpu_distances = result
    CTX-->>ORCH: cp.ndarray (n×n on GPU)

    ORCH->>CTX: get_gpu_distances()
    Note right of CTX: CACHE HIT: return _gpu_distances<br/>No transfer!
    CTX-->>ORCH: Same cp.ndarray

    Note over ORCH,GPU_MEM: === Phase 4: Auxiliary Data ===

    ORCH->>CTX: get_cpu_demands()
    CTX-->>ORCH: problem.demands or None

    ORCH->>CTX: get_gpu_demands()
    alt demands is not None
        CTX->>GPU_MEM: cp.asarray(demands)
        GPU_MEM-->>CTX: cp.ndarray
    end
    CTX-->>ORCH: cp.ndarray or None

    ORCH->>CTX: get_cpu_coordinates()
    CTX-->>ORCH: problem.coordinates or None

    ORCH->>CTX: get_gpu_coordinates()
    alt coordinates is not None
        CTX->>GPU_MEM: cp.asarray(coordinates)
        GPU_MEM-->>CTX: cp.ndarray
    end
    CTX-->>ORCH: cp.ndarray or None
```

### How Each Algorithm Variant Uses ProblemContext

```mermaid
flowchart TB
    CTX["ProblemContext"]

    subgraph CPU_Variant["GeneticAlgorithmCPU"]
        CPU_D["get_cpu_distances() only<br/>All computation on CPU<br/>H2D: 0 bytes | D2H: 0 bytes"]
    end

    subgraph HN_Variant["GeneticAlgorithmHybridNaive"]
        HN_CD["get_cpu_distances() — for GA ops"]
        HN_GD["get_gpu_distances() — for 2-opt kernel"]
        HN_T["256 individual H2D transfers per gen<br/>256 individual D2H transfers per gen<br/>~20 MB/gen for n=1000"]
    end

    subgraph HO_Variant["GeneticAlgorithmHybridOptimized"]
        HO_CD["get_cpu_distances() — initial only"]
        HO_GD["get_gpu_distances() — batch kernel"]
        HO_T["1 batch H2D transfer per gen<br/>D2H: costs only (2 KB)<br/>Kernel chaining: 2-opt → cost calc"]
    end

    subgraph FG_Variant["GeneticAlgorithmFullGPUEarlyStop"]
        FG_GD["get_gpu_distances() — once at start"]
        FG_T["H2D: n²×8 bytes (distance matrix, once)<br/>D2H: n×4 bytes (best tour, once)<br/>3 kernel launches total"]
    end

    CTX --> CPU_D
    CTX --> HN_CD
    CTX --> HN_GD
    CTX --> HO_CD
    CTX --> HO_GD
    CTX --> FG_GD

    style CPU_Variant fill:#e8ffe8
    style HN_Variant fill:#fff0e0
    style HO_Variant fill:#e0f0ff
    style FG_Variant fill:#ffe0e0
```

### Memory Transfer Comparison

| Variant | H2D per Generation | D2H per Generation | Kernel Launches/Gen | Total H2D (500 gen, n=1000) |
|---------|-------------------|-------------------|--------------------|-----------------------------|
| **CPU** | 0 | 0 | 0 | 0 |
| **Hybrid Naive** | 256 × n × 4B × 10 | 256 × n × 4B × 10 | 256 | ~5 GB |
| **Hybrid Optimized** | 256 × n × 4B × 1 | 256 × 4B (costs) | 2 (batch 2-opt + cost) | ~500 MB |
| **Full GPU** | 0 (data stays on GPU) | 0 (data stays on GPU) | 0 (monolithic) | n² × 8B once (~8 MB) |

---

## 6. Protocol Architecture Diagram

```mermaid
classDiagram
    class BackendModule {
        <<Protocol>>
        +array(object, dtype) ndarray
        +asarray(a, dtype) ndarray
        +zeros(shape, dtype) ndarray
        +ones(shape, dtype) ndarray
        +empty(shape, dtype) ndarray
        +arange(start, stop, step) ndarray
        +sum(a, axis) ndarray
        +min(a, axis) ndarray
        +max(a, axis) ndarray
        +argmin(a, axis) int
        +argmax(a, axis) int
        +sqrt(x) ndarray
        +exp(x) ndarray
        +random RandomState
    }

    class numpy {
        +array()
        +asarray()
        +zeros()
        +ones()
        +empty()
        +arange()
        +sum()
        +min()
        +max()
        +argmin()
        +argmax()
        +sqrt()
        +exp()
        +random
    }

    class cupy {
        +array()
        +asarray()
        +zeros()
        +ones()
        +empty()
        +arange()
        +sum()
        +min()
        +max()
        +argmin()
        +argmax()
        +sqrt()
        +exp()
        +random
    }

    BackendModule <|.. numpy : satisfies (structural)
    BackendModule <|.. cupy : satisfies (structural)

    class BinPackingStrategy {
        <<Protocol>>
        +pack(context: ProblemContext, customer_indices: List~int~) List~List~int~~
    }

    class TspConstructionStrategy {
        <<Protocol>>
        +construct(context: ProblemContext, customers: ndarray) ndarray
    }

    class ClusteringStrategy {
        <<Protocol>>
        +cluster(context: ProblemContext, customers: ndarray) List~List~int~~
    }

    class TspImprovementStrategy {
        <<Protocol>>
        +improve(tour: ndarray, distances: ndarray) ndarray
    }

    class TspMetaheuristicStrategy {
        <<Protocol>>
        +evolve(context: ProblemContext, customers: ndarray, max_generations: int, optimal_cost: float, patience: int) tuple
    }

    class TournamentSelection {
        -tournament_size: int
        +select(population, fitness, n_select) ndarray
    }

    class RouletteWheelSelection {
        -epsilon: float
        +select(population, fitness, n_select) ndarray
    }

    class CrowdingSelection {
        +select(population, fitness, n_select) ndarray
    }

    class OrderCrossover {
        +crossover(parent1, parent2, problem) ndarray
    }

    class PartiallyMappedCrossover {
        +crossover(parent1, parent2, problem) ndarray
    }

    class SwapMutation {
        +mutate(tour, problem) ndarray
    }

    class InversionMutation {
        +mutate(tour, problem) ndarray
    }

    class InsertionMutation {
        +mutate(tour, problem) ndarray
    }

    class GeneticAlgorithmBase {
        <<Abstract>>
        #selection_strategy: TournamentSelection
        #crossover_strategy: OrderCrossover
        #mutation_strategy: SwapMutation
        #population_size: int
        #mutation_rate: float
        +evolve(context, customers, max_gen, optimal, patience) tuple
        +_initialize_population(n)* ndarray
        +_select_parents(pop, fitness)* ndarray
        +_create_offspring(pop, parents)* ndarray
        +_improve_population(offspring, distances, xp)* ndarray
        +_evaluate_population(pop, distances, xp)* ndarray
        +_survival_selection(pop, off, pf, of)* ndarray
    }

    class GeneticAlgorithmCPU {
        +_improve_population() ndarray
        +_evaluate_population() ndarray
    }

    class GeneticAlgorithmHybridNaive {
        -two_opt_kernel: RawKernel
        +_improve_population() ndarray
        +_evaluate_population() ndarray
    }

    class GeneticAlgorithmHybridOptimized {
        -two_opt_batch_kernel: RawKernel
        -cost_kernel: RawKernel
        +_improve_population() ndarray
        +_evaluate_population() ndarray
    }

    class GeneticAlgorithmFullGPUEarlyStop {
        -init_rand_kernel: RawKernel
        -init_pop_kernel: RawKernel
        -evolution_kernel: RawKernel
        +evolve() tuple
    }

    GeneticAlgorithmBase <|-- GeneticAlgorithmCPU
    GeneticAlgorithmBase <|-- GeneticAlgorithmHybridNaive
    GeneticAlgorithmBase <|-- GeneticAlgorithmHybridOptimized
    GeneticAlgorithmBase <|-- GeneticAlgorithmFullGPUEarlyStop

    TspMetaheuristicStrategy <|.. GeneticAlgorithmBase : satisfies (structural)

    class ProblemContext {
        +problem: Problem
        +xp: BackendModule
        +dimension: int
        +capacity: Optional~int~
        +seed: Optional~int~
        +rng: Generator
        -_cpu_distances: Optional~ndarray~
        -_gpu_distances: Optional~ndarray~
        -_cpu_demands: Optional~ndarray~
        -_gpu_demands: Optional~ndarray~
        -_cpu_coordinates: Optional~ndarray~
        -_gpu_coordinates: Optional~ndarray~
        +get_cpu_distances() ndarray
        +get_gpu_distances() ndarray
        +get_cpu_demands() Optional~ndarray~
        +get_gpu_demands() Optional~ndarray~
        +get_cpu_coordinates() Optional~ndarray~
        +get_gpu_coordinates() Optional~ndarray~
    }

    class Problem {
        <<frozen dataclass>>
        +name: str
        +dimension: int
        +problem_type: str
        +edge_type: str
        +coordinates: Optional~ndarray~
        +distances: Optional~ndarray~
        +capacity: Optional~int~
        +demands: Optional~ndarray~
    }

    ProblemContext --> Problem : contains
    ProblemContext --> BackendModule : uses xp
```

### Structural Subtyping (PEP 544) Key Concept

Unlike nominal inheritance, protocols use **structural subtyping**:

```
# No explicit inheritance needed!
# numpy satisfies BackendModule because it HAS the right attributes/methods:
#   numpy.array ✓  numpy.zeros ✓  numpy.sum ✓  ...

# Similarly, GeneticAlgorithmBase satisfies TspMetaheuristicStrategy
# because it HAS the evolve() method with the right signature.
```

This means:
- `numpy` and `cupy` **never** import `BackendModule`
- `GeneticAlgorithmBase` **never** imports `TspMetaheuristicStrategy`
- Type checkers (mypy) verify compatibility at check time, not at runtime

---

## 7. GPU Helper Utilities Flowchart

```mermaid
flowchart TB
    subgraph load_kernel["load_kernel(kernel_filename, kernel_function_name, caller_file_path)"]
        LK_START([Start]) --> LK_RESOLVE["Resolve kernel directory:<br/>caller_dir = Path(caller_file_path).parent<br/>kernel_dir = caller_dir / 'kernels'"]
        LK_RESOLVE --> LK_PATH["kernel_path = kernel_dir / kernel_filename<br/>e.g., 'two_opt_batch.cu'"]
        LK_PATH --> LK_EXISTS{kernel_path<br/>exists?}
        LK_EXISTS -->|No| LK_ERROR["Raise FileNotFoundError:<br/>'Kernel file not found: {kernel_path}'"]
        LK_EXISTS -->|Yes| LK_READ["Read kernel source:<br/>source = kernel_path.read_text()"]
        LK_READ --> LK_COMPILE["Compile with CuPy:<br/>kernel = cp.RawKernel(<br/>  source,<br/>  kernel_function_name,<br/>  options=('--std=c++11',)<br/>)"]
        LK_COMPILE --> LK_RETURN(["Return cp.RawKernel"])
    end

    subgraph get_backend["get_backend()"]
        GB_START([Start]) --> GB_TRY["Try: import cupy as cp"]
        GB_TRY --> GB_IMPORT{Import<br/>succeeded?}
        GB_IMPORT -->|No (ImportError)| GB_NP["Return numpy module"]
        GB_IMPORT -->|Yes| GB_DEVICE["Try: cp.cuda.runtime<br/>.getDeviceCount()"]
        GB_DEVICE --> GB_COUNT{Device<br/>count > 0?}
        GB_COUNT -->|No| GB_NP
        GB_COUNT -->|Yes| GB_CP["Return cupy module"]
    end

    subgraph is_gpu_available["is_gpu_available()"]
        IG_START([Start]) --> IG_TRY["Try: import cupy as cp"]
        IG_TRY --> IG_IMPORT{Import<br/>succeeded?}
        IG_IMPORT -->|No| IG_FALSE["Return False"]
        IG_IMPORT -->|Yes| IG_PROBE["Try: cp.cuda.runtime<br/>.getDeviceCount()"]
        IG_PROBE --> IG_AVAIL{count > 0 and<br/>no exception?}
        IG_AVAIL -->|No| IG_FALSE
        IG_AVAIL -->|Yes| IG_TRUE["Return True"]
    end

    subgraph Usage["Usage by Algorithm Variants"]
        HN_USE["GeneticAlgorithmHybridNaive.__init__():<br/>self.kernel = load_kernel(<br/>  'two_opt_single.cu',<br/>  'two_opt_single',<br/>  __file__<br/>)"]

        HO_USE["GeneticAlgorithmHybridOptimized.__init__():<br/>self.two_opt_kernel = load_kernel(<br/>  'two_opt_batch.cu',<br/>  'two_opt_batch',<br/>  __file__<br/>)<br/>self.cost_kernel = load_kernel(<br/>  'cost_calculator.cu',<br/>  'calculate_costs',<br/>  __file__<br/>)"]

        FG_USE["GeneticAlgorithmFullGPUEarlyStop.__init__():<br/>self.init_rand = load_kernel(<br/>  'ga_fujimoto_early_stop.cu',<br/>  'init_rand_states', __file__<br/>)<br/>self.init_pop = load_kernel(<br/>  'ga_fujimoto_early_stop.cu',<br/>  'initialize_population', __file__<br/>)<br/>self.evolve_kernel = load_kernel(<br/>  'ga_fujimoto_early_stop.cu',<br/>  'ga_evolution_early_stop', __file__<br/>)"]
    end

    load_kernel --> HN_USE
    load_kernel --> HO_USE
    load_kernel --> FG_USE
    get_backend --> ORCH_USE["orchestration.py:<br/>xp = get_backend()"]
    is_gpu_available --> CHECK_USE["config / runner:<br/>if is_gpu_available(): ..."]

    style load_kernel fill:#f9f9f9,stroke:#333
    style get_backend fill:#f9f9f9,stroke:#333
    style is_gpu_available fill:#f9f9f9,stroke:#333
    style Usage fill:#fffff0,stroke:#999
```

### Kernel File Resolution

```
code/src/algorithms/metaheuristics/
├── genetic_algorithm_hybrid_naive.py      ← caller (__file__)
├── genetic_algorithm_hybrid_optimized.py  ← caller (__file__)
├── genetic_algorithm_full_gpu_early_stop.py ← caller (__file__)
└── kernels/                               ← resolved via Path(__file__).parent / 'kernels'
    ├── two_opt_single.cu                  ← Used by HybridNaive
    ├── two_opt_batch.cu                   ← Used by HybridOptimized
    ├── cost_calculator.cu                 ← Used by HybridOptimized
    └── ga_fujimoto_early_stop.cu          ← Used by FullGPUEarlyStop (3 functions)
```

---

## Appendix: Benchmarking Orchestration Module Map

```mermaid
flowchart TB
    subgraph Orchestration["benchmarking_v2/ — 9 Modules"]
        ORCH["orchestration.py<br/>─────────────────<br/>run_comprehensive_benchmark()<br/>Main entry point"]
        CONF["config_loader.py<br/>─────────────────<br/>GAParams dataclass<br/>AlgorithmConfig dataclass<br/>ProblemConfig dataclass<br/>BenchmarkConfig dataclass<br/>load_all_configs()"]
        RUNNER["algorithm_runner.py<br/>─────────────────<br/>adaptive_generations(n)<br/>run_single_algorithm()"]
        CKPT["checkpoint_io.py<br/>─────────────────<br/>CheckpointManager<br/>save_checkpoint()<br/>load_checkpoint()"]
        STATS["statistics.py<br/>─────────────────<br/>StatisticalAnalyzer<br/>StatisticalSummary<br/>Shapiro-Wilk normality<br/>Paired t-test / Wilcoxon<br/>Cohen's d effect size<br/>Bootstrap confidence intervals"]
        PSTATS["problem_statistics.py<br/>─────────────────<br/>perform_statistical_analysis()<br/>Friedman rank test<br/>Nemenyi post-hoc"]
        ASTATS["aggregate_statistics.py<br/>─────────────────<br/>perform_cross_problem_<br/>analysis_stratified()<br/>Size-stratified analysis"]
        REPORT["report_generator.py<br/>─────────────────<br/>generate_result_tables()<br/>Markdown output<br/>LaTeX output"]
        SUTILS["stats_utils.py<br/>─────────────────<br/>calculate_r2()<br/>adjusted_r2()<br/>Curve fitting helpers"]
    end

    ORCH --> CONF
    ORCH --> RUNNER
    ORCH --> CKPT
    ORCH --> PSTATS
    ORCH --> ASTATS
    ORCH --> REPORT
    PSTATS --> STATS
    ASTATS --> STATS
    ASTATS --> SUTILS
    REPORT --> STATS

    style Orchestration fill:#fff0f0,stroke:#cc0000
```

---

*Generated for the `gpu_accelerated` project — documenting the complete module interaction architecture of `code/src/`.*
