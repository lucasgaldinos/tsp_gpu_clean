-- ============================================================================
-- DuckDB Database Schema for GPU-Accelerated Routing Optimization Benchmarks
-- ============================================================================
-- Purpose: Store benchmark results from routing optimization algorithms
-- Database: results.duckdb
-- Created: 2025-11-23
-- Updated: 2025-11-23 - Simplified to single-table design
-- ============================================================================

-- ============================================================================
-- TABLE: benchmark_runs
-- ============================================================================
-- Purpose: Main results table storing complete benchmark data
-- Schema: Exact match to checkpoint JSON structure (no normalization)
-- Row structure: One row per checkpoint file (~130 total rows)
-- Storage: Raw arrays stored as DuckDB LIST type for exact reproduction
-- Design: Single table with all data from checkpoint JSONs (self-contained)
-- ============================================================================

CREATE SEQUENCE IF NOT EXISTS benchmark_runs_seq;

CREATE TABLE IF NOT EXISTS benchmark_runs (
    -- ========================================================================
    -- Primary Key (auto-incremented from sequence)
    -- ========================================================================
    run_id INTEGER PRIMARY KEY DEFAULT nextval('benchmark_runs_seq'),
    
    -- ========================================================================
    -- Problem Identification (from checkpoint JSON)
    -- ========================================================================
    problem_name VARCHAR NOT NULL,
    problem_size INTEGER NOT NULL,
    optimal_cost DOUBLE NOT NULL,
    
    -- ========================================================================
    -- Algorithm Identification
    -- ========================================================================
    algorithm VARCHAR NOT NULL,         -- "CPU", "HybridNaive", "HybridOptimized", "FullGPU"
    backend VARCHAR NOT NULL,           -- "CPU" or "GPU"
    
    -- ========================================================================
    -- Run Metadata
    -- ========================================================================
    repetitions INTEGER NOT NULL,       -- Number of runs (2=test, 15=production)
    successful_runs INTEGER NOT NULL,   -- Number that completed successfully
    timestamp TIMESTAMP NOT NULL,       -- When benchmark was run (ISO 8601)
    max_generations_configured INTEGER NOT NULL,  -- Max generations allowed
    
    -- ========================================================================
    -- Aggregate Statistics: Execution Time (seconds)
    -- ========================================================================
    mean_time DOUBLE NOT NULL,
    std_time DOUBLE NOT NULL,
    min_time DOUBLE NOT NULL,
    max_time DOUBLE NOT NULL,
    
    -- ========================================================================
    -- Aggregate Statistics: Solution Cost/Quality
    -- ========================================================================
    mean_cost DOUBLE NOT NULL,
    std_cost DOUBLE NOT NULL,
    best_cost DOUBLE NOT NULL,
    worst_cost DOUBLE NOT NULL,
    mean_initial_cost DOUBLE,           -- Average random initialization cost
    std_initial_cost DOUBLE,
    
    -- ========================================================================
    -- Aggregate Statistics: Gap from Optimal (percentage)
    -- ========================================================================
    mean_gap DOUBLE NOT NULL,           -- Average % above optimal
    std_gap DOUBLE NOT NULL,
    best_gap DOUBLE NOT NULL,           -- Best % above optimal (often 0.0)
    mean_improvement DOUBLE,            -- % improvement from initial to final
    
    -- ========================================================================
    -- Aggregate Statistics: Generations/Iterations
    -- ========================================================================
    mean_generations DOUBLE NOT NULL,
    std_generations DOUBLE NOT NULL,
    min_generations INTEGER,
    max_generations INTEGER,
    
    -- ========================================================================
    -- Aggregate Statistics: GPU Memory Transfer (MB)
    -- ========================================================================
    mean_h2d_mb DOUBLE,                 -- Host to Device transfer
    mean_d2h_mb DOUBLE,                 -- Device to Host transfer
    total_transfer_mb DOUBLE,           -- Total bidirectional transfer
    mean_kernels DOUBLE,                -- Average kernel launches per run
    
    -- ========================================================================
    -- Raw Arrays: Exact Reproduction Data
    -- DuckDB LIST type (variable length): INTEGER[], DOUBLE[], VARCHAR[]
    -- ========================================================================
    raw_times DOUBLE[] NOT NULL,        -- [run1_time, run2_time, ...]
    raw_costs DOUBLE[] NOT NULL,        -- [run1_cost, run2_cost, ...]
    raw_gaps DOUBLE[] NOT NULL,         -- [run1_gap, run2_gap, ...]
    raw_initial_costs DOUBLE[],         -- Initial costs before optimization
    raw_generations INTEGER[],          -- Generations taken per run
    raw_stop_reasons VARCHAR[],         -- Stop reason per run ("completed", "optimal reached", etc.)
    
    -- ========================================================================
    -- Configuration: Algorithm Parameters (JSON)
    -- DuckDB JSON type (parsed, not VARCHAR)
    -- ========================================================================
    algorithm_config JSON NOT NULL,     -- {"population_size": 256, "mutation_rate": 0.02, ...}
    strategies JSON NOT NULL,           -- {"selection": "Tournament", "crossover": "Order", ...}
    construction_heuristic VARCHAR,     -- "random_initialization", "nearest_neighbor", etc.
    
    -- ========================================================================
    -- Computed Columns (VIRTUAL - computed on read)
    -- Note: DuckDB currently only supports VIRTUAL, not STORED
    -- ========================================================================
    is_test BOOLEAN GENERATED ALWAYS AS (repetitions <= 5) VIRTUAL
);

-- ============================================================================
-- INDEXES: Query Performance Optimization
-- ============================================================================
-- Strategy: Compound indexes for common query patterns
-- Note: DuckDB automatically optimizes index usage in WHERE/JOIN clauses
-- ============================================================================

-- Index 1: Most common query pattern - filter by problem AND algorithm
-- Usage: SELECT * FROM benchmark_runs WHERE problem_name = 'berlin52' AND algorithm = 'HybridOptimized';
CREATE INDEX IF NOT EXISTS idx_problem_algorithm 
ON benchmark_runs(problem_name, algorithm);

-- Index 2: Algorithm performance comparison across problems
-- Usage: SELECT * FROM benchmark_runs WHERE algorithm = 'FullGPU' ORDER BY mean_time;
CREATE INDEX IF NOT EXISTS idx_algorithm_time 
ON benchmark_runs(algorithm, mean_time);

-- Index 3: Backend comparison (CPU vs GPU)
-- Usage: SELECT * FROM benchmark_runs WHERE backend = 'GPU';
CREATE INDEX IF NOT EXISTS idx_backend 
ON benchmark_runs(backend);

-- Index 4: Time-series analysis (chronological benchmarks)
-- Usage: SELECT * FROM benchmark_runs ORDER BY timestamp DESC LIMIT 10;
CREATE INDEX IF NOT EXISTS idx_timestamp 
ON benchmark_runs(timestamp DESC);

-- Index 5: Filter by problem scale/size
-- Usage: SELECT * FROM benchmark_runs WHERE problem_size <= 100;
CREATE INDEX IF NOT EXISTS idx_problem_size 
ON benchmark_runs(problem_size);

-- Index 6: Separate test runs from production runs
-- Usage: SELECT * FROM benchmark_runs WHERE is_test = FALSE;
-- Note: Index on generated column works in DuckDB
CREATE INDEX IF NOT EXISTS idx_is_test 
ON benchmark_runs(is_test);

-- ============================================================================
-- EXAMPLE QUERIES
-- ============================================================================

-- Example 1: Get all algorithms tested on a specific problem
-- SELECT algorithm, mean_time, best_cost, optimal_cost
-- FROM benchmark_runs
-- WHERE problem_name = 'berlin52';

-- Example 2: Get unique problems with their metadata
-- SELECT DISTINCT problem_name, problem_size, optimal_cost
-- FROM benchmark_runs
-- ORDER BY problem_size;

-- Example 3: Get problems with CPU data (for CPU vs GPU comparison)
-- SELECT DISTINCT problem_name, problem_size
-- FROM benchmark_runs 
-- WHERE algorithm = 'CPU' 
-- ORDER BY problem_size;

-- Example 4: Production runs only (filter out tests)
-- SELECT * 
-- FROM benchmark_runs 
-- WHERE is_test = FALSE 
-- ORDER BY timestamp DESC;

-- Example 5: Performance ranking across all problems
-- SELECT algorithm, AVG(mean_time) as avg_time, AVG(mean_gap) as avg_gap
-- FROM benchmark_runs
-- WHERE is_test = FALSE
-- GROUP BY algorithm
-- ORDER BY avg_time;

-- Example 6: Problem complexity analysis
-- SELECT problem_size, problem_name, MIN(mean_time) as fastest_time, AVG(mean_time) as avg_time
-- FROM benchmark_runs
-- WHERE is_test = FALSE
-- GROUP BY problem_size, problem_name
-- ORDER BY problem_size;

-- ============================================================================
-- MAINTENANCE NOTES
-- ============================================================================
-- 1. DuckDB auto-commits by default (no explicit COMMIT needed)
-- 2. Single-table design: All data from checkpoint JSONs (self-contained)
-- 3. Generated columns (is_test) computed on SELECT (minimal overhead)
-- 4. LIST types stored efficiently with compression
-- 5. JSON types parsed once and cached for queries
-- 6. Indexes updated automatically on INSERT/UPDATE
-- 7. Get unique problems: SELECT DISTINCT problem_name, problem_size, optimal_cost
-- ============================================================================
