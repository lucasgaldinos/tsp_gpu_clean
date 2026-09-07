# Database Operations for Benchmark Results

## Overview

This directory contains the complete database solution for storing and querying GPU-accelerated routing optimization benchmark results.

**Location**: `code/benchmarks/result_database/`

**Architecture**: Single-table design - All data from checkpoint JSONs (self-contained)

## Single-Table Design

The database uses a **single `benchmark_runs` table** that stores all benchmark data directly from checkpoint JSON files. No separate problems table or JOIN operations required.

### Why Single-Table?

**Checkpoint JSONs are self-contained**:
```json
{
  "problem_name": "kroA100",
  "problem_size": 100,
  "optimal_cost": 21282,
  "algorithm": "HybridNaive",
  "mean_time": 2.19,
  ...
}
```

Each checkpoint already includes problem metadata, so storing it separately would be redundant normalization without benefit.

### Design Benefits

1. **Simplicity**: Direct mapping from JSON to database row
2. **No JOINs needed**: All data in one place
3. **Self-contained**: Each row has complete information
4. **Fast queries**: No foreign key lookups

### Getting Unique Problems

```sql
-- Get all unique problems with their metadata
SELECT DISTINCT problem_name, problem_size, optimal_cost
FROM benchmark_runs
ORDER BY problem_size;

-- Count unique problems
SELECT COUNT(DISTINCT problem_name) FROM benchmark_runs;
```

## Files

### 1. `schema.sql` - Database Schema

Complete DuckDB-specific SQL for creating the benchmark_runs table and indexes.

**Key features**:
- **LIST types** (`DOUBLE[]`, `INTEGER[]`, `VARCHAR[]`) for variable-length arrays
- **JSON types** for algorithm configurations
- **Generated columns** (VIRTUAL) for computed fields like `is_test`
- **6 optimized indexes** for common query patterns

**Execute schema**:
```bash
# Create database and apply schema
duckdb results.duckdb < schema.sql
```

### 2. `db_operations.py` - Pure Functions Module

Type-safe database operations with comprehensive validation.

**Functions**:
- `create_database(db_path)` - Initialize database with schema
- `validate_benchmark_data(data)` - Type safety checks (NaN, NULL, array lengths)
- `insert_benchmark_run(conn, data)` - Insert with validation
- `get_problems_with_cpu(conn)` - Query utility for CPU data
- `get_algorithm_summary(conn, problem, algorithm)` - Get summary statistics
- `compute_speedup_matrix(conn, baseline, comparisons)` - Speedup analysis

**Type safety features**:
- ✅ NaN detection in DOUBLE fields
- ✅ NULL validation for required fields
- ✅ Array length consistency (`len(raw_times) == repetitions`)
- ✅ JSON structure validation
- ✅ Timestamp format validation
- ✅ Foreign key existence checks

**Example usage**:
```python
import duckdb
from results.db_operations import create_database, insert_benchmark_run

# Create database
conn = create_database('results/results.duckdb')

# Insert benchmark data
data = {
    'problem_name': 'berlin52',
    'algorithm': 'HybridOptimized',
    'repetitions': 15,
    'mean_time': 1.234,
    'raw_times': [1.2, 1.3, 1.1, ...],
    # ... other fields
}

try:
    run_id = insert_benchmark_run(conn, data)
    print(f"✓ Inserted run_id: {run_id}")
except ValueError as e:
    print(f"✗ Validation error: {e}")

conn.close()
```

### 3. `import_checkpoints.py` - Batch Import Script

Imports all checkpoint JSON files into the database.

**Features**:
- ✅ Scans multiple checkpoint directories
- ✅ Handles old (2 reps) and new (15 reps) formats
- ✅ Validates data before insertion
- ✅ Skips duplicates based on (problem_name, algorithm, timestamp)
- ✅ Reports progress and statistics

**Usage**:
```bash
# Default: Create database in current working directory
cd /home/lucas_galdino/chimera/gpu_accelerated
python code/benchmarks/result_database/import_checkpoints.py

# Specify output directory
python code/benchmarks/result_database/import_checkpoints.py --output-dir ./results/
python code/benchmarks/result_database/import_checkpoints.py -o /tmp/benchmark_results/
```

**Expected output**:
```
================================================================================
Importing Checkpoint JSONs to DuckDB Database
================================================================================

📁 Scanning checkpoint directories...
   Found 130 checkpoint files

🗄️  Creating/connecting to database...
   Database: results/results.duckdb

📊 Extracting problem metadata...
   Found 37 unique problems

💾 Inserting problem metadata...
   ✓ berlin52 (n=52)
   ✓ st70 (n=70)
   ✓ ts225 (n=225)
   ...

🚀 Importing benchmark runs...
   ✓ berlin52      CPU              (15 reps, prod) -> run_id=1
   ✓ berlin52      HybridOptimized  (15 reps, prod) -> run_id=2
   ...

================================================================================
Import Summary
================================================================================
Total files:          130
✓ Imported:           128
⊘ Duplicates:         0
✗ Validation errors:  2
✗ Other errors:       0

📊 Database Statistics
================================================================================
Total benchmark runs: 128

By Algorithm:
  HybridOptimized     :  37 runs
  HybridNaive         :  37 runs
  FullGPU             :  37 runs
  CPU                 :  12 runs

By Type:
  Production runs     : 123 runs
  Test runs           :   5 runs

Problems with CPU data: 12

✅ Import completed successfully!
```

## Schema Details

### Table 1: `problems`

| Column | Type | Description |
|--------|------|-------------|
| `name` | VARCHAR (PK) | Problem name (e.g., "berlin52") |
| `size` | INTEGER | Number of cities |
| `optimal_cost` | DOUBLE | Known optimal tour cost |
| `type` | VARCHAR | Problem type (TSP, ATSP, CVRP) |

**Row count**: ~37 problems

### Table 2: `benchmark_runs`

Complete schema matching checkpoint JSON structure. See `schema.sql` for full details.

**Key columns**:
- **Identification**: `run_id` (PK), `problem_name` (FK), `algorithm`, `backend`
- **Metadata**: `repetitions`, `timestamp`, `max_generations_configured`
- **Aggregates**: `mean_time`, `std_time`, `mean_cost`, `mean_gap`, etc.
- **Raw arrays**: `raw_times[]`, `raw_costs[]`, `raw_gaps[]`, `raw_generations[]`, etc.
- **Configs**: `algorithm_config` (JSON), `strategies` (JSON)
- **Computed**: `is_test` (VIRTUAL, `repetitions <= 5`)

**Row count**: ~130 benchmark runs (48 small + 75 large + 5 tests)

## Common Queries

### Get all algorithms for a problem
```sql
SELECT br.algorithm, br.mean_time, br.best_cost, p.optimal_cost
FROM benchmark_runs br
JOIN problems p ON br.problem_name = p.name
WHERE br.problem_name = 'berlin52'
ORDER BY br.mean_time;
```

### Get problems with CPU data (for comparison)
```sql
SELECT DISTINCT problem_name, problem_size
FROM benchmark_runs
WHERE algorithm = 'CPU'
ORDER BY problem_size;
```

### Production runs only (filter out tests)
```sql
SELECT *
FROM benchmark_runs
WHERE is_test = FALSE
ORDER BY timestamp DESC;
```

### Performance ranking across all problems
```sql
SELECT algorithm, 
       AVG(mean_time) as avg_time, 
       AVG(mean_gap) as avg_gap,
       COUNT(*) as num_problems
FROM benchmark_runs
WHERE is_test = FALSE
GROUP BY algorithm
ORDER BY avg_time;
```

### Speedup analysis: GPU vs CPU
```sql
WITH cpu_times AS (
    SELECT problem_name, mean_time as cpu_time
    FROM benchmark_runs
    WHERE algorithm = 'CPU' AND is_test = FALSE
),
gpu_times AS (
    SELECT problem_name, algorithm, mean_time as gpu_time
    FROM benchmark_runs
    WHERE algorithm != 'CPU' AND is_test = FALSE
)
SELECT 
    g.problem_name,
    g.algorithm,
    c.cpu_time,
    g.gpu_time,
    c.cpu_time / g.gpu_time as speedup
FROM gpu_times g
JOIN cpu_times c ON g.problem_name = c.problem_name
ORDER BY speedup DESC;
```

### Extract raw data for statistical analysis
```sql
SELECT 
    problem_name,
    algorithm,
    UNNEST(raw_times) as time,
    UNNEST(raw_costs) as cost,
    UNNEST(raw_gaps) as gap
FROM benchmark_runs
WHERE problem_name = 'berlin52' AND algorithm = 'CPU';
```

## DuckDB-Specific Features Used

### 1. LIST Type (Variable-Length Arrays)
```sql
raw_times DOUBLE[]  -- LIST type, not ARRAY (fixed-length)
```
**Why**: Checkpoint files have variable repetitions (2 vs 15), so variable-length LIST is needed.

### 2. JSON Type (Native Parsing)
```sql
algorithm_config JSON  -- Parsed JSON, not VARCHAR
```
**Access**:
```sql
SELECT algorithm_config->>'$.population_size' as pop_size
FROM benchmark_runs;
```

### 3. Generated Columns (VIRTUAL)
```sql
is_test BOOLEAN GENERATED ALWAYS AS (repetitions <= 5) VIRTUAL
```
**Note**: DuckDB currently only supports VIRTUAL (computed on read), not STORED.

### 4. UNNEST for Array Expansion
```sql
SELECT UNNEST(raw_times) FROM benchmark_runs;
```
Expands array into rows for analysis.

## Type Safety Validation

The `validate_benchmark_data()` function performs comprehensive checks:

1. **Required fields**: All fields marked NOT NULL must be present
2. **NaN detection**: `math.isnan()` checks for DOUBLE columns
3. **Infinite values**: Rejects `math.isinf()` values
4. **Array consistency**: `len(raw_times) == repetitions`
5. **JSON structure**: Validates parseable JSON
6. **Timestamp format**: ISO 8601 format validation

**Example validation error**:
```python
ValueError: raw_times length (14) != repetitions (15)
```

## Integration with Validation Output (Option C)

User chose **Option C**: Remove `session_id`, can add to validation output later.

**Future enhancement** (from `chapter4_validation.py`):
```python
# After benchmark completion
run_id = insert_benchmark_run(conn, checkpoint_data)

# Add to validation output
validation_results = {
    'problem_name': 'berlin52',
    'algorithm': 'HybridOptimized',
    'run_id': run_id,  # Database reference
    'database_path': 'results/results.duckdb',
    # ... other validation fields
}
```

## Testing

Run module self-test:
```bash
python results/db_operations.py
```

Expected output:
```
Database Operations Module - Quick Test
============================================================
✓ Created test problem
✓ Inserted test benchmark run (run_id: 1)
✓ Retrieved summary: mean_time=1.50s

✅ All tests passed! Module ready for use.
```

## Dependencies

```bash
pip install duckdb
```

DuckDB Python API documentation: https://duckdb.org/docs/api/python/overview

## Next Steps

1. ✅ Create database and schema (`schema.sql`)
2. ✅ Implement pure functions module (`db_operations.py`)
3. ✅ Create import script (`import_checkpoints.py`)
4. ⏳ Run import script to populate database
5. ⏳ Create query utility functions for statistical analysis
6. ⏳ Update `statistical_tests_comprehensive_guide.md` with database queries
7. ⏳ Integrate with `chapter4_validation.py` for automatic result storage

## Architecture Decision Record

**Chosen**: Solution 1 - Pure Functions Module

**Rationale**:
- **YAGNI-compliant**: ~200 LOC, no unnecessary abstractions
- **Type safe**: Comprehensive validation (NaN, NULL, arrays, JSON)
- **Simple**: Easy to understand and maintain
- **Flexible**: Can refactor to Solution 3 (modular) if needed

**Trade-offs**:
- ✅ Pros: Simple, fast, type-safe, easy to test
- ⚠️ Cons: Less abstraction than OOP repository pattern
- 💡 Future: If grows beyond 300 LOC, refactor to Solution 3 (modular structure)

## Authors

GPU Accelerated Routing Optimization Team  
TCC (Undergraduate Thesis) - 2025
