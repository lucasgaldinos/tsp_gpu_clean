# Quick Start: Database Operations

## File Location

**Directory**: `code/benchmarks/result_database/`

Files:
- `schema.sql` - Database schema
- `db_operations.py` - Pure functions module
- `import_checkpoints.py` - Batch import script (with CLI support)
- `README.md` - Full documentation

## 1. Understanding Table 1 (problems)

**Purpose**: Reference table for JOIN operations

**Why needed**: Prevents data duplication
- Without: `optimal_cost` stored 130+ times in benchmark_runs
- With: `optimal_cost` stored once in problems table

**Usage pattern**:
```sql
-- JOIN to get problem metadata
SELECT br.*, p.optimal_cost, p.size
FROM benchmark_runs br
JOIN problems p ON br.problem_name = p.name;
```

## 2. Quick Setup (3 Steps)

### Step 1: Create Database (Optional - Import script does this automatically)

```bash
cd /home/lucas_galdino/chimera/gpu_accelerated
duckdb results.duckdb < code/benchmarks/result_database/schema.sql
```

### Step 2: Import Checkpoints

```bash
# Default: Creates results.duckdb in current working directory
python code/benchmarks/result_database/import_checkpoints.py

# Or specify output directory
python code/benchmarks/result_database/import_checkpoints.py --output-dir ./results/
python code/benchmarks/result_database/import_checkpoints.py -o /tmp/my_results/
```

### Step 3: Test Query

```bash
duckdb results.duckdb
```

```sql
-- Count benchmark runs
SELECT COUNT(*) FROM benchmark_runs;

-- Get CPU problems
SELECT DISTINCT problem_name FROM benchmark_runs WHERE algorithm = 'CPU';
```

## 3. Type Safety Features

The module validates:
- ✅ No NaN values in DOUBLE fields
- ✅ Array lengths match repetitions: `len(raw_times) == repetitions`
- ✅ Valid JSON structure for configs
- ✅ ISO 8601 timestamp format
- ✅ Foreign keys exist (problem_name in problems table)

**Example**:
```python
data = {
    'repetitions': 15,
    'raw_times': [1.2, 1.3, 1.1, ...],  # Must have exactly 15 values
    'mean_time': float('nan')  # ❌ Will raise ValueError
}

validate_benchmark_data(data)  # Raises: "NaN value not allowed in field: mean_time"
```

## 4. Common Use Cases

### Import new benchmarks
```python
from results.db_operations import create_database, insert_benchmark_run
import json

conn = create_database('results/results.duckdb')

# Load checkpoint
with open('checkpoints/berlin52_CPU.json') as f:
    data = json.load(f)

# Insert with validation
run_id = insert_benchmark_run(conn, data)
print(f"Inserted run_id: {run_id}")

conn.close()
```

### Query for statistical analysis
```python
from results.db_operations import create_database, get_problems_with_cpu

conn = create_database('results/results.duckdb')

# Get problems with CPU data (for CPU vs GPU comparison)
cpu_problems = get_problems_with_cpu(conn)
print(f"CPU data available for: {cpu_problems}")

conn.close()
```

### Speedup analysis
```python
from results.db_operations import create_database, compute_speedup_matrix

conn = create_database('results/results.duckdb')

# Compute speedup of GPU algorithms vs CPU
speedups = compute_speedup_matrix(conn, 'CPU', ['HybridOptimized', 'FullGPU'])

for problem, algo, speedup, baseline_time in speedups:
    print(f"{problem}: {algo} is {speedup:.2f}x faster than CPU")

conn.close()
```

## 5. DuckDB-Specific Syntax

### Arrays (LIST type)
```sql
-- Create table with array
CREATE TABLE test (times DOUBLE[]);

-- Insert array
INSERT INTO test VALUES ([1.2, 1.3, 1.4]);

-- Access element (1-based indexing)
SELECT times[1] FROM test;  -- Returns 1.2

-- Expand array to rows
SELECT UNNEST(times) FROM test;
```

### JSON columns
```sql
-- Cast string to JSON
INSERT INTO benchmark_runs (algorithm_config, ...) 
VALUES ('{"population_size": 256}'::JSON, ...);

-- Extract JSON field
SELECT algorithm_config->>'$.population_size' FROM benchmark_runs;
```

### Generated columns
```sql
-- Computed column (VIRTUAL only)
is_test BOOLEAN GENERATED ALWAYS AS (repetitions <= 5) VIRTUAL
```

## 6. File Organization

```
code/benchmarks/result_database/
├── schema.sql              # Database schema (tables + indexes)
├── db_operations.py        # Pure functions module (~200 LOC)
├── import_checkpoints.py   # Batch import script (with CLI)
├── README.md               # Full documentation
└── QUICKSTART.md           # This file
```

**Database location**: Created in output directory (default: current working directory)
- Default: `./results.duckdb`
- Custom: Specified by `--output-dir` argument

## 7. Integration with Validation (Option C)

User chose to remove `session_id`, can add `run_id` to validation output later:

```python
# In chapter4_validation.py
from results.db_operations import create_database, insert_benchmark_run

# After benchmark completion
checkpoint_data = save_checkpoint(...)
conn = create_database('results/results.duckdb')
run_id = insert_benchmark_run(conn, checkpoint_data)

# Add to validation output
validation_output = {
    'run_id': run_id,  # Database reference for this benchmark
    'database_path': 'results/results.duckdb',
    # ... other validation fields
}
```

## 8. Solution 1 Benefits

**YAGNI-compliant** (~200 LOC):
- No unnecessary classes or abstractions
- Simple function signatures
- Easy to understand and maintain

**Type-safe**:
- Comprehensive validation before insertion
- Clear error messages
- Prevents data corruption

**DuckDB-native**:
- Uses LIST type for arrays
- Uses JSON type for configs
- VIRTUAL generated columns

**Flexible**:
- If grows beyond 300 LOC → refactor to Solution 3 (modular)
- Pure functions easy to test and compose

## 9. Next Steps

1. Run import script to populate database
2. Test queries with your data
3. Integrate with chapter4_validation.py
4. Update statistical_tests_comprehensive_guide.md with database queries
