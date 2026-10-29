# Quick Start: Database Operations

## File Location

**Directory**: `src/benchmarking/database/`

Files:
- `schema.sql` - Database schema
- `schema.py` - Table name and standard query constants
- `db_operations.py` - Pure functions module
- `import_checkpoints.py` - Batch import script (with CLI support)
- `validate_consistency.py` - Database validation against JSON checkpoints
- `README.md` - Full documentation

## 1. Quick Setup (3 Steps)

### Step 1: Create Database (Optional - Import script does this automatically)

```bash
duckdb results.duckdb < src/benchmarking/database/schema.sql
```

### Step 2: Import Checkpoints

```bash
# Default: Creates results.duckdb in current working directory
python -m benchmarking.database.import_checkpoints --input ./results/checkpoints/

# Or specify output directory
python -m benchmarking.database.import_checkpoints --input ./results/checkpoints/ --output-dir ./results/
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

## 2. Type Safety Features

The module validates:
- No NaN values in DOUBLE fields
- Array lengths match repetitions: `len(raw_times) == repetitions`
- Valid JSON structure for configs
- ISO 8601 timestamp format

**Example**:
```python
data = {
  "repetitions": 15,
  "raw_times": [1.2, 1.3, 1.1, ...],  # Must have exactly 15 values
  "mean_time": float("nan"),  # Will raise ValueError
}

validate_benchmark_data(data)  # Raises: "NaN value not allowed in field: mean_time"
```

## 3. Common Use Cases

### Import new benchmarks
```python
from benchmarking.database.db_operations import create_database, insert_benchmark_run
import json

conn = create_database("results/results.duckdb")

# Load checkpoint
with open("checkpoints/berlin52_CPU.json") as f:
  data = json.load(f)

# Insert with validation
run_id = insert_benchmark_run(conn, data)
print(f"Inserted run_id: {run_id}")

conn.close()
```

### Query for statistical analysis
```python
from benchmarking.database.db_operations import create_database, get_problems_with_cpu

conn = create_database("results/results.duckdb")

# Get problems with CPU data (for CPU vs GPU comparison)
cpu_problems = get_problems_with_cpu(conn)
print(f"CPU data available for: {cpu_problems}")

conn.close()
```

### Speedup analysis
```python
from benchmarking.database.db_operations import create_database, compute_speedup_matrix

conn = create_database("results/results.duckdb")

# Compute speedup of GPU algorithms vs CPU
speedups = compute_speedup_matrix(conn, "CPU", ["HybridOptimized", "FullGPU"])

for problem, algo, speedup, baseline_time in speedups:
  print(f"{problem}: {algo} is {speedup:.2f}x faster than CPU")

conn.close()
```

