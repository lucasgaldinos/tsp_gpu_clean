# Database Module Migration & CLI Enhancement

## Summary of Changes

### 1. File Location Update

**Before**: `results/` (root level)
**After**: `code/benchmarks/result_database/` (organized in benchmarks directory)

**Rationale**: Better organization alongside benchmark code

### 2. Import Script Enhanced with CLI Arguments

#### New Features

**Output Directory Specification**:
```bash
# Default: Create in current working directory
python import_checkpoints.py

# Custom output directory
python import_checkpoints.py --output-dir ./results/
python import_checkpoints.py -o /tmp/my_results/
```

**Future-Ready Input Specification** (commented out):
```python
# In parse_arguments():
# parser.add_argument('--input', '-i', required=True, 
#                     help='Input directory (bulk) or JSON file path')

# Helper function ready:
# def resolve_input_paths(input_arg: str) -> List[Path]:
#     """Resolve to checkpoint files from dir or single JSON"""
```

### 3. Path Resolution Updates

#### Project Root Detection
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# From: code/benchmarks/result_database/import_checkpoints.py
# To:   /home/lucas_galdino/chimera/gpu_accelerated/
```

#### Checkpoint Directories (Hardcoded for Current Workflow)
```python
CHECKPOINT_DIRS = [
    PROJECT_ROOT / 'documentation' / 'benchmark_results' / 'checkpoints',
    PROJECT_ROOT / 'code' / 'benchmarks' / 'results' / 'checkpoints',
    PROJECT_ROOT / 'results' / 'checkpoints'
]
```

#### Database Output Path
```python
# Before (hardcoded):
DB_PATH = Path('results/results.duckdb')

# After (CLI-configurable):
def resolve_output_path(output_dir: Optional[str]) -> Path:
    if output_dir is None:
        return Path.cwd()  # Current working directory
    else:
        path = Path(output_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

db_path = resolve_output_path(args.output_dir) / 'results.duckdb'
```

### 4. Import Path Fixes

#### db_operations.py
```python
# Schema loading already uses relative path
schema_path = Path(__file__).parent / 'schema.sql'
# ✅ Works correctly in new location
```

#### import_checkpoints.py
```python
# Before:
from results.db_operations import ...

# After (same directory import):
from db_operations import (
    create_database,
    insert_problem,
    insert_benchmark_run,
    validate_benchmark_data
)
```

### 5. Enhanced Output Messages

```python
print(f"📂 Output directory: {output_path}")
print(f"💾 Database path: {db_path}")
print(f"   Found {len(found)} files in {checkpoint_dir.relative_to(PROJECT_ROOT)}")
```

### 6. Function Signature Update

```python
# Before:
def import_checkpoints():
    """Main import function."""
    # Hardcoded DB_PATH

# After:
def import_checkpoints(output_dir: Optional[Path] = None):
    """
    Main import function.
    
    Args:
        output_dir: Directory for results.duckdb (default: CWD)
    """
    output_path = resolve_output_path(output_dir)
    db_path = output_path / 'results.duckdb'
```

## Usage Examples

### Example 1: Default Behavior (Current Workflow)
```bash
cd /home/lucas_galdino/chimera/gpu_accelerated
python code/benchmarks/result_database/import_checkpoints.py

# Creates: ./results.duckdb (in current working directory)
# Scans:   hardcoded CHECKPOINT_DIRS (3 directories)
```

### Example 2: Custom Output Directory
```bash
# Create database in specific directory
python code/benchmarks/result_database/import_checkpoints.py --output-dir ./my_results/

# Creates: ./my_results/results.duckdb
# Scans:   hardcoded CHECKPOINT_DIRS (3 directories)
```

### Example 3: Temporary Results
```bash
# For testing or one-off analysis
python code/benchmarks/result_database/import_checkpoints.py -o /tmp/test_results/

# Creates: /tmp/test_results/results.duckdb
# Deleted on system reboot
```

### Example 4: Future Input Specification (Commented Out)
```bash
# FUTURE: When input argument is enabled
# python import_checkpoints.py --input ./custom_checkpoints/ --output-dir ./results/
# python import_checkpoints.py -i single_file.json -o ./results/
```

## Migration Checklist

- [x] Move files to `code/benchmarks/result_database/`
- [x] Update import paths (relative imports)
- [x] Add CLI argument parsing (`argparse`)
- [x] Implement `--output-dir` / `-o` argument
- [x] Add `resolve_output_path()` function
- [x] Comment out future `--input` / `-i` argument (ready for activation)
- [x] Add `resolve_input_paths()` helper (commented out)
- [x] Update `PROJECT_ROOT` path calculation
- [x] Update `CHECKPOINT_DIRS` to use `PROJECT_ROOT`
- [x] Update `import_checkpoints()` signature
- [x] Update documentation (README.md, QUICKSTART.md)
- [x] Test CLI help message
- [x] Verify relative imports work

## Documentation Updates

### README.md
- Added location: `code/benchmarks/result_database/`
- Updated usage examples with CLI arguments
- Updated file paths in examples

### QUICKSTART.md
- Added file location section
- Updated Step 2 with CLI examples
- Updated file organization structure
- Added database location explanation

## Future Enhancements (Ready to Enable)

### Uncomment in `parse_arguments()`:
```python
parser.add_argument(
    '--input', '-i',
    type=str,
    required=True,
    help='Input directory (bulk) or JSON file path'
)
```

### Uncomment `resolve_input_paths()` function:
```python
def resolve_input_paths(input_arg: str) -> List[Path]:
    """Handle single file or directory scanning"""
    # Implementation ready - just uncomment
```

### Update `import_checkpoints()` to use input arg:
```python
def import_checkpoints(output_dir=None, input_paths=None):
    if input_paths:
        checkpoint_files = resolve_input_paths(input_paths)
    else:
        checkpoint_files = find_all_checkpoints()  # Hardcoded
```

## Testing

### Test 1: Help Message ✅
```bash
python code/benchmarks/result_database/import_checkpoints.py --help
# Output: Shows usage, arguments, and examples
```

### Test 2: Default Behavior (Pending)
```bash
python code/benchmarks/result_database/import_checkpoints.py
# Expected: Creates ./results.duckdb in current directory
```

### Test 3: Custom Output (Pending)
```bash
python code/benchmarks/result_database/import_checkpoints.py -o /tmp/test/
# Expected: Creates /tmp/test/results.duckdb
```

## Notes

1. **Hardcoded CHECKPOINT_DIRS**: Kept for current workflow as requested
2. **Input argument**: Commented out, ready to enable when needed
3. **Path resolution**: Uses `Path.cwd()` as default, not script location
4. **Directory creation**: `mkdir(parents=True, exist_ok=True)` ensures output dir exists
5. **Relative imports**: Uses `from db_operations import ...` (same directory)

## Backwards Compatibility

**Breaking changes**:
- File location changed (old: `results/`, new: `code/benchmarks/result_database/`)
- Import paths changed (requires updating any external scripts)

**Non-breaking changes**:
- CLI arguments are optional (default behavior preserved)
- Function signatures backward compatible (optional parameters)
- Database schema unchanged
