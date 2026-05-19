"""
Standalone statistical recomputation from results_v2 checkpoints.

Reads existing checkpoints from results_v2/checkpoints/ — no algorithm reruns.
Outputs all artifacts to tmp/stats_output/.

Usage (from workspace root):
    uv run python tmp/recompute_stats.py

Outputs:
    tmp/stats_output/problem_statistics/<problem>_stats.json  — per-problem tests
    tmp/stats_output/cross_problem_stats.json                 — stratified aggregate
    tmp/stats_output/tables/chapter4_validation.md            — Markdown table
    tmp/stats_output/tables/chapter4_validation.tex           — LaTeX table
"""

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add workspace root so src.* imports resolve
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent  # tmp/../ = workspace root
CODE_ROOT = WORKSPACE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

# ---------------------------------------------------------------------------
# Imports from existing library (no modifications to those files)
# ---------------------------------------------------------------------------
from code.src.benchmarking_v2.config_loader import load_all_configs, ProblemConfig
from code.src.benchmarking_v2.checkpoint_io import CheckpointManager, load_checkpoint
from code.src.benchmarking_v2.statistics import StatisticalAnalyzer
from code.src.benchmarking_v2.problem_statistics import perform_statistical_analysis
from code.src.benchmarking_v2.aggregate_statistics import (
    perform_cross_problem_analysis_stratified,
)
from code.src.benchmarking_v2.report_generator import generate_result_tables

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = WORKSPACE_ROOT / "results_v2" / "checkpoints"
CONFIG_DIR = CODE_ROOT / "benchmarks_v2" / "configs"

OUTPUT_ROOT = WORKSPACE_ROOT / "tmp" / "stats_output"
PROBLEM_STATS_DIR = OUTPUT_ROOT / "problem_statistics"
TABLES_DIR = OUTPUT_ROOT / "tables"
CROSS_PROBLEM_JSON = OUTPUT_ROOT / "cross_problem_stats.json"

# All four algorithm names used in checkpoints
ALL_ALGORITHMS = ["CPU", "HybridNaive", "HybridOptimized", "FullGPU"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROBLEM_STATS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _load_problem_results(
    problem_name: str,
) -> dict:
    """Load all available algorithm checkpoints for one problem."""
    results = {}
    for algo in ALL_ALGORITHMS:
        path = CHECKPOINT_DIR / f"{problem_name}_{algo}.json"
        if path.exists():
            results[algo] = load_checkpoint(path)
            log.debug("  Loaded %s / %s", problem_name, algo)
        else:
            log.debug("  Missing checkpoint: %s — skipping", path.name)
    return results


def _save_problem_stats(problem_name: str, stats: dict) -> None:
    out_path = PROBLEM_STATS_DIR / f"{problem_name}_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("  Saved problem stats → %s", out_path.relative_to(WORKSPACE_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _make_dirs()

    # 1. Load configs (problem list, algorithm definitions, benchmark params)
    log.info("Loading configs from %s", CONFIG_DIR)
    configs = load_all_configs(CONFIG_DIR)
    problem_configs: list[ProblemConfig] = configs["problems"]
    benchmark_config = configs["benchmark"]
    cpu_threshold: int = benchmark_config.cpu_size_threshold

    log.info(
        "Config loaded: %d problems, cpu_threshold=%d",
        len(problem_configs),
        cpu_threshold,
    )

    # 2. Instantiate the statistical analyzer
    analyzer = StatisticalAnalyzer()

    # 3. Loop over problems — load checkpoints + per-problem stats
    all_results: dict[str, dict[str, dict]] = {}

    log.info("=" * 70)
    log.info("PER-PROBLEM ANALYSIS  (reading from %s)", CHECKPOINT_DIR)
    log.info("=" * 70)

    for prob_config in problem_configs:
        pname = prob_config.name
        optimal = prob_config.optimal
        log.info("\n--- %s  (n=%d, optimal=%.0f) ---", pname, prob_config.size, optimal)

        problem_results = _load_problem_results(pname)

        if not problem_results:
            log.warning("  No checkpoints found for %s — skipping.", pname)
            continue

        log.info(
            "  Algorithms present: %s", ", ".join(problem_results.keys())
        )

        all_results[pname] = problem_results

        # Per-problem statistical tests (Friedman, Wilcoxon, Holm-Bonferroni)
        stats = perform_statistical_analysis(
            problem_name=pname,
            optimal_cost=optimal,
            results=problem_results,
            analyzer=analyzer,
        )
        _save_problem_stats(pname, stats)

    # 4. Cross-problem stratified analysis
    log.info("\n" + "=" * 70)
    log.info("CROSS-PROBLEM STRATIFIED ANALYSIS")
    log.info("=" * 70)

    cross_stats = perform_cross_problem_analysis_stratified(
        all_results=all_results,
        problem_configs=problem_configs,
        analyzer=analyzer,
        cpu_threshold=cpu_threshold,
    )

    with open(CROSS_PROBLEM_JSON, "w", encoding="utf-8") as f:
        json.dump(cross_stats, f, indent=2, default=str)
    log.info("Cross-problem stats saved → %s", CROSS_PROBLEM_JSON.relative_to(WORKSPACE_ROOT))

    # 5. Generate result tables (Markdown + LaTeX)
    log.info("\n" + "=" * 70)
    log.info("GENERATING TABLES → %s", TABLES_DIR)
    log.info("=" * 70)

    generate_result_tables(
        all_results=all_results,
        problem_configs=problem_configs,
        output_dir=TABLES_DIR,
        benchmark_name="chapter4_validation",
    )

    log.info("\n" + "=" * 70)
    log.info("DONE — outputs written to tmp/stats_output/")
    log.info("=" * 70)
    log.info("  Problem stats : %s/", PROBLEM_STATS_DIR.relative_to(WORKSPACE_ROOT))
    log.info("  Cross-problem : %s", CROSS_PROBLEM_JSON.relative_to(WORKSPACE_ROOT))
    log.info("  Tables        : %s/", TABLES_DIR.relative_to(WORKSPACE_ROOT))


if __name__ == "__main__":
    main()
