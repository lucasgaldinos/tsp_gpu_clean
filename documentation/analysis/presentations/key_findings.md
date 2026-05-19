# key findings

## Key Findings Summary

### CRITICAL Issues (4)

1. **Per-instance data fabricated or from unknown source.** The berlin52/kroA100/pr1002 tables in both the presentation and draft do not match V1, V2, or V3 checkpoint data. HybridNaive times are ~3× overstated, CPU times are ~30% off, and speedup values are consequently wrong.

2. **pr1002 gap values are ~4× understated** — presentation claims 1.15%-2.10% gaps, actual V2 data shows 4.14%-8.04%.

3. **"30 repetitions" is stated but all V2 checkpoints have 26.** Either the runs are incomplete or the text needs updating.

4. **"FullGPU finds optimal 100% of the time on small instances"** — actual mean gap is 0.23%, not 0.00%.

### IMPORTANT Issues (3)

1. **Shapiro-Wilk** is not from Demšar (0 mentions in the paper). Cite `@shapiro1965analysis`.
2. **Cohen's d** is not from Demšar. Cite `@cohen1988statistical`.
3. **Category gap values** consistently understated (e.g., large: 1.85% claimed vs 2.83% actual).

### Verified Correct

- **Aggregated results** (mean times + mean gaps across 38 instances) match V2 data exactly
- **Hardware specs** (640 CUDA cores, 112 GB/s, PCIe ~16 GB/s) confirmed via Wikipedia
- **Fujimoto 2011** claims (24.2×, GTX 285, OX+2-opt) confirmed from paper
- **Crainic taxonomy** (Type 1/2/3) confirmed; all 4 variants correctly classified as Type 1
- **Demšar**: Friedman, Nemenyi, Holm all confirmed in the paper
- **Larrañaga**: OX and swap mutation confirmed (but NOT tournament selection)

## plan

I saved the handoff plan in session memory and grounded it with discovery across the benchmark pipeline, the diagram sources, and the presentation. The plan now assumes your two scope decisions: use only results_v2, and persist one detailed research report per subagent under tmp during execution.

1. First, rerun up to 4 quick-research Explore subagents and save one detailed report per track under tmp: results_v2 data, benchmark code paths, diagram inventory, and presentation insertion points.
2. Before any reevaluation, audit and fix the output-path routing so regeneration stays inside results_v2. This is necessary because run_chapter4_benchmark.py still advertises legacy output folders, checkpoint_io.py defaults to results_v3, and orchestration.py instantiates that default manager and writes tables to results_v3.
3. Recompute statistics, tables, and any benchmark-backed figures from the existing checkpoints only. The executor should treat those JSON checkpoints as the per-run source of truth and regenerate problem_statistics and tables from them.
4. Rewrite fact_check_report.md as a results_v2-only report. It should explicitly exclude V1, V3, and mixed comparisons, preserve valid source-paper checks, and rebuild every benchmark-backed claim from results_v2.
5. Repair the quantitative parts of thesis_presentation.md so the slides no longer rely on the invalid prior analysis. That includes wrong per-instance values, wrong repetition counts, and any category summaries that do not trace cleanly to results_v2.
6. Create the missing Mermaid source set under diagrams: one editable class diagram for the GA family, one improved shared GA-loop flowchart based on the notes, one separated execution diagram for each variant, and separate CPU/GPU architecture diagrams.
7. Add machine/state diagrams for the GPU variants that explicitly show where parallelization starts and finishes, especially around the 2-opt and fitness stages.
8. Insert the requested explanatory content into thesis_presentation.md: heuristics vs metaheuristics in the GA + 2-opt theory section, CPU vs GPU plus MIMD/SIMD/SIMT before the Crainic taxonomy, OX rationale near the parameter discussion, and a direct explanation of where 2-opt sits in the GA cycle and how each GPU variant parallelizes it.
9. Validate every Mermaid diagram after creation and preview it before integration. The execution phase should use the Mermaid workflow for flowchart, classDiagram, and stateDiagram sources.
10. Finish with traceability verification: every retained numeric claim in the fact check and presentation must map back to results_v2 only, with spot checks on representative checkpoints such as berlin52, kroA100, and pr1002.

**Key placement decisions**

- The CPU vs GPU architecture slide should be inserted before the Crainic taxonomy, because the current presentation uses SIMD, SIMT, and MIMD before defining them.
- The heuristics vs metaheuristics explanation belongs inside the GA + 2-opt theory area, not as a detached slide later in the presentation.
- The crossover note should explain why OX is used for TSP without claiming it is universally best.
- The Mermaid sources should live in diagrams, while the Beamer presentation should likely consume rendered assets if direct Mermaid rendering is unreliable.

**Important discovery**

The main execution risk is path drift into results_v3. Until that is fixed or explicitly overridden, any “recompute from existing checkpoints” step can silently regenerate artifacts into the wrong folder and recreate the same confusion you want to eliminate.

## Adjustments

1. You won't be rerunning the checkpoints. DO NOT CHANGE. You shall rerun the benchmarks only. These benchmarks should be at the #file:tmp folder.
2. This means you should start at step 3. I already have the results. Recompute the statistics.

The discovery is something I already knew. You won't be running the program, just the benchmarks. The benchmarks have output paths. Please do the research properly.

## Plan: Recompute Statistics from results_v2 Checkpoints → tmp

Write a standalone python script `tmp/recompute_stats.py` that directly calls the existing `benchmarking_v2` library functions, loading checkpoints from checkpoints and writing all outputs to `tmp/stats_output/`. No changes to any existing code.

**Steps**

**Phase 1 — Build the standalone recomputation script**

1. Create `tmp/recompute_stats.py` as a self-contained CLI script that:
   + Adds workspace root to `sys.path` so `from src.benchmarking_v2.*` imports work
   + Loads problem/algorithm configs via `load_all_configs(config_dir)` from configs
   + Instantiates `CheckpointManager(checkpoint_dir="results_v2/checkpoints", problem_stats_dir="tmp/stats_output/problem_statistics")` — explicit path override, no code changes
   + For each problem in `problem_configs`, loads existing checkpoint JSONs: `{problem}_{algo}.json` from checkpoints (algorithms: `CPU`, `HybridNaive`, `HybridOptimized`, `FullGPU`)
   + Builds `problem_results: Dict[str, Dict[str, Any]]` — only adds an algorithm entry if its checkpoint file actually exists (handles large problems without CPU)
   + Calls `perform_statistical_analysis(problem_name, optimal_cost, results, analyzer)` per problem; saves output JSON to `tmp/stats_output/problem_statistics/`
   + Calls `perform_cross_problem_analysis_stratified(all_results, problem_configs, analyzer, cpu_threshold=100)` once after all problems
   + Calls `generate_result_tables(all_results, problem_configs, output_dir="tmp/stats_output/tables/")` to produce `.md` and `.tex` tables
   + Saves the full cross-problem stats dict to `tmp/stats_output/cross_problem_stats.json`
   + Runs with: `uv run python tmp/recompute_stats.py`

**Phase 2 — Validate outputs**

1. Spot-check `tmp/stats_output/tables/chapter4_validation.md` against known numbers from master_results_v2.csv (berlin52 CPU=25.82s, gap=0.17%; kroA100 HybridNaive=1.81s; pr1002 FullGPU gap=4.14%)
2. Confirm `tmp/stats_output/problem_statistics/berlin52_stats.json` contains Friedman result, pairwise Wilcoxon + Holm-Bonferroni for the 4-algorithm case
3. Confirm cross-problem stratum coverage: Stratum 1 = problems with n≤100 (12 problems, 4 algos); Stratum 2 = all 38 problems (GPU-only)

**Phase 3 — Fact check rewrite (depends on Phase 1)**

1. Rewrite fact_check_report.md using only outputs from `tmp/stats_output/`; fix the 8 known discrepancies (berlin52 naive time, kroA100 naive time, pr1002 GPU gap, category gaps, "30 repetitions" → 26, speedup 5.54x → verify against tables)

**Phase 4 — Presentation fixes (depends on Phase 3)**

1. Fix quantitative slides in thesis_presentation.md using confirmed `tmp/stats_output` numbers

---

**Relevant files**

- `results_v2/checkpoints/*.json` — 126 input files; `raw_costs`, `raw_gaps`, `raw_times` present ✓
- problem_statistics.py — `perform_statistical_analysis()`, `_get_data_field()` handles `raw_*` prefix
- aggregate_statistics.py — `perform_cross_problem_analysis_stratified()`
- report_generator.py — `generate_result_tables()`, writes `{output_dir}/chapter4_validation.md` and `.tex`
- config_loader.py — `load_all_configs()`, `ProblemConfig(name, optimal, size)`
- statistics.py — `StatisticalAnalyzer()` (needs instantiation)
- checkpoint_io.py — `CheckpointManager(checkpoint_dir, problem_stats_dir)`, `load_checkpoint(path)`
- configs — `problems.json`, `algorithms.json`, `benchmark.json`
- tmp — empty, target for all outputs

**Verification**

1. `uv run python tmp/recompute_stats.py` completes without errors
2. `tmp/stats_output/tables/chapter4_validation.md` exists and berlin52 CPU row shows ~25.82s
3. `tmp/stats_output/cross_problem_stats.json` shows stratum1_count=12, stratum2_count=38
4. HybridOptimized speedup in tables matches `speedup_summary_v2.csv` median of ~5.7×

**Decisions**

- **DO NOT** modify any existing code (orchestration.py, checkpoint_io.py, `run_chapter4_benchmark.py`)
- `tmp/recompute_stats.py` is a standalone script; it reuses library functions by import, not subprocess calls
- Outputs go only to `tmp/stats_output/` — results_v2 is read-only, results_v3 is untouched
- `_save_problem_statistics_file()` in orchestration is **not reused** (it's tightly coupled to the run loop); the script saves per-problem stats directly via `json.dump`

**Further Considerations**

1. `StatisticalAnalyzer` import path: needs verification — likely `from src.benchmarking_v2.statistics import StatisticalAnalyzer`. Confirm by checking statistics.py before writing.
2. Cross-problem analysis output: `perform_cross_problem_analysis_stratified()` returns a dict but doesn't write files — the script needs to serialize it. Should it be split into two JSON files (stratum1, stratum2) or one combined file? Recommend one `cross_problem_stats.json`.

---

Ready to implement. Shall I proceed with writing `tmp/recompute_stats.py`?
