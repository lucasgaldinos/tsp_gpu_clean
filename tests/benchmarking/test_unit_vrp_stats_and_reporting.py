"""Unit tests for Deb (2000) feasibility-aware statistics and conditional VRP reporting (Task 17.12, DP17-4)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarking.analysis.cohort import BenchmarkCohort
from benchmarking.analysis.cohort_analysis import analyze_cohort
from benchmarking.analysis.problem_statistics import analyze_problem, build_feasibility_aware_scores
from benchmarking.analysis.statistics import StatisticalAnalyzer
from benchmarking.reporting.table_reports import generate_latex_tables, generate_markdown_tables
from benchmarking.runner.config_loader import ProblemConfig
from protocols.vrp_strategies import compute_normalized_violations

if TYPE_CHECKING:
  from pathlib import Path


def test_build_feasibility_aware_scores_preserves_tsp_and_enforces_deb_dominance() -> None:
  """WHAT: `build_feasibility_aware_scores` is identity when 100% feasible and applies Deb (2000) otherwise.

  WHY: Unconstrained TSP statistics must remain bit-identical, while constrained VRP samples
  must rank every feasible solution strictly ahead of any infeasible solution regardless of distance.
  """
  costs = np.array([100.0, 120.0, 90.0], dtype=np.float64)

  # 1. None or all-True -> exact unchanged float64 array
  np.testing.assert_array_equal(build_feasibility_aware_scores(costs, None, None), costs)
  np.testing.assert_array_equal(
    build_feasibility_aware_scores(costs, [0.0, 0.0, 0.0], [True, True, True]),
    costs,
  )

  # 2. Infeasible solution with smaller distance (90.0) strictly loses to feasible solutions (100.0, 120.0)
  scored = build_feasibility_aware_scores(costs, [0.0, 0.0, 5.0], [True, True, False])
  # C_max = 120.0 -> infeasible score = 120.0 + 1.0 + 5.0 = 126.0
  np.testing.assert_allclose(scored, [100.0, 120.0, 126.0])
  assert scored[1] < scored[2]

  # 3. 0%-feasible sample uses C_max = 0.0 and orders strictly by lower violation
  zero_feas = build_feasibility_aware_scores([50.0, 40.0], [10.0, 3.0], [False, False])
  np.testing.assert_allclose(zero_feas, [11.0, 4.0])
  assert zero_feas[1] < zero_feas[0]


def test_analyze_problem_deb_ranking_and_penalty_weight_invariance() -> None:
  """WHAT: Per-problem Friedman & Wilcoxon rank feasible ahead of shorter infeasible and ignore omega_Q.

  WHY: Changing penalty weight omega_Q (e.g., 100 vs 100_000) must not alter non-parametric
  rankings or p-values, and an infeasible solver with shorter raw travel distance must lose.
  """

  def _make_problem_results(omega_q: float) -> dict[str, dict[str, Any]]:
    # 6 paired seeds:
    # FeasGood: feasible, unpenalized ~ 200
    # FeasWorse: feasible, unpenalized ~ 220
    # InfeasShort: infeasible (viol=4..6), unpenalized ~ 150 (shorter than both feasible!)
    viols = [4.0, 5.0, 6.0, 4.5, 5.5, 6.5]
    infeas_unpen = [150.0, 151.0, 149.0, 152.0, 150.5, 148.5]
    infeas_pen = [c + omega_q * v for c, v in zip(infeas_unpen, viols, strict=True)]
    return {
      "FeasGood": {
        "problem_type": "CVRP",
        "mean_cost": 200.0,
        "mean_time": 1.0,
        "raw_costs": [198.0, 200.0, 202.0, 199.0, 201.0, 200.0],
        "raw_unpenalized_costs": [198.0, 200.0, 202.0, 199.0, 201.0, 200.0],
        "raw_violations": [0.0] * 6,
        "raw_is_feasible": [True] * 6,
        "raw_times": [1.0] * 6,
        "feasible_runs": 6,
        "feasibility_rate": 100.0,
      },
      "FeasWorse": {
        "problem_type": "CVRP",
        "mean_cost": 220.0,
        "mean_time": 1.0,
        "raw_costs": [218.0, 220.0, 222.0, 219.0, 221.0, 220.0],
        "raw_unpenalized_costs": [218.0, 220.0, 222.0, 219.0, 221.0, 220.0],
        "raw_violations": [0.0] * 6,
        "raw_is_feasible": [True] * 6,
        "raw_times": [1.0] * 6,
        "feasible_runs": 6,
        "feasibility_rate": 100.0,
      },
      "InfeasShort": {
        "problem_type": "CVRP",
        "mean_cost": None,
        "mean_time": 1.0,
        "raw_costs": infeas_pen,
        "raw_unpenalized_costs": infeas_unpen,
        "raw_violations": viols,
        "raw_is_feasible": [False] * 6,
        "raw_times": [1.0] * 6,
        "feasible_runs": 0,
        "feasibility_rate": 0.0,
      },
    }

  res_small_omega = analyze_problem("A-n32-k5", 190.0, _make_problem_results(100.0), StatisticalAnalyzer(alpha=0.05))
  res_large_omega = analyze_problem(
    "A-n32-k5", 190.0, _make_problem_results(100_000.0), StatisticalAnalyzer(alpha=0.05)
  )

  assert res_small_omega.cost_omnibus is not None
  assert res_large_omega.cost_omnibus is not None

  # FeasGood (rank 1.0) < FeasWorse (rank 2.0) < InfeasShort (rank 3.0)
  ranks = res_small_omega.cost_omnibus.ranks
  assert ranks["FeasGood"] == pytest.approx(1.0)
  assert ranks["FeasWorse"] == pytest.approx(2.0)
  assert ranks["InfeasShort"] == pytest.approx(3.0)

  # Exact invariance to omega_Q (100 vs 100_000)
  assert res_small_omega.cost_omnibus.ranks == res_large_omega.cost_omnibus.ranks
  assert res_small_omega.cost_omnibus.chi2_stat == pytest.approx(res_large_omega.cost_omnibus.chi2_stat)
  assert res_small_omega.cost_omnibus.friedman_pvalue == pytest.approx(res_large_omega.cost_omnibus.friedman_pvalue)
  for rec_a, rec_b in zip(res_small_omega.cost_pairwise, res_large_omega.cost_pairwise, strict=True):
    assert rec_a.summary.p_value == pytest.approx(rec_b.summary.p_value)
    assert rec_a.summary.effect_size == pytest.approx(rec_b.summary.effect_size)


def test_cohort_analysis_deb_ranking_includes_zero_feasible_blocks_and_is_omega_invariant() -> None:
  """WHAT: Multi-problem Friedman & Wilcoxon in `analyze_cohort` retain 0%-feasible blocks via Deb scoring.

  WHY: Dropping `mean_cost=None` blocks would exclude failing solvers; Deb scoring keeps them
  and ranks them strictly worse than every feasible solver, independent of penalty weights.
  """

  def _build_cohort(omega_q: float) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for idx in range(5):
      pname = f"vrp_p{idx + 1}"
      viol_b = 5.0 + idx
      viol_c = 12.0 + idx
      out[pname] = {
        "SplitFeas": {
          "problem_type": "CVRP",
          "problem_size": 30 + idx,
          "mean_cost": 500.0 + idx * 10.0,
          "std_cost": 2.0,
          "mean_gap": 1.0,
          "std_gap": 0.1,
          "mean_time": 1.0,
          "std_time": 0.05,
          "feasible_runs": 3,
          "successful_runs": 3,
          "feasibility_rate": 100.0,
          "mean_vehicles": 5.0,
          "raw_is_feasible": [True, True, True],
          "raw_violations": [0.0, 0.0, 0.0],
          "raw_costs": [500.0 + idx * 10.0] * 3,
        },
        "PenalizedLowViol": {
          "problem_type": "CVRP",
          "problem_size": 30 + idx,
          "mean_cost": None,
          "std_cost": None,
          "mean_gap": None,
          "std_gap": None,
          "mean_time": 0.8,
          "std_time": 0.05,
          "feasible_runs": 0,
          "successful_runs": 3,
          "feasibility_rate": 0.0,
          "mean_vehicles": None,
          "raw_is_feasible": [False, False, False],
          "raw_violations": [viol_b] * 3,
          "raw_costs": [400.0 + omega_q * viol_b] * 3,
        },
        "PenalizedHighViol": {
          "problem_type": "CVRP",
          "problem_size": 30 + idx,
          "mean_cost": None,
          "std_cost": None,
          "mean_gap": None,
          "std_gap": None,
          "mean_time": 0.7,
          "std_time": 0.05,
          "feasible_runs": 0,
          "successful_runs": 3,
          "feasibility_rate": 0.0,
          "mean_vehicles": None,
          "raw_is_feasible": [False, False, False],
          "raw_violations": [viol_c] * 3,
          "raw_costs": [350.0 + omega_q * viol_c] * 3,
        },
      }
    return out

  a1 = analyze_cohort(_build_cohort(100.0), baseline_algorithm="SplitFeas")
  a2 = analyze_cohort(_build_cohort(100_000.0), baseline_algorithm="SplitFeas")

  assert a1.quality_omnibus is not None
  assert a2.quality_omnibus is not None
  assert a1.quality_omnibus.n_blocks == 5
  assert a1.quality_omnibus.ranks["SplitFeas"] == pytest.approx(1.0)
  assert a1.quality_omnibus.ranks["PenalizedLowViol"] == pytest.approx(2.0)
  assert a1.quality_omnibus.ranks["PenalizedHighViol"] == pytest.approx(3.0)
  assert a1.quality_omnibus.ranks == a2.quality_omnibus.ranks
  assert a1.quality_omnibus.friedman_pvalue == pytest.approx(a2.quality_omnibus.friedman_pvalue)
  assert len(a1.quality_wilcoxon_vs_baseline) == 2
  for w1, w2 in zip(a1.quality_wilcoxon_vs_baseline, a2.quality_wilcoxon_vs_baseline, strict=True):
    assert w1.p_value == pytest.approx(w2.p_value)


def test_table_reports_conditional_vrp_columns_and_zero_feasible_em_dash(tmp_path: Path) -> None:
  """WHAT: `generate_markdown_tables` & `generate_latex_tables` emit VRP columns only for VRP cohorts.

  WHY: Pure TSP reports must remain byte-for-byte identical, whereas VRP cohorts must show
  `Feas. (%)` and `Vehicles` and render `"—"` when `feasible_runs == 0`.
  """
  tsp_results = {
    "berlin52": {
      "CPU": {
        "problem_type": "TSP",
        "mean_cost": 7542.0,
        "std_cost": 10.0,
        "mean_gap": 0.0,
        "std_gap": 0.1,
        "mean_time": 2.0,
        "std_time": 0.1,
      },
      "GPU": {
        "problem_type": "TSP",
        "mean_cost": 7550.0,
        "std_cost": 12.0,
        "mean_gap": 0.11,
        "std_gap": 0.15,
        "mean_time": 0.2,
        "std_time": 0.02,
      },
    }
  }
  tsp_cfgs = [ProblemConfig(name="berlin52", size=52, optimal=7542.0)]
  assert BenchmarkCohort.from_results(tsp_results, tsp_cfgs).is_vrp_cohort is False

  tsp_md = generate_markdown_tables(tsp_results, tsp_cfgs, tmp_path / "tsp", benchmark_name="tsp_test").read_text(
    encoding="utf-8"
  )
  tsp_tex = generate_latex_tables(tsp_results, tsp_cfgs, tmp_path / "tsp", benchmark_name="tsp_test").read_text(
    encoding="utf-8"
  )
  assert "Feas. (%)" not in tsp_md
  assert "Vehicles" not in tsp_md
  assert "Feas. (\\%)" not in tsp_tex

  vrp_results: dict[str, dict[str, dict[str, Any]]] = {
    "A-n32-k5": {
      "SplitGA": {
        "problem_type": "CVRP",
        "mean_cost": 790.0,
        "std_cost": 5.0,
        "mean_gap": 0.77,
        "std_gap": 0.2,
        "mean_time": 1.0,
        "std_time": 0.1,
        "feasible_runs": 5,
        "feasibility_rate": 100.0,
        "mean_vehicles": 5.0,
      },
      "PenalizedZero": {
        "problem_type": "CVRP",
        "mean_cost": None,
        "std_cost": None,
        "mean_gap": None,
        "std_gap": None,
        "mean_time": 0.5,
        "std_time": 0.05,
        "feasible_runs": 0,
        "feasibility_rate": 0.0,
        "mean_vehicles": None,
      },
    }
  }
  vrp_cfgs = [ProblemConfig(name="A-n32-k5", size=32, optimal=784.0)]
  assert BenchmarkCohort.from_results(vrp_results, vrp_cfgs).is_vrp_cohort is True

  vrp_md = generate_markdown_tables(vrp_results, vrp_cfgs, tmp_path / "vrp", benchmark_name="vrp_test").read_text(
    encoding="utf-8"
  )
  vrp_tex = generate_latex_tables(vrp_results, vrp_cfgs, tmp_path / "vrp", benchmark_name="vrp_test").read_text(
    encoding="utf-8"
  )
  assert "| Feas. (%) | Vehicles |" in vrp_md
  assert "| A-n32-k5 | PenalizedZero | — | — | — | 0.0 | — | 0.50 | 2.00x |" in vrp_md
  assert "| A-n32-k5 | SplitGA | 790.00 | 5.00 | 0.77 | 100.0 | 5.00 | 1.00 | 1.00x |" in vrp_md
  assert "Feas. (\\%) & Vehicles" in vrp_tex
  assert "0.0" in vrp_tex
  assert "—" in vrp_tex


def test_cohort_seed_averaged_deb_ranks_29_of_30_and_rejects_3_of_30_lucky_solver() -> None:
  """WHAT: Cohort ``R_bar_{p,j}`` ranks 29/30@0.1% ahead of 30/30@45%, and 30/30@16350 ahead of 3/30@16280.

  WHY: Task 17.25 (`DP17-10`) must neither collapse a 29/30 feasible solver to 0% feasible
  nor let a 3/30 lucky solver beat a 30/30 feasible solver via survivor-biased feasible means.
  """
  # Case 1: AlgoA (29/30 feasible at 0.1% gap, 1 infeasible) vs AlgoC (30/30 feasible at 45% gap)
  # Plus AlgoD (0/30 feasible) so k >= 3 for Friedman omnibus.
  cohort_a_vs_c: dict[str, dict[str, dict[str, Any]]] = {}
  for p_idx in range(4):
    opt = 1000.0 + 100.0 * p_idx
    a_unpen = [opt * 1.001] * 29 + [opt * 0.90]
    a_feas = [True] * 29 + [False]
    a_nu = [0.0] * 29 + [0.05]
    c_unpen = [opt * 1.45] * 30
    d_unpen = [opt * 0.80] * 30
    cohort_a_vs_c[f"vrp_c1_{p_idx}"] = {
      "AlgoA_29_30": {
        "problem_type": "CVRP",
        "mean_cost": opt * 1.001,
        "mean_gap": 0.1,
        "mean_time": 1.0,
        "feasibility_rate": 2900.0 / 30.0,
        "feasible_runs": 29,
        "successful_runs": 30,
        "raw_is_feasible": a_feas,
        "raw_unpenalized_costs": a_unpen,
        "raw_costs": [c if f else float("nan") for c, f in zip(a_unpen, a_feas, strict=True)],
        "raw_normalized_violations": a_nu,
      },
      "AlgoC_30_30_45pct": {
        "problem_type": "CVRP",
        "mean_cost": opt * 1.45,
        "mean_gap": 45.0,
        "mean_time": 1.0,
        "feasibility_rate": 100.0,
        "feasible_runs": 30,
        "successful_runs": 30,
        "raw_is_feasible": [True] * 30,
        "raw_unpenalized_costs": c_unpen,
        "raw_costs": c_unpen,
        "raw_normalized_violations": [0.0] * 30,
      },
      "AlgoD_0_30": {
        "problem_type": "CVRP",
        "mean_cost": None,
        "mean_gap": None,
        "mean_time": 1.0,
        "feasibility_rate": 0.0,
        "feasible_runs": 0,
        "successful_runs": 30,
        "raw_is_feasible": [False] * 30,
        "raw_unpenalized_costs": d_unpen,
        "raw_costs": [float("nan")] * 30,
        "raw_normalized_violations": [0.5] * 30,
      },
    }

  analysis_c1 = analyze_cohort(cohort_a_vs_c, baseline_algorithm="AlgoC_30_30_45pct")
  assert analysis_c1.quality_omnibus is not None
  # R_bar for AlgoA_29_30 = (29*1 + 1*2)/30 = 31/30 < R_bar for AlgoC = 59/30 < AlgoD = 3.0
  assert analysis_c1.quality_omnibus.ranks["AlgoA_29_30"] == pytest.approx(1.0)
  assert analysis_c1.quality_omnibus.ranks["AlgoC_30_30_45pct"] == pytest.approx(2.0)
  assert analysis_c1.quality_omnibus.ranks["AlgoD_0_30"] == pytest.approx(3.0)
  _, rbar_mat_c1 = BenchmarkCohort.from_results(cohort_a_vs_c).quality_score_matrix("mean_gap")
  assert rbar_mat_c1[0, 0] == pytest.approx(31.0 / 30.0)
  assert rbar_mat_c1[0, 1] == pytest.approx(59.0 / 30.0)

  # Case 2: AlgoA (30/30 feasible at 16,350) vs WeakAlgoB (3/30 lucky feasible at 16,280, 27/30 infeasible)
  cohort_a_vs_b: dict[str, dict[str, dict[str, Any]]] = {}
  for p_idx in range(4):
    a_costs = [16350.0 + p_idx * 10.0] * 30
    b_unpen = [16280.0 + p_idx * 10.0] * 3 + [15000.0] * 27
    b_feas = [True] * 3 + [False] * 27
    b_nu = [0.0] * 3 + [0.4] * 27
    c_costs = [18000.0 + p_idx * 10.0] * 30
    cohort_a_vs_b[f"vrp_c2_{p_idx}"] = {
      "AlgoA_30_30": {
        "problem_type": "CVRP",
        "mean_cost": 16350.0 + p_idx * 10.0,
        "mean_time": 1.0,
        "feasibility_rate": 100.0,
        "feasible_runs": 30,
        "successful_runs": 30,
        "raw_is_feasible": [True] * 30,
        "raw_unpenalized_costs": a_costs,
        "raw_costs": a_costs,
        "raw_normalized_violations": [0.0] * 30,
      },
      "WeakAlgoB_3_30": {
        "problem_type": "CVRP",
        "mean_cost": 16280.0 + p_idx * 10.0,
        "mean_time": 1.0,
        "feasibility_rate": 10.0,
        "feasible_runs": 3,
        "successful_runs": 30,
        "raw_is_feasible": b_feas,
        "raw_unpenalized_costs": b_unpen,
        "raw_costs": [c if f else float("nan") for c, f in zip(b_unpen, b_feas, strict=True)],
        "raw_normalized_violations": b_nu,
      },
      "AlgoC_30_30_slow": {
        "problem_type": "CVRP",
        "mean_cost": 18000.0 + p_idx * 10.0,
        "mean_time": 1.0,
        "feasibility_rate": 100.0,
        "feasible_runs": 30,
        "successful_runs": 30,
        "raw_is_feasible": [True] * 30,
        "raw_unpenalized_costs": c_costs,
        "raw_costs": c_costs,
        "raw_normalized_violations": [0.0] * 30,
      },
    }

  analysis_c2 = analyze_cohort(cohort_a_vs_b, baseline_algorithm="AlgoA_30_30")
  assert analysis_c2.quality_omnibus is not None
  assert analysis_c2.quality_omnibus.ranks["AlgoA_30_30"] < analysis_c2.quality_omnibus.ranks["WeakAlgoB_3_30"]
  _, rbar_mat_c2 = BenchmarkCohort.from_results(cohort_a_vs_b).quality_score_matrix("mean_cost")
  # On 3 seeds WeakAlgoB gets rank 1 and AlgoA gets rank 2; on 27 seeds AlgoA gets 1 and WeakAlgoB gets 3
  assert rbar_mat_c2[0, 0] == pytest.approx((3.0 * 2.0 + 27.0 * 1.0) / 30.0)
  assert rbar_mat_c2[0, 1] == pytest.approx((3.0 * 1.0 + 27.0 * 3.0) / 30.0)


def test_tier1_constrained_binomial_sign_invariance_and_nan_ci_under_six_feasible() -> None:
  """WHAT: Unit scaling (x1,000) and omega changes preserve Tier-1 stats bit-for-bit without fake CIs.

  WHY: Dimensionless ``hat_nu`` and exact Binomial Sign + Deb ranks eliminate unit/jump distortion,
  and descriptive means/CIs never consume ``C^max + 1 + nu`` sentinels (returning ``NaN`` CI if ``< 6``).
  """

  def _build_problem(scale_q: float, omega_q: float) -> dict[str, dict[str, Any]]:
    ctx: Any = SimpleNamespace(
      constraints=SimpleNamespace(capacity=100.0 * scale_q, max_route_distance=200.0, max_vehicles=5)
    )
    # 8 seeds: AlgoA has 4 feasible + 4 infeasible (< 6 feasible -> NaN CI!),
    # AlgoB has 8 feasible with worse cost on seeds 0..3 and beats AlgoA on seeds 4..7,
    # AlgoC has 0 feasible with higher hat_nu.
    raw_comps_a = np.array(
      [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [10.0 * scale_q, 20.0, 0.0],
        [15.0 * scale_q, 10.0, 0.0],
        [12.0 * scale_q, 15.0, 1.0],
        [20.0 * scale_q, 5.0, 0.0],
      ],
      dtype=np.float64,
    )
    raw_comps_c = np.array([[50.0 * scale_q, 40.0, 1.0]] * 8, dtype=np.float64)
    hat_nu_a = compute_normalized_violations(raw_comps_a, ctx).tolist()
    hat_nu_c = compute_normalized_violations(raw_comps_c, ctx).tolist()
    raw_viol_a = np.sum(raw_comps_a, axis=1).tolist()
    raw_viol_c = np.sum(raw_comps_c, axis=1).tolist()

    feas_a = [True, True, True, True, False, False, False, False]
    unpen_a = [200.0, 202.0, 204.0, 206.0, 150.0, 151.0, 152.0, 153.0]
    pen_a = [c + omega_q * v for c, v in zip(unpen_a, raw_viol_a, strict=True)]
    unpen_b = [230.0, 232.0, 234.0, 236.0, 238.0, 240.0, 242.0, 244.0]
    unpen_c = [140.0] * 8
    pen_c = [c + omega_q * v for c, v in zip(unpen_c, raw_viol_c, strict=True)]

    return {
      "AlgoA": {
        "problem_type": "CVRP",
        "mean_time": 1.0,
        "raw_costs": [c if f else p for c, p, f in zip(unpen_a, pen_a, feas_a, strict=True)],
        "raw_unpenalized_costs": unpen_a,
        "raw_violations": raw_viol_a,
        "raw_normalized_violations": hat_nu_a,
        "raw_is_feasible": feas_a,
        "raw_times": [1.0] * 8,
        "feasible_runs": 4,
        "feasibility_rate": 50.0,
      },
      "AlgoB": {
        "problem_type": "CVRP",
        "mean_time": 1.2,
        "raw_costs": unpen_b,
        "raw_unpenalized_costs": unpen_b,
        "raw_violations": [0.0] * 8,
        "raw_normalized_violations": [0.0] * 8,
        "raw_is_feasible": [True] * 8,
        "raw_times": [1.2] * 8,
        "feasible_runs": 8,
        "feasibility_rate": 100.0,
      },
      "AlgoC": {
        "problem_type": "CVRP",
        "mean_time": 0.9,
        "raw_costs": pen_c,
        "raw_unpenalized_costs": unpen_c,
        "raw_violations": raw_viol_c,
        "raw_normalized_violations": hat_nu_c,
        "raw_is_feasible": [False] * 8,
        "raw_times": [0.9] * 8,
        "feasible_runs": 0,
        "feasibility_rate": 0.0,
      },
    }

  res_kg = analyze_problem("VRP-scale", 190.0, _build_problem(1.0, 100.0), StatisticalAnalyzer(alpha=0.05))
  res_g = analyze_problem("VRP-scale", 190.0, _build_problem(1000.0, 50_000.0), StatisticalAnalyzer(alpha=0.05))

  assert res_kg.cost_omnibus is not None and res_g.cost_omnibus is not None
  assert res_kg.cost_omnibus.ranks == res_g.cost_omnibus.ranks
  assert res_kg.cost_omnibus.chi2_stat == res_g.cost_omnibus.chi2_stat
  assert res_kg.cost_omnibus.friedman_pvalue == res_g.cost_omnibus.friedman_pvalue

  ab = next(rec.summary for rec in res_kg.cost_pairwise if rec.summary.comparison_label == "AlgoA vs AlgoB")
  ab_g = next(rec.summary for rec in res_g.cost_pairwise if rec.summary.comparison_label == "AlgoA vs AlgoB")
  assert ab.test_used == "exact_binomial_sign_deb"
  assert ab.sample_size == 8
  assert ab.p_value == ab_g.p_value == pytest.approx(1.0)
  assert ab.effect_size == ab_g.effect_size == pytest.approx(0.0)
  assert ab.vargha_delaney_a12 == pytest.approx(0.5)
  # Only 4 feasible runs for AlgoA -> mean_a is 203.0 (never C^max+1!), ci_95_a is (nan, nan),
  # and only 4 mutually feasible pairs -> mean_difference is -30.0, ci_95_difference is (nan, nan).
  assert ab.mean_a == pytest.approx(203.0)
  assert math.isnan(ab.ci_95_a[0]) and math.isnan(ab.ci_95_a[1])
  assert ab.mean_difference == pytest.approx(-30.0)
  assert math.isnan(ab.ci_95_difference[0]) and math.isnan(ab.ci_95_difference[1])
  # AlgoB has 8 feasible runs -> valid finite Hodges-Lehmann CI
  assert math.isfinite(ab.ci_95_b[0]) and math.isfinite(ab.ci_95_b[1])

  # AlgoB (8/8 feas) vs AlgoC (0/8 feas) -> Kerby-sign effect_size == -1.0 (AlgoB beats AlgoC), A12 == 1.0
  bc = next(rec.summary for rec in res_kg.cost_pairwise if rec.summary.comparison_label == "AlgoB vs AlgoC")
  bc_g = next(rec.summary for rec in res_g.cost_pairwise if rec.summary.comparison_label == "AlgoB vs AlgoC")
  assert bc.p_value == bc_g.p_value == pytest.approx(2.0**-7)
  assert bc.effect_size == bc_g.effect_size == pytest.approx(-1.0)
  assert bc.vargha_delaney_a12 == pytest.approx(1.0)
  assert math.isnan(bc.mean_b) and math.isnan(bc.mean_difference)


def test_100_percent_feasible_vrp_matches_unconstrained_demsar_byte_for_byte() -> None:
  """WHAT: 100%-feasible VRP produces identical Tier-1 and Tier-2 results to unconstrained TSP.

  WHY: Preserves the 100% feasible identity contract (exact Wilcoxon / Hodges-Lehmann / Friedman / Nemenyi).
  """
  rng = np.random.default_rng(7)
  tsp_prob: dict[str, dict[str, Any]] = {}
  vrp_prob: dict[str, dict[str, Any]] = {}
  for idx, name in enumerate(("Alg1", "Alg2", "Alg3")):
    costs = (1000.0 + idx * 25.0 + rng.normal(0.0, 5.0, size=10)).tolist()
    times = (1.0 + idx * 0.2 + rng.uniform(0.0, 0.05, size=10)).tolist()
    tsp_prob[name] = {
      "problem_type": "TSP",
      "mean_cost": float(np.mean(costs)),
      "mean_time": float(np.mean(times)),
      "raw_costs": costs,
      "raw_times": times,
    }
    vrp_prob[name] = {
      "problem_type": "CVRP",
      "mean_cost": float(np.mean(costs)),
      "mean_time": float(np.mean(times)),
      "raw_costs": costs,
      "raw_unpenalized_costs": costs,
      "raw_is_feasible": [True] * 10,
      "raw_violations": [0.0] * 10,
      "raw_normalized_violations": [0.0] * 10,
      "raw_times": times,
      "feasible_runs": 10,
      "feasibility_rate": 100.0,
    }

  analyzer = StatisticalAnalyzer(alpha=0.05)
  tsp_res = analyze_problem("p_ident", 950.0, tsp_prob, analyzer)
  vrp_res = analyze_problem("p_ident", 950.0, vrp_prob, analyzer)

  assert tsp_res.to_dict() == vrp_res.to_dict()
  assert all(rec.summary.test_used == "wilcoxon_signed_rank" for rec in vrp_res.cost_pairwise)
  assert all(rec.summary.vargha_delaney_a12 is None for rec in vrp_res.cost_pairwise)
