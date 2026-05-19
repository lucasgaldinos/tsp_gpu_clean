#!/usr/bin/env python3
"""Test early stopping convergence detection in FullGPU GA kernel.

Validates:
- Optimal cost detection triggers early exit
- Stagnation detection triggers early exit after patience generations
- Solution quality within acceptable bounds
- H2D memory transfers remain minimal (full-GPU approach)
"""

import sys
from pathlib import Path

import pytest
import cupy as cp

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.metaheuristics.genetic_algorithm_full_gpu_early_stop import (
    GeneticAlgorithmFullGPUEarlyStop,
)
from src.loaders.database_loader import DatabaseLoader
from src.protocols.problem_context import ProblemContext


# -- Fixtures ----------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent.parent.parent / "datasets" / "routing.duckdb"

KROA100_OPTIMAL = 21282


@pytest.fixture(scope="module")
def kroa100_problem():
    """Load kroA100 once for all tests in this module."""
    with DatabaseLoader(str(DB_PATH)) as loader:
        yield loader.load("kroA100")


@pytest.fixture(scope="module")
def kroa100_context(kroa100_problem):
    """Create ProblemContext for kroA100."""
    return ProblemContext(kroa100_problem, xp=cp)


@pytest.fixture(scope="module")
def kroa100_customers(kroa100_problem):
    """Return customer indices."""
    return list(range(kroa100_problem.dimension))


def _make_ga(**overrides):
    defaults = dict(
        population_size=256,
        mutation_rate=0.02,
        tournament_size=5,
        two_opt_iterations=10,
        seed=42,
    )
    defaults.update(overrides)
    return GeneticAlgorithmFullGPUEarlyStop(**defaults)


# -- Tests -------------------------------------------------------------------


class TestOptimalDetection:
    """Early stopping when optimal cost is reached."""

    def test_stops_before_max_generations(
        self, kroa100_context, kroa100_customers
    ):
        ga = _make_ga()
        _tour, stats = ga.evolve(
            kroa100_context,
            kroa100_customers,
            max_generations=20000,
            optimal_cost=KROA100_OPTIMAL,
            patience=50,
        )
        # Should stop well before 20000 generations
        actual_gens = stats.get(
            "actual_generations", len(stats["best_cost_history"])
        )
        assert actual_gens < 20000, (
            f"Expected early stop but ran {actual_gens}/20000 generations"
        )

    def test_solution_quality_within_5pct(
        self, kroa100_context, kroa100_customers
    ):
        ga = _make_ga()
        _tour, stats = ga.evolve(
            kroa100_context,
            kroa100_customers,
            max_generations=20000,
            optimal_cost=KROA100_OPTIMAL,
            patience=50,
        )
        gap_pct = (stats["best_fitness"] - KROA100_OPTIMAL) / KROA100_OPTIMAL * 100
        assert gap_pct <= 5.0, (
            f"Best cost {stats['best_fitness']:.0f} is {gap_pct:.2f}% from optimal"
        )


class TestStagnationDetection:
    """Early stopping when no improvement for `patience` generations."""

    def test_stagnation_stops_early(
        self, kroa100_context, kroa100_customers
    ):
        ga = _make_ga(seed=99)
        _tour, stats = ga.evolve(
            kroa100_context,
            kroa100_customers,
            max_generations=2000,
            optimal_cost=None,  # Disable optimal detection
            patience=50,
        )
        actual_gens = stats.get(
            "actual_generations", len(stats["best_cost_history"])
        )
        assert actual_gens < 2000, (
            f"Expected stagnation stop but ran {actual_gens}/2000 generations"
        )

    def test_stagnation_produces_good_solution(
        self, kroa100_context, kroa100_customers
    ):
        ga = _make_ga(seed=99)
        _tour, stats = ga.evolve(
            kroa100_context,
            kroa100_customers,
            max_generations=2000,
            optimal_cost=None,
            patience=50,
        )
        gap_pct = (stats["best_fitness"] - KROA100_OPTIMAL) / KROA100_OPTIMAL * 100
        assert gap_pct <= 10.0, (
            f"Stagnation solution {stats['best_fitness']:.0f} is {gap_pct:.2f}% "
            f"from optimal — expected ≤10%"
        )


class TestMemoryTransfers:
    """Full-GPU approach should have minimal H2D transfers."""

    def test_h2d_under_1mb(self, kroa100_context, kroa100_customers):
        ga = _make_ga()
        _tour, stats = ga.evolve(
            kroa100_context,
            kroa100_customers,
            max_generations=2000,
            optimal_cost=KROA100_OPTIMAL,
            patience=50,
        )
        h2d_mb = stats["h2d_bytes"] / (1024 * 1024)
        assert h2d_mb < 1.0, (
            f"H2D transfers {h2d_mb:.2f} MB — expected < 1 MB for full-GPU"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
