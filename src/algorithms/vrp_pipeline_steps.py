"""VRP adaptive-penalty and fleet bin-packing pipeline steps (`DP17-7`, `DP17-9`).

Split into a dedicated module so ``pipeline_steps.py`` stays concise while
preserving every symbol and ``PIPELINE_STEP_REGISTRY`` entry on ``pipeline_steps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from protocols.vrp_strategies import BinPackingStrategy, PenaltyPolicy

if TYPE_CHECKING:
  from algorithms.metaheuristics.genetic_algorithm import GenerationState
  from algorithms.pipeline_steps import StageAnchor
  from protocols.problem_context import ProblemContext
  from utils.phase_timer import PhaseTimer


class PenaltyAdaptationStep:
  """Adapt ``PenaltyPolicy`` multipliers every ``interval`` generations and refresh fitness (`DP17-9`)."""

  stage_anchor: ClassVar[StageAnchor] = "post_survival"

  def __init__(
    self,
    penalty_policy: PenaltyPolicy | None = None,
    interval: int = 5,
  ) -> None:
    if penalty_policy is not None and not isinstance(penalty_policy, PenaltyPolicy):
      raise TypeError(f"penalty_policy must implement PenaltyPolicy, got {type(penalty_policy).__name__}")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
      raise ValueError(f"interval must be a positive int, got {interval!r}")
    if penalty_policy is None:
      from algorithms.strategies.penalties import AdaptivePenaltyPolicy

      self.penalty_policy: PenaltyPolicy = AdaptivePenaltyPolicy()
    else:
      self.penalty_policy = penalty_policy
    self.interval: int = interval

  def strategies(self) -> list[Any]:
    """Expose owned penalty policy so ``GeneticAlgorithm`` resets/warms it automatically."""
    return [self.penalty_policy]

  def public_params(self) -> dict[str, Any]:
    """Return declarative cadence parameters for checkpoint serialization."""
    return {"interval": self.interval}

  def warmup(self, context: ProblemContext) -> None:
    """Delegate warm-up to the owned penalty policy if supported."""
    hook = getattr(self.penalty_policy, "warmup", None)
    if callable(hook):
      hook(context)

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Adapt penalty weights on cadence and immediately refresh ``state.fitness`` in O(N)."""
    _ = timer
    if state.generation % self.interval != 0:
      return
    if state.population_assessment is None:
      return

    adapt_hook = getattr(self.penalty_policy, "adapt", None)
    if callable(adapt_hook):
      adapt_hook(state.population_assessment, state.context)

    omega_q, omega_l, omega_k = self.penalty_policy.weights(state.context)
    state.refresh_penalized_fitness(omega_q, omega_l, omega_k)


class FleetBinPackStep:
  """Self-gated fleet-repacking pipeline step activated when ``max_vehicles`` is set (`DP17-7`, `DP17-9`)."""

  def __init__(
    self,
    bin_packing: BinPackingStrategy | None = None,
    target: Literal["offspring", "population"] = "offspring",
  ) -> None:
    if bin_packing is not None and not isinstance(bin_packing, BinPackingStrategy):
      raise TypeError(f"bin_packing must implement BinPackingStrategy, got {type(bin_packing).__name__}")
    if target not in ("offspring", "population"):
      raise ValueError(f"target must be 'offspring' or 'population', got {target!r}")
    if bin_packing is None:
      from algorithms.strategies.bin_packing import BinPackingCPUFFD

      self.bin_packing: BinPackingStrategy = BinPackingCPUFFD()
    else:
      self.bin_packing = bin_packing
    self.target: Literal["offspring", "population"] = target

  @property
  def stage_anchor(self) -> StageAnchor:
    """Run before evaluation for offspring or after survival for the surviving population."""
    return "pre_evaluation" if self.target == "offspring" else "post_survival"

  def strategies(self) -> list[Any]:
    """Return owned bin-packing strategy for telemetry reset and warm-up."""
    return [self.bin_packing]

  def public_params(self) -> dict[str, Any]:
    """Return declarative target setting for checkpoint serialization."""
    return {"target": self.target}

  def warmup(self, context: ProblemContext) -> None:
    """Warm up the owned bin-packing strategy when supported."""
    hook = getattr(self.bin_packing, "warmup", None)
    if callable(hook):
      hook(context)

  def run(self, state: GenerationState, timer: PhaseTimer) -> None:
    """Repack target batch when ``is_vrp`` and ``max_vehicles is not None``; O(1) no-op otherwise."""
    if not state.context.problem.is_vrp or state.context.constraints.max_vehicles is None:
      return

    if self.target == "offspring":
      if state.offspring is None:
        raise RuntimeError("FleetBinPackStep(target='offspring') called before offspring exists.")
      state.offspring = timer.measure(
        "decoding",
        self.bin_packing.pack_batch,
        state.offspring,
        state.context,
      )
    else:
      state.population = timer.measure(
        "decoding",
        self.bin_packing.pack_batch,
        state.population,
        state.context,
      )
      state.population_assessment = None


__all__: list[str] = ["FleetBinPackStep", "PenaltyAdaptationStep"]
