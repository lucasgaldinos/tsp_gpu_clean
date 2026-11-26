"""
Modular evolutionary operator strategies for TSP and routing optimization.

All route-transforming operators (`CrossoverStrategy`, `MutationStrategy`,
`LocalSearchStrategy`, `EvaluationStrategy`) accept `RouteOrRoutes`
(`(n_cities,)` or `(N, n_cities)`), enabling arbitrary pipeline ordering.

Every class name is prefixed with the slot it fills (`Initialization*`, `Selection*`,
`Crossover*`, `Mutation*`, `LocalSearch*`, `Evaluation*`) and every class derives from
`StrategyBase`, so the registry key reads as "which slot, which variant" (`DP11-1`).
"""

from algorithms.pipeline_steps import PIPELINE_STEP_REGISTRY, register_pipeline_step
from protocols.algorithm_strategies import (
  CrossoverStrategy,
  EvaluationStrategy,
  InitializationStrategy,
  LocalSearchStrategy,
  MutationStrategy,
  SelectionStrategy,
)

from .base import StrategyBase
from .crossover import (
  CrossoverGPUOrder,
  CrossoverGPUPartiallyMapped,
  CrossoverOrder,
  CrossoverPartiallyMapped,
)
from .evaluation import (
  EvaluationCPUCost,
  EvaluationGPUCost,
)
from .initialization import InitializationNearestNeighbor, InitializationRandom
from .local_search import LocalSearchCPUTwoOpt, LocalSearchGPUTwoOpt, LocalSearchNoOp
from .mutation import (
  MutationGPUInversion,
  MutationGPUSwap,
  MutationInversion,
  MutationSwap,
)
from .selection import (
  SelectionGPUTournament,
  SelectionRouletteWheel,
  SelectionTournament,
)

STRATEGY_REGISTRY: dict[str, type] = {
  "CrossoverGPUOrder": CrossoverGPUOrder,
  "CrossoverGPUPartiallyMapped": CrossoverGPUPartiallyMapped,
  "CrossoverOrder": CrossoverOrder,
  "CrossoverPartiallyMapped": CrossoverPartiallyMapped,
  "EvaluationCPUCost": EvaluationCPUCost,
  "EvaluationGPUCost": EvaluationGPUCost,
  "InitializationNearestNeighbor": InitializationNearestNeighbor,
  "InitializationRandom": InitializationRandom,
  "LocalSearchCPUTwoOpt": LocalSearchCPUTwoOpt,
  "LocalSearchGPUTwoOpt": LocalSearchGPUTwoOpt,
  "LocalSearchNoOp": LocalSearchNoOp,
  "MutationGPUInversion": MutationGPUInversion,
  "MutationGPUSwap": MutationGPUSwap,
  "MutationInversion": MutationInversion,
  "MutationSwap": MutationSwap,
  "SelectionGPUTournament": SelectionGPUTournament,
  "SelectionRouletteWheel": SelectionRouletteWheel,
  "SelectionTournament": SelectionTournament,
}


_STRATEGY_PROTOCOLS: tuple[type, ...] = (
  InitializationStrategy,
  SelectionStrategy,
  CrossoverStrategy,
  MutationStrategy,
  LocalSearchStrategy,
  EvaluationStrategy,
)

# Derive required method names reflectively from the PEP 544 protocols so validation
# stays locked to ``protocols.algorithm_strategies`` without instantiating custom classes.
_STRATEGY_PROTOCOL_METHODS: dict[str, tuple[str, ...]] = {
  proto.__name__: tuple(
    sorted(name for name, attr in vars(proto).items() if not name.startswith("_") and callable(attr))
  )
  for proto in _STRATEGY_PROTOCOLS
}


def register_strategy(name: str, strategy_cls: type, *, overwrite: bool = False) -> None:
  """Register a custom operator strategy class in ``STRATEGY_REGISTRY`` without modifying ``src/``.

  Validates that ``strategy_cls`` is a class defining every required callable method of at
  least one PEP 544 strategy protocol in ``protocols.algorithm_strategies`` without
  instantiating it.

  Args:
      name: Unique registry key referenced from declarative JSON strategy slots.
      strategy_cls: Class implementing one of the strategy protocols in
          ``protocols.algorithm_strategies``.
      overwrite: Allow replacing an existing registration with the same ``name``.

  Raises:
      ValueError: If ``name`` is empty or already registered with ``overwrite=False``.
      TypeError: If ``strategy_cls`` is not a class or does not implement any strategy protocol.
  """
  if not isinstance(name, str) or not name.strip():
    raise ValueError("Strategy name must be a non-empty string.")
  if not isinstance(strategy_cls, type):
    raise TypeError(f"Expected a strategy class (type), got {type(strategy_cls).__name__}.")
  if not any(
    all(callable(getattr(strategy_cls, method, None)) for method in methods)
    for methods in _STRATEGY_PROTOCOL_METHODS.values()
  ):
    expected_desc = ", ".join(f"{proto}({', '.join(methods)})" for proto, methods in _STRATEGY_PROTOCOL_METHODS.items())
    raise TypeError(
      f"Strategy class '{strategy_cls.__name__}' does not implement any strategy protocol in "
      f"protocols.algorithm_strategies. Expected one of: {expected_desc}."
    )
  if name in STRATEGY_REGISTRY and not overwrite:
    raise ValueError(f"Strategy '{name}' is already registered. Pass overwrite=True to replace it.")
  STRATEGY_REGISTRY[name] = strategy_cls


__all__: list[str] = [
  "PIPELINE_STEP_REGISTRY",
  "STRATEGY_REGISTRY",
  "CrossoverGPUOrder",
  "CrossoverGPUPartiallyMapped",
  "CrossoverOrder",
  "CrossoverPartiallyMapped",
  "EvaluationCPUCost",
  "EvaluationGPUCost",
  "InitializationNearestNeighbor",
  "InitializationRandom",
  "LocalSearchCPUTwoOpt",
  "LocalSearchGPUTwoOpt",
  "LocalSearchNoOp",
  "MutationGPUInversion",
  "MutationGPUSwap",
  "MutationInversion",
  "MutationSwap",
  "SelectionGPUTournament",
  "SelectionRouletteWheel",
  "SelectionTournament",
  "StrategyBase",
  "register_pipeline_step",
  "register_strategy",
]
