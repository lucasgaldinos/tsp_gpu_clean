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
from protocols.vrp_strategies import (
  BinPackingStrategy,
  PenaltyPolicy,
  RouteDecoder,
)

from .base import StrategyBase
from .bin_packing import (
  BinPackingCPUBFD,
  BinPackingCPUFFD,
  BinPackingGPUBFD,
  BinPackingGPUFFD,
)
from .crossover import (
  CrossoverCPUPermOrder,
  CrossoverCPUPermPartiallyMapped,
  CrossoverCPURouteRBX,
  CrossoverGPUPermOrder,
  CrossoverGPUPermPartiallyMapped,
  CrossoverGPURouteRBX,
)
from .decoding import GreedySplitDecoder, PrinsSplitDecoder
from .evaluation import (
  EvaluationCPUCost,
  EvaluationGPUCost,
)
from .initialization import (
  InitializationCPUPermRandom,
  InitializationCPURouteClarkeWright,
  InitializationCPURouteSweep,
  InitializationCPUTourNearestNeighbor,
  InitializationGPUPermRandom,
  InitializationGPURouteClarkeWright,
  InitializationGPURouteSweep,
  InitializationGPUTourNearestNeighbor,
)
from .local_search import (
  LocalSearchCPURouteTwoOptStar,
  LocalSearchCPUTourTwoOpt,
  LocalSearchGPURouteTwoOptStar,
  LocalSearchGPUTourTwoOpt,
  LocalSearchNoOp,
)
from .mutation import (
  MutationCPUPermInversion,
  MutationCPUPermSwap,
  MutationCPURouteRelocate,
  MutationCPURouteSwap,
  MutationGPUPermInversion,
  MutationGPUPermSwap,
  MutationGPURouteRelocate,
  MutationGPURouteSwap,
)
from .penalties import (
  AdaptivePenaltyPolicy,
  FixedPenaltyPolicy,
  InstanceScaledPenaltyPolicy,
  StrictFeasibilityPolicy,
)
from .selection import (
  SelectionCPURouletteWheel,
  SelectionCPUTournament,
  SelectionGPUTournament,
)

STRATEGY_REGISTRY: dict[str, type] = {
  "CrossoverCPUPermOrder": CrossoverCPUPermOrder,
  "CrossoverCPUPermPartiallyMapped": CrossoverCPUPermPartiallyMapped,
  "CrossoverCPURouteRBX": CrossoverCPURouteRBX,
  "CrossoverGPUPermOrder": CrossoverGPUPermOrder,
  "CrossoverGPUPermPartiallyMapped": CrossoverGPUPermPartiallyMapped,
  "CrossoverGPURouteRBX": CrossoverGPURouteRBX,
  "EvaluationCPUCost": EvaluationCPUCost,
  "EvaluationGPUCost": EvaluationGPUCost,
  "InitializationCPUPermRandom": InitializationCPUPermRandom,
  "InitializationCPURouteClarkeWright": InitializationCPURouteClarkeWright,
  "InitializationCPURouteSweep": InitializationCPURouteSweep,
  "InitializationCPUTourNearestNeighbor": InitializationCPUTourNearestNeighbor,
  "InitializationGPUPermRandom": InitializationGPUPermRandom,
  "InitializationGPURouteClarkeWright": InitializationGPURouteClarkeWright,
  "InitializationGPURouteSweep": InitializationGPURouteSweep,
  "InitializationGPUTourNearestNeighbor": InitializationGPUTourNearestNeighbor,
  "LocalSearchCPURouteTwoOptStar": LocalSearchCPURouteTwoOptStar,
  "LocalSearchCPUTourTwoOpt": LocalSearchCPUTourTwoOpt,
  "LocalSearchGPURouteTwoOptStar": LocalSearchGPURouteTwoOptStar,
  "LocalSearchGPUTourTwoOpt": LocalSearchGPUTourTwoOpt,
  "LocalSearchNoOp": LocalSearchNoOp,
  "MutationCPUPermInversion": MutationCPUPermInversion,
  "MutationCPUPermSwap": MutationCPUPermSwap,
  "MutationCPURouteRelocate": MutationCPURouteRelocate,
  "MutationCPURouteSwap": MutationCPURouteSwap,
  "MutationGPUPermInversion": MutationGPUPermInversion,
  "MutationGPUPermSwap": MutationGPUPermSwap,
  "MutationGPURouteRelocate": MutationGPURouteRelocate,
  "MutationGPURouteSwap": MutationGPURouteSwap,
  "SelectionCPURouletteWheel": SelectionCPURouletteWheel,
  "SelectionCPUTournament": SelectionCPUTournament,
  "SelectionGPUTournament": SelectionGPUTournament,
}

SUB_STRATEGY_REGISTRY: dict[str, type] = {
  "AdaptivePenaltyPolicy": AdaptivePenaltyPolicy,
  "BinPackingCPUBFD": BinPackingCPUBFD,
  "BinPackingCPUFFD": BinPackingCPUFFD,
  "BinPackingGPUBFD": BinPackingGPUBFD,
  "BinPackingGPUFFD": BinPackingGPUFFD,
  "FixedPenaltyPolicy": FixedPenaltyPolicy,
  "GreedySplitDecoder": GreedySplitDecoder,
  "InstanceScaledPenaltyPolicy": InstanceScaledPenaltyPolicy,
  "PrinsSplitDecoder": PrinsSplitDecoder,
  "StrictFeasibilityPolicy": StrictFeasibilityPolicy,
}

_SUB_STRATEGY_PROTOCOLS: tuple[type, ...] = (
  PenaltyPolicy,
  RouteDecoder,
  BinPackingStrategy,
)


def register_sub_strategy(name: str, sub_strategy_cls: type, *, overwrite: bool = False) -> None:
  """Register a custom composable sub-strategy class in ``SUB_STRATEGY_REGISTRY``."""
  if not isinstance(name, str) or not name.strip():
    raise ValueError("Sub-strategy name must be a non-empty string.")
  if not isinstance(sub_strategy_cls, type):
    raise TypeError(f"Expected a sub-strategy class (type), got {type(sub_strategy_cls).__name__}.")
  if not any(issubclass(sub_strategy_cls, proto) for proto in _SUB_STRATEGY_PROTOCOLS):
    expected = ", ".join(proto.__name__ for proto in _SUB_STRATEGY_PROTOCOLS)
    raise TypeError(f"Sub-strategy class '{sub_strategy_cls.__name__}' does not implement any of: {expected}.")
  if name in SUB_STRATEGY_REGISTRY and not overwrite:
    raise ValueError(f"Sub-strategy '{name}' is already registered. Pass overwrite=True to replace it.")
  SUB_STRATEGY_REGISTRY[name] = sub_strategy_cls


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
  "SUB_STRATEGY_REGISTRY",
  "AdaptivePenaltyPolicy",
  "BinPackingCPUBFD",
  "BinPackingCPUFFD",
  "BinPackingGPUBFD",
  "BinPackingGPUFFD",
  "CrossoverCPUPermOrder",
  "CrossoverCPUPermPartiallyMapped",
  "CrossoverCPURouteRBX",
  "CrossoverGPUPermOrder",
  "CrossoverGPUPermPartiallyMapped",
  "CrossoverGPURouteRBX",
  "EvaluationCPUCost",
  "EvaluationGPUCost",
  "FixedPenaltyPolicy",
  "GreedySplitDecoder",
  "InitializationCPUPermRandom",
  "InitializationCPURouteClarkeWright",
  "InitializationCPURouteSweep",
  "InitializationCPUTourNearestNeighbor",
  "InitializationGPUPermRandom",
  "InitializationGPURouteClarkeWright",
  "InitializationGPURouteSweep",
  "InitializationGPUTourNearestNeighbor",
  "InstanceScaledPenaltyPolicy",
  "LocalSearchCPURouteTwoOptStar",
  "LocalSearchCPUTourTwoOpt",
  "LocalSearchGPURouteTwoOptStar",
  "LocalSearchGPUTourTwoOpt",
  "LocalSearchNoOp",
  "MutationCPUPermInversion",
  "MutationCPUPermSwap",
  "MutationCPURouteRelocate",
  "MutationCPURouteSwap",
  "MutationGPUPermInversion",
  "MutationGPUPermSwap",
  "MutationGPURouteRelocate",
  "MutationGPURouteSwap",
  "PrinsSplitDecoder",
  "SelectionCPURouletteWheel",
  "SelectionCPUTournament",
  "SelectionGPUTournament",
  "StrategyBase",
  "StrictFeasibilityPolicy",
  "register_pipeline_step",
  "register_strategy",
  "register_sub_strategy",
]
