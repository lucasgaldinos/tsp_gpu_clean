"""Role-oriented VRP route decoding strategies (`PrinsSplitDecoder`, `GreedySplitDecoder`, `Task 17.21`)."""

from .delimiter_ops import SplitBatchResult, compact_delimited_route, strip_depot_delimiters
from .greedy_split import GreedySplitDecoder
from .prins_split import PrinsSplitDecoder
from .split_base import DEFAULT_MAX_SHARED_CUSTOMERS

__all__: list[str] = [
  "DEFAULT_MAX_SHARED_CUSTOMERS",
  "GreedySplitDecoder",
  "PrinsSplitDecoder",
  "SplitBatchResult",
  "compact_delimited_route",
  "strip_depot_delimiters",
]
