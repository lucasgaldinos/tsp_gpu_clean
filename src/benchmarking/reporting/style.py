"""Neutral presentation configuration and shared helpers for benchmark reporting.

All formatting choices that belong to a specific publication, language, or experiment
(variant display names, decimal separators, LaTeX table label prefixes, problem-size
stratum thresholds, and figure/diagram axis titles) are carried in ``ReportStyle`` and
injected externally via Python or a JSON file (``--report-style``). Nothing in ``src/``
hardcodes experiment-specific variant names or non-English prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from benchmarking.analysis.stats_utils import finite_metric

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from benchmarking.runner.config_loader import ProblemConfig

ReportArtifact = Literal["markdown", "latex", "macros", "cd", "plots"]

ALL_REPORT_ARTIFACTS: tuple[ReportArtifact, ...] = (
  "markdown",
  "latex",
  "macros",
  "cd",
  "plots",
)

DEFAULT_SIZE_THRESHOLDS: tuple[int, int] = (100, 400)

_VALID_ARTIFACT_SET: frozenset[str] = frozenset(ALL_REPORT_ARTIFACTS)


def escape_latex_identifier(name: str) -> str:
  """Escape LaTeX special characters (``_`` and ``%``) in identifiers.

  Replaces ``_`` with ``\\_`` and ``%`` with ``\\%`` so algorithm, problem, and
  metric identifiers compile without subscript math-mode errors or comment truncation.
  """
  return name.replace("_", "\\_").replace("%", "\\%")


def validate_size_thresholds(size_thresholds: Sequence[int]) -> tuple[int, int]:
  """Validate and normalize a ``(small_max, medium_max)`` size-stratum boundary pair."""
  if len(size_thresholds) != 2:
    raise ValueError(
      f"size_thresholds must contain exactly 2 integers (small_max, medium_max), got {size_thresholds!r}"
    )
  raw_small, raw_medium = size_thresholds[0], size_thresholds[1]
  if isinstance(raw_small, bool) or isinstance(raw_medium, bool):
    raise ValueError(f"size_thresholds entries must be integers, not booleans: {size_thresholds!r}")
  small_max = int(raw_small)
  medium_max = int(raw_medium)
  if small_max <= 0 or medium_max <= small_max:
    raise ValueError(f"size_thresholds must satisfy 0 < small_max < medium_max, got ({small_max}, {medium_max})")
  return (small_max, medium_max)


def size_category_labels(size_thresholds: tuple[int, int] = DEFAULT_SIZE_THRESHOLDS) -> tuple[str, str, str]:
  """Build ``(Small, Medium, Large)`` stratum labels from ``(small_max, medium_max)``."""
  small_max, medium_max = validate_size_thresholds(size_thresholds)
  return (
    f"Small (n≤{small_max})",
    f"Medium ({small_max}<n≤{medium_max})",
    f"Large (n>{medium_max})",
  )


def categorize_by_size(
  problem_names: Sequence[str],
  problem_configs: Sequence[ProblemConfig],
  size_thresholds: tuple[int, int] = DEFAULT_SIZE_THRESHOLDS,
) -> dict[str, list[str]]:
  """Categorize problem names into ``Small`` / ``Medium`` / ``Large`` size strata."""
  small_max, medium_max = validate_size_thresholds(size_thresholds)
  small_label, medium_label, large_label = size_category_labels((small_max, medium_max))
  size_by_name = {cfg.name: int(cfg.size) for cfg in problem_configs}
  categories: dict[str, list[str]] = {
    small_label: [],
    medium_label: [],
    large_label: [],
  }

  for problem in problem_names:
    size = size_by_name.get(problem)
    if size is None:
      continue
    if size <= small_max:
      categories[small_label].append(problem)
    elif size <= medium_max:
      categories[medium_label].append(problem)
    else:
      categories[large_label].append(problem)

  return categories


def parse_report_artifacts(
  values: Sequence[str] | None,
  *,
  default: Sequence[ReportArtifact],
) -> tuple[ReportArtifact, ...]:
  """Normalize CLI/API ``--reports`` selections into an ordered deduplicated tuple.

  Accepts ``"none"``, ``"all"``, or any subset of ``ALL_REPORT_ARTIFACTS``.
  """
  if values is None:
    return tuple(default)

  normalized = [v.strip().lower() for v in values if v.strip()]
  if not normalized:
    return ()

  if "none" in normalized:
    if len(normalized) > 1:
      raise ValueError("Cannot combine 'none' with other --reports values.")
    return ()

  if "all" in normalized:
    unknown_extra = [v for v in normalized if v != "all" and v not in _VALID_ARTIFACT_SET]
    if unknown_extra:
      raise ValueError(
        f"Unsupported report artifact(s): {unknown_extra}. "
        f"Expected 'none', 'all', or a subset of {list(ALL_REPORT_ARTIFACTS)}."
      )
    return ALL_REPORT_ARTIFACTS

  unknown = [v for v in normalized if v not in _VALID_ARTIFACT_SET]
  if unknown:
    raise ValueError(
      f"Unsupported report artifact(s): {unknown}. Expected 'none', 'all', or a subset of {list(ALL_REPORT_ARTIFACTS)}."
    )

  ordered: list[ReportArtifact] = []
  for item in normalized:
    artifact = cast("ReportArtifact", item)
    if artifact not in ordered:
      ordered.append(artifact)
  return tuple(ordered)


@dataclass(frozen=True)
class ReportStyle:
  """Injectable presentation settings for tables, LaTeX macros, CD diagrams, and scaling plots.

  Defaults are strictly neutral English with standard decimal points, literal variant names,
  and unprefixed LaTeX table labels (``\\label{tab:summary}``).
  """

  display_names: Mapping[str, str] = field(default_factory=dict)
  decimal_separator: str = "."
  cd_decimal_sep: str = "."
  plot_decimal_comma: bool = False
  latex_label_prefix: str = ""
  size_thresholds: tuple[int, int] = DEFAULT_SIZE_THRESHOLDS
  show_size_guides: bool = True
  cd_quality_label: str = "Optimality Gap"
  cd_cost_label: str = "Mean Cost"
  cd_time_label: str = "Execution Time"
  cd_rank_label: str = "Average rank"
  plot_title: str = "Execution Time and Solution Quality Scaling vs. Problem Dimension"
  plot_time_ylabel: str = "Execution Time (s)"
  plot_gap_ylabel: str = "Optimality Gap (%)"
  plot_cost_ylabel: str = "Mean Cost"
  plot_xlabel: str = r"Problem Dimension ($n$)"
  plot_band_label: str | None = None
  algorithm_order: tuple[str, ...] | None = None

  def __post_init__(self) -> None:
    validated_thresholds = validate_size_thresholds(self.size_thresholds)
    object.__setattr__(self, "size_thresholds", validated_thresholds)
    object.__setattr__(self, "display_names", dict(self.display_names))
    if self.algorithm_order is not None:
      object.__setattr__(self, "algorithm_order", tuple(str(name) for name in self.algorithm_order))
    if not self.decimal_separator:
      raise ValueError("decimal_separator must be a non-empty string")
    if not self.cd_decimal_sep:
      raise ValueError("cd_decimal_sep must be a non-empty string")

  def display_name(self, algorithm: str) -> str:
    """Return the configured display label for ``algorithm`` or its canonical key."""
    return self.display_names.get(algorithm, algorithm)

  def table_label(self, suffix: str) -> str:
    """Build a LaTeX ``\\label{...}`` identifier with the configured optional prefix."""
    return f"tab:{self.latex_label_prefix}{suffix}"

  def size_labels(self) -> tuple[str, str, str]:
    """Return ``(Small, Medium, Large)`` stratum headings for this style's thresholds."""
    return size_category_labels(self.size_thresholds)


_STYLE_ALLOWED_KEYS: frozenset[str] = frozenset({f.name for f in fields(ReportStyle)} | {"$schema"})


def load_report_style(path: str | Path) -> ReportStyle:
  """Load a ``ReportStyle`` instance from a JSON configuration file."""
  style_path = Path(path)
  if not style_path.is_file():
    raise FileNotFoundError(f"Report style config not found: {style_path}")

  with style_path.open("r", encoding="utf-8") as fh:
    raw = json.load(fh)

  if not isinstance(raw, dict):
    raise ValueError(f"Report style JSON must be an object, got {type(raw).__name__}: {style_path}")

  unknown_keys = sorted(set(raw.keys()) - _STYLE_ALLOWED_KEYS)
  if unknown_keys:
    raise ValueError(f"Unknown key(s) in report style '{style_path}': {unknown_keys}")

  kwargs: dict[str, Any] = {k: v for k, v in raw.items() if k != "$schema"}
  if "display_names" in kwargs:
    if not isinstance(kwargs["display_names"], dict):
      raise ValueError("'display_names' must be a JSON object mapping variant keys to display labels")
    for k, v in kwargs["display_names"].items():
      if not isinstance(k, str) or not isinstance(v, str):
        raise ValueError("'display_names' entries must map string keys to string labels")
    kwargs["display_names"] = dict(kwargs["display_names"])

  for bool_key in ("plot_decimal_comma", "show_size_guides"):
    if bool_key in kwargs and not isinstance(kwargs[bool_key], bool):
      raise ValueError(f"'{bool_key}' must be a boolean, got {type(kwargs[bool_key]).__name__}")

  for str_key in (
    "decimal_separator",
    "cd_decimal_sep",
    "latex_label_prefix",
    "cd_quality_label",
    "cd_cost_label",
    "cd_time_label",
    "cd_rank_label",
    "plot_title",
    "plot_time_ylabel",
    "plot_gap_ylabel",
    "plot_cost_ylabel",
    "plot_xlabel",
  ):
    if str_key in kwargs and not isinstance(kwargs[str_key], str):
      raise ValueError(f"'{str_key}' must be a string, got {type(kwargs[str_key]).__name__}")

  if "plot_band_label" in kwargs and kwargs["plot_band_label"] is not None:
    if not isinstance(kwargs["plot_band_label"], str):
      raise ValueError(f"'plot_band_label' must be a string or null, got {type(kwargs['plot_band_label']).__name__}")

  if "size_thresholds" in kwargs:
    raw_thresholds = kwargs["size_thresholds"]
    if not isinstance(raw_thresholds, list) or len(raw_thresholds) != 2:
      raise ValueError("'size_thresholds' must be a 2-element JSON array [small_max, medium_max]")
    kwargs["size_thresholds"] = validate_size_thresholds(raw_thresholds)
  if "algorithm_order" in kwargs and kwargs["algorithm_order"] is not None:
    if not isinstance(kwargs["algorithm_order"], list) or not all(
      isinstance(x, str) for x in kwargs["algorithm_order"]
    ):
      raise ValueError("'algorithm_order' must be a JSON array of strings or null")
    kwargs["algorithm_order"] = tuple(kwargs["algorithm_order"])

  return ReportStyle(**kwargs)


__all__ = [
  "ALL_REPORT_ARTIFACTS",
  "DEFAULT_SIZE_THRESHOLDS",
  "ReportArtifact",
  "ReportStyle",
  "categorize_by_size",
  "escape_latex_identifier",
  "finite_metric",
  "load_report_style",
  "parse_report_artifacts",
  "size_category_labels",
  "validate_size_thresholds",
]
