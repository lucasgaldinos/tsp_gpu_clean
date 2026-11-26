"""Scaling and Quality Trade-Off Visualization Module (`DP13-9`, `DP14-3`, `DP14-5`).

Generates publication-ready dual-panel vector figures (PDF and SVG by default)
relating problem dimension ``n`` to execution time (logarithmic scale with
configurable empirical dispersion bands: IQR, ±1 SD, or 95% Student-t CI) and
percentage optimality gap relative to known TSPLIB optima.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy import stats

from benchmarking.analysis.cohort import resolve_algorithm_names
from benchmarking.reporting.style import DEFAULT_SIZE_THRESHOLDS, ReportStyle, validate_size_thresholds
from benchmarking.runner.checkpoint_io import load_checkpoints_dir

if TYPE_CHECKING:
  from collections.abc import Mapping, Sequence

  from matplotlib.axes import Axes
  from matplotlib.text import Annotation

  from benchmarking.runner.config_loader import ProblemConfig

logger = logging.getLogger(__name__)

DispersionMetric = Literal["iqr", "std", "ci"]
XAxisMode = Literal["continuous", "ordinal"]
XTickMode = Literal["instances", "auto"]

_VALID_DISPERSION_METRICS: frozenset[str] = frozenset({"iqr", "std", "ci"})
_VALID_X_AXIS_MODES: frozenset[str] = frozenset({"continuous", "ordinal"})
_VALID_X_TICK_MODES: frozenset[str] = frozenset({"instances", "auto"})

_DEFAULT_BAND_LABELS: dict[str, str] = {
  "iqr": "IQR ($Q_{25}$\u2013$Q_{75}$, median)",
  "std": "Mean ± 1 SD",
  "ci": "95% Student-t CI",
}

# ABNT A4 margins (3 cm left, 2 cm right) yield 160 mm (~6.30 in) printable text width.
TEXT_WIDTH_INCHES: float = 6.3
DEFAULT_FIGURE_HEIGHT_INCHES: float = 5.2

_PREFERRED_SERIF_FONT: str = "Libertinus Serif"
_FALLBACK_SERIF_FONT: str = "DejaVu Serif"

# Colorblind-safe Okabe-Ito palette with distinct markers and line styles so
# series remain distinguishable in grayscale printouts.
_PALETTE_COLORS: tuple[str, ...] = (
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#D55E00",
  "#CC79A7",
  "#56B4E9",
  "#000000",
  "#F0E442",
)
_PALETTE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
_PALETTE_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")


def _is_font_installed(font_name: str) -> bool:
  """Check whether ``font_name`` is available in Matplotlib's font registry."""
  return any(entry.name == font_name for entry in font_manager.fontManager.ttflist)


def _register_texlive_libertinus_fonts() -> None:
  """Register TeX Live Libertinus Serif fonts if present and not yet loaded."""
  font_dirs = (
    Path.home() / ".tex2026" / "texmf-dist" / "fonts" / "opentype" / "public" / "libertinus-fonts",
    Path.home() / ".tex2026" / "texmf-dist" / "fonts" / "truetype" / "public" / "libertinus-fonts",
  )
  registered_paths = {str(entry.fname) for entry in font_manager.fontManager.ttflist}
  for font_dir in font_dirs:
    if not font_dir.is_dir():
      continue
    for pattern in ("LibertinusSerif-*.otf", "LibertinusSerif-*.ttf"):
      for font_file in sorted(font_dir.glob(pattern)):
        font_str = str(font_file)
        if font_str not in registered_paths:
          font_manager.fontManager.addfont(font_str)
          registered_paths.add(font_str)


def get_academic_plot_rc_params() -> dict[str, Any]:
  """Return Matplotlib rcParams for publication-grade Type 42 vector output.

  Configures ``Libertinus Serif`` (auto-registering TeX Live font files when
  available) with fallback to ``DejaVu Serif``, TrueType font embedding
  (``pdf.fonttype = 42``, ``ps.fonttype = 42``) to eliminate Type 3 bitmap fonts,
  and matching mathtext font mappings without emitting ``findfont`` warnings when
  ``Libertinus Serif`` is not installed locally.
  """
  if not _is_font_installed(_PREFERRED_SERIF_FONT):
    _register_texlive_libertinus_fonts()
  math_family = _PREFERRED_SERIF_FONT if _is_font_installed(_PREFERRED_SERIF_FONT) else _FALLBACK_SERIF_FONT
  return {
    "font.family": "serif",
    "font.serif": [_PREFERRED_SERIF_FONT, _FALLBACK_SERIF_FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": math_family,
    "mathtext.it": f"{math_family}:italic",
    "mathtext.bf": f"{math_family}:bold",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
  }


def _resolve_style(
  alg_name: str,
  index: int,
  *,
  legend_labels: Mapping[str, str] | None = None,
  algorithm_styles: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
  """Return color, marker, linestyle, and legend label for an algorithm."""
  base = {
    "color": _PALETTE_COLORS[index % len(_PALETTE_COLORS)],
    "marker": _PALETTE_MARKERS[index % len(_PALETTE_MARKERS)],
    "linestyle": _PALETTE_LINESTYLES[index % len(_PALETTE_LINESTYLES)],
    "label": legend_labels[alg_name] if (legend_labels and alg_name in legend_labels) else alg_name,
  }
  if algorithm_styles and alg_name in algorithm_styles:
    override = algorithm_styles[alg_name]
    for key in ("color", "marker", "linestyle", "label"):
      if key in override:
        base[key] = override[key]
  return base


def _build_size_lookup(problem_configs: Sequence[ProblemConfig] | None) -> dict[str, int]:
  """Build a problem-name to problem-dimension lookup from configs."""
  if not problem_configs:
    return {}
  return {cfg.name: int(cfg.size) for cfg in problem_configs}


def _resolve_problem_size(
  problem_name: str,
  problem_entries: Mapping[str, Mapping[str, Any]],
  size_lookup: dict[str, int],
) -> int | None:
  """Resolve problem dimension ``n`` from configs or checkpoint metadata."""
  configured_size = size_lookup.get(problem_name)
  if configured_size is not None and configured_size > 0:
    return configured_size

  for entry in problem_entries.values():
    raw_size = entry.get("problem_size")
    if isinstance(raw_size, (int, float)) and not isinstance(raw_size, bool) and int(raw_size) > 0:
      return int(raw_size)
    meta = entry.get("problem_metadata")
    if isinstance(meta, dict):
      meta_size = meta.get("dimension") if meta.get("dimension") is not None else meta.get("size")
      if isinstance(meta_size, (int, float)) and not isinstance(meta_size, bool) and int(meta_size) > 0:
        return int(meta_size)
  return None


def _summarize_metric(
  entry: Mapping[str, Any],
  raw_key: str,
  mean_key: str,
  std_key: str,
  *,
  dispersion_metric: DispersionMetric = "ci",
) -> tuple[float, float, float] | None:
  """Return ``(center, band_low, band_high)`` according to ``dispersion_metric``.

  Supported dispersion metrics:
  - ``"iqr"``: empirical median and interquartile range ``(median, Q25, Q75)``;
  - ``"std"``: sample mean and ``±1 SD`` (Bessel-corrected ``ddof=1``);
  - ``"ci"``: sample mean and 95% Student-t confidence interval.
  """
  if dispersion_metric not in _VALID_DISPERSION_METRICS:
    raise ValueError(
      f"Unsupported dispersion_metric '{dispersion_metric}'. Expected one of {sorted(_VALID_DISPERSION_METRICS)}."
    )

  raw_values = entry.get(raw_key)
  if isinstance(raw_values, (list, tuple, np.ndarray)) and len(raw_values) > 0:
    arr = np.asarray([float(v) for v in raw_values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size >= 2:
      if dispersion_metric == "iqr":
        q25, median, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
        return float(median), float(q25), float(q75)
      mean = float(np.mean(arr))
      sd = float(np.std(arr, ddof=1))
      if dispersion_metric == "std":
        return mean, mean - sd, mean + sd
      half_ci = float(stats.t.ppf(0.975, df=int(arr.size) - 1) * (sd / np.sqrt(arr.size)))
      return mean, mean - half_ci, mean + half_ci
    if arr.size == 1:
      val = float(arr[0])
      return val, val, val

  raw_mean = entry.get(mean_key)
  if not isinstance(raw_mean, (int, float)) or isinstance(raw_mean, bool) or not np.isfinite(float(raw_mean)):
    return None

  mean = float(raw_mean)
  raw_std = entry.get(std_key)
  sd = (
    float(raw_std)
    if isinstance(raw_std, (int, float)) and not isinstance(raw_std, bool) and np.isfinite(float(raw_std))
    else 0.0
  )
  if sd <= 0.0:
    return mean, mean, mean

  if dispersion_metric == "iqr":
    # Normal-equivalent IQR half-width z_{0.75} * sd when only summary stats exist.
    half_iqr = 0.6744897501960817 * sd
    return mean, mean - half_iqr, mean + half_iqr
  if dispersion_metric == "std":
    return mean, mean - sd, mean + sd

  n_reps = entry.get("successful_runs") or entry.get("repetitions")
  if isinstance(n_reps, (int, float)) and not isinstance(n_reps, bool) and int(n_reps) >= 2:
    half_ci = float(stats.t.ppf(0.975, df=int(n_reps) - 1) * (sd / np.sqrt(int(n_reps))))
  else:
    half_ci = sd
  return mean, mean - half_ci, mean + half_ci


def _collect_algorithm_series(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig] | None,
  *,
  algorithm_order: Sequence[str] | None = None,
  dispersion_metric: DispersionMetric = "ci",
) -> dict[str, list[dict[str, Any]]]:
  """Collect sorted per-algorithm scaling series from ``all_results``."""
  size_lookup = _build_size_lookup(problem_configs)
  ordered_algs = resolve_algorithm_names(all_results, algorithm_order)
  series_by_alg: dict[str, list[dict[str, Any]]] = {alg: [] for alg in ordered_algs}

  for problem_name, prob_entries in all_results.items():
    size = _resolve_problem_size(problem_name, prob_entries, size_lookup)
    if size is None or size <= 0:
      continue

    for alg_name, entry in prob_entries.items():
      time_stats = _summarize_metric(
        entry,
        "raw_times",
        "mean_time",
        "std_time",
        dispersion_metric=dispersion_metric,
      )
      gap_stats = _summarize_metric(
        entry,
        "raw_gaps",
        "mean_gap",
        "std_gap",
        dispersion_metric=dispersion_metric,
      )
      cost_stats = _summarize_metric(
        entry,
        "raw_costs",
        "mean_cost",
        "std_cost",
        dispersion_metric=dispersion_metric,
      )
      if time_stats is None and gap_stats is None and cost_stats is None:
        continue

      point: dict[str, Any] = {"problem": problem_name, "n": float(size)}
      if time_stats is not None:
        t_mean, t_lo, t_hi = time_stats
        point["time_mean"] = t_mean
        point["time_lo"] = t_lo
        point["time_hi"] = t_hi
      if gap_stats is not None:
        g_mean, g_lo, g_hi = gap_stats
        point["gap_mean"] = g_mean
        point["gap_lo"] = g_lo
        point["gap_hi"] = g_hi
      if cost_stats is not None:
        c_mean, c_lo, c_hi = cost_stats
        point["cost_mean"] = c_mean
        point["cost_lo"] = c_lo
        point["cost_hi"] = c_hi

      series_by_alg[alg_name].append(point)

  for alg_points in series_by_alg.values():
    alg_points.sort(key=lambda pt: (float(pt["n"]), str(pt["problem"])))

  return {alg: pts for alg, pts in series_by_alg.items() if pts}


def _collect_ordinal_slots(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig] | None,
) -> list[tuple[int, str]]:
  """Return ordered ``(size, problem_name)`` slots for ordinal axis rendering."""
  size_lookup = _build_size_lookup(problem_configs)
  slots: list[tuple[int, str]] = []
  for problem_name, prob_entries in all_results.items():
    size = _resolve_problem_size(problem_name, prob_entries, size_lookup)
    if size is not None and size > 0:
      slots.append((size, problem_name))
  slots.sort(key=lambda item: (item[0], item[1]))
  return slots


def _pava_min_spacing(positions: list[float], min_spacing: float) -> list[float]:
  """Project sorted 1D coordinates onto ``x_{i+1} - x_i >= min_spacing`` with minimal L2 shift."""
  if len(positions) <= 1:
    return list(positions)

  shifted = [positions[i] - i * min_spacing for i in range(len(positions))]
  blocks: list[list[float]] = [[val, 1.0] for val in shifted]
  idx = 0
  while idx < len(blocks) - 1:
    if blocks[idx][0] > blocks[idx + 1][0]:
      total_w = blocks[idx][1] + blocks[idx + 1][1]
      avg = (blocks[idx][0] * blocks[idx][1] + blocks[idx + 1][0] * blocks[idx + 1][1]) / total_w
      blocks[idx] = [avg, total_w]
      blocks.pop(idx + 1)
      if idx > 0:
        idx -= 1
    else:
      idx += 1

  expanded: list[float] = []
  for val, cnt in blocks:
    expanded.extend([val] * int(cnt))
  return [expanded[i] + i * min_spacing for i in range(len(positions))]


def _apply_continuous_instance_ticks(
  ax_gap: Axes,
  fig: Figure,
  canvas: Any,
  unique_ns: list[int],
) -> list[Annotation]:
  """Place exact continuous ticks at ``unique_ns`` with rotated + two-tier staggered collision-free labels."""
  ax_gap.set_xticks(unique_ns)
  ax_gap.set_xticklabels([])
  ax_gap.tick_params(axis="x", which="major", length=3.0, pad=2)

  if not unique_ns:
    return []

  if len(unique_ns) == 1:
    ann = ax_gap.annotate(
      str(unique_ns[0]),
      xy=(unique_ns[0], 0.0),
      xycoords=("data", "axes fraction"),
      xytext=(0.0, -5.0),
      textcoords="offset points",
      ha="center",
      va="top",
      fontsize=7.2,
      annotation_clip=False,
    )
    return [ann]

  canvas.draw()
  renderer = canvas.get_renderer()
  ax_bbox = ax_gap.get_window_extent(renderer)
  x_min, x_max = ax_gap.get_xlim()
  span = x_max - x_min
  if span <= 0.0:
    return []

  pts_per_unit = (ax_bbox.width * 72.0 / fig.dpi) / span
  orig_pts = [float(n) * pts_per_unit for n in unique_ns]

  # Measure actual rendered label dimensions under the active font (Libertinus
  # or DejaVu fallback) so 45-deg rotated boxes never collide regardless of
  # digit count or figure width.
  tick_fontsize = 7.0
  fp = FontProperties(size=tick_fontsize)
  scale_pt = 72.0 / fig.dpi
  max_h_pt = 0.0
  max_diag_pt = 0.0
  for n in unique_ns:
    w_px, h_px, _ = renderer.get_text_width_height_descent(str(n), fp, ismath=False)
    w_pt = float(w_px) * scale_pt
    h_pt = float(h_px) * scale_pt
    max_h_pt = max(max_h_pt, h_pt)
    max_diag_pt = max(max_diag_pt, (w_pt + h_pt) / math.sqrt(2.0))

  same_tier_min_pts = max(12.0, max_h_pt * math.sqrt(2.0) + 4.0)
  min_adj_pts = max(9.0, 0.55 * same_tier_min_pts)
  tier_y_pts = (-5.0, -(5.0 + max(14.5, max_diag_pt + 2.5)))

  left_bound_pts = x_min * pts_per_unit + 2.0
  right_bound_pts = x_max * pts_per_unit - 2.0
  num_ticks = len(unique_ns)
  available_pts = right_bound_pts - left_bound_pts
  if (num_ticks - 1) * min_adj_pts <= available_pts:
    placed_pts = _pava_min_spacing(orig_pts, min_adj_pts)
    upper_q = right_bound_pts - (num_ticks - 1) * min_adj_pts
    placed_pts = [
      min(max(placed_pts[idx] - idx * min_adj_pts, left_bound_pts), upper_q) + idx * min_adj_pts
      for idx in range(num_ticks)
    ]
  else:
    step = available_pts / max(1, num_ticks - 1) if available_pts > 0.0 else min_adj_pts
    placed_pts = [left_bound_pts + idx * step for idx in range(num_ticks)]

  # Assign tiers on the final placed horizontal coordinates so PAVA shifts
  # never place two insufficiently separated labels on the same vertical tier.
  tiers = [0] * num_ticks
  for idx in range(1, num_ticks):
    if placed_pts[idx] - placed_pts[idx - 1] < same_tier_min_pts:
      tiers[idx] = 1 - tiers[idx - 1]

  annotations: list[Annotation] = []
  for idx, n in enumerate(unique_ns):
    dx_pts = placed_pts[idx] - orig_pts[idx]
    dy_pts = tier_y_pts[tiers[idx]]
    moved = abs(dx_pts) > 1.0 or tiers[idx] == 1
    ann = ax_gap.annotate(
      str(n),
      xy=(n, 0.0),
      xycoords=("data", "axes fraction"),
      xytext=(dx_pts, dy_pts),
      textcoords="offset points",
      ha="right",
      va="top",
      rotation=45,
      rotation_mode="anchor",
      fontsize=tick_fontsize,
      arrowprops=({"arrowstyle": "-", "color": "#777777", "lw": 0.55, "shrinkA": 0, "shrinkB": 0} if moved else None),
      annotation_clip=False,
    )
    annotations.append(ann)

  canvas.draw()
  renderer = canvas.get_renderer()
  min_y_px = min(ann.get_window_extent(renderer).y0 for ann in annotations)
  ax_bottom_px = ax_gap.get_window_extent(renderer).y0
  needed_pad_pts = max(6.0, (ax_bottom_px - min_y_px) * 72.0 / fig.dpi + 6.0)
  ax_gap.xaxis.labelpad = needed_pad_pts
  canvas.draw()
  return annotations


def _draw_stratum_guides(
  ax: Axes,
  x_max: float,
  *,
  annotate: bool,
  ordinal_sizes: Sequence[int] | None = None,
  size_thresholds: tuple[int, int] = DEFAULT_SIZE_THRESHOLDS,
) -> None:
  """Draw subtle vertical reference lines for configured size strata when in range."""
  small_max, medium_max = validate_size_thresholds(size_thresholds)
  guide_specs = (
    (small_max, f"n = {small_max}"),
    (medium_max, f"n = {medium_max}"),
  )
  if ordinal_sizes is not None:
    for threshold, label in guide_specs:
      boundary_idx: float | None = None
      for idx in range(len(ordinal_sizes) - 1):
        if ordinal_sizes[idx] <= threshold < ordinal_sizes[idx + 1]:
          boundary_idx = idx + 0.5
          break
      if boundary_idx is not None:
        ax.axvline(boundary_idx, color="#7F7F7F", linestyle=":", linewidth=0.9, alpha=0.65, zorder=1)
        if annotate:
          ax.text(
            boundary_idx,
            0.98,
            f" {label}",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="left",
            fontsize=8,
            color="#555555",
          )
    return

  for threshold, label in guide_specs:
    if threshold <= x_max * 1.05:
      ax.axvline(threshold, color="#7F7F7F", linestyle=":", linewidth=0.9, alpha=0.65, zorder=1)
      if annotate:
        ax.text(
          threshold,
          0.98,
          f" {label}",
          transform=ax.get_xaxis_transform(),
          va="top",
          ha="left",
          fontsize=8,
          color="#555555",
        )


def generate_scaling_plots(
  all_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
  problem_configs: Sequence[ProblemConfig] | None,
  output_dir: Path | str,
  filename_stem: str = "scaling_time_gap",
  *,
  style: ReportStyle | None = None,
  title: str | None = None,
  time_ylabel: str | None = None,
  gap_ylabel: str | None = None,
  cost_ylabel: str | None = None,
  xlabel: str | None = None,
  band_label: str | None = None,
  legend_labels: Mapping[str, str] | None = None,
  algorithm_order: Sequence[str] | None = None,
  algorithm_styles: Mapping[str, Mapping[str, str]] | None = None,
  formats: Sequence[str] = ("pdf", "svg"),
  figure_width: float = TEXT_WIDTH_INCHES,
  figure_height: float = DEFAULT_FIGURE_HEIGHT_INCHES,
  show_size_guides: bool | None = None,
  size_thresholds: tuple[int, int] | None = None,
  dispersion_metric: DispersionMetric = "iqr",
  x_axis_mode: XAxisMode = "continuous",
  x_tick_mode: XTickMode = "instances",
  decimal_comma: bool | None = None,
) -> tuple[Path, ...]:
  """Generate dual-panel Time x Dimension and Gap x Dimension scaling plots.

  Top panel renders execution time (seconds) on a logarithmic y-axis with
  empirical dispersion bands (``"iqr"``, ``"std"``, or ``"ci"``) across
  repetitions. Bottom panel renders percentage optimality gap relative to the
  TSPLIB optimum on a linear y-axis (or raw mean cost when optima are absent).
  Presentation options default to ``style`` (or neutral ``ReportStyle()``) and
  may be individually overridden via keyword arguments.

  Returns:
      Tuple of written figure paths corresponding to ``formats``.
  """
  if not all_results:
    raise ValueError("all_results dictionary is empty")
  if not formats:
    raise ValueError("formats must contain at least one output format")
  if dispersion_metric not in _VALID_DISPERSION_METRICS:
    raise ValueError(
      f"Unsupported dispersion_metric '{dispersion_metric}'. Expected one of {sorted(_VALID_DISPERSION_METRICS)}."
    )
  if x_axis_mode not in _VALID_X_AXIS_MODES:
    raise ValueError(f"Unsupported x_axis_mode '{x_axis_mode}'. Expected one of {sorted(_VALID_X_AXIS_MODES)}.")
  if x_tick_mode not in _VALID_X_TICK_MODES:
    raise ValueError(f"Unsupported x_tick_mode '{x_tick_mode}'. Expected one of {sorted(_VALID_X_TICK_MODES)}.")

  base_style = style or ReportStyle()
  resolved_title = base_style.plot_title if title is None else title
  resolved_time_ylabel = base_style.plot_time_ylabel if time_ylabel is None else time_ylabel
  resolved_gap_ylabel = base_style.plot_gap_ylabel if gap_ylabel is None else gap_ylabel
  resolved_cost_ylabel = base_style.plot_cost_ylabel if cost_ylabel is None else cost_ylabel
  resolved_xlabel = base_style.plot_xlabel if xlabel is None else xlabel
  resolved_band_label = base_style.plot_band_label if band_label is None else band_label
  resolved_legend_labels = (base_style.display_names or None) if legend_labels is None else legend_labels
  resolved_algorithm_order = base_style.algorithm_order if algorithm_order is None else algorithm_order
  resolved_show_size_guides = base_style.show_size_guides if show_size_guides is None else show_size_guides
  resolved_size_thresholds = base_style.size_thresholds if size_thresholds is None else size_thresholds
  resolved_decimal_comma = base_style.plot_decimal_comma if decimal_comma is None else decimal_comma

  normalized_formats = [fmt.lower().lstrip(".") for fmt in formats]
  supported_formats = {"pdf", "svg", "png"}
  unsupported = [fmt for fmt in normalized_formats if fmt not in supported_formats]
  if unsupported:
    raise ValueError(
      f"Unsupported plot output format(s): {unsupported}. Expected subset of {sorted(supported_formats)}."
    )

  series_by_alg = _collect_algorithm_series(
    all_results,
    problem_configs,
    algorithm_order=resolved_algorithm_order,
    dispersion_metric=dispersion_metric,
  )
  if not series_by_alg:
    raise ValueError("No plottable (problem, algorithm) scaling data found in all_results")

  plottable_problems = {str(pt["problem"]) for pts in series_by_alg.values() for pt in pts}
  ordinal_slots: list[tuple[int, str]] = []
  ordinal_x_by_problem: dict[str, float] = {}
  if x_axis_mode == "ordinal":
    ordinal_slots = [
      slot for slot in _collect_ordinal_slots(all_results, problem_configs) if slot[1] in plottable_problems
    ]
    for idx, (_, slot_problem) in enumerate(ordinal_slots):
      ordinal_x_by_problem[slot_problem] = float(idx)

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  output_paths = tuple(out_dir / f"{filename_stem}.{fmt}" for fmt in normalized_formats)

  x_values = [float(pt["n"]) for pts in series_by_alg.values() for pt in pts]
  unique_ns = sorted({round(val) for val in x_values})
  x_min = min(x_values)
  x_max = max(x_values)
  effective_band_label = _DEFAULT_BAND_LABELS[dispersion_metric] if resolved_band_label is None else resolved_band_label

  with mpl.rc_context(cast("Any", get_academic_plot_rc_params())):
    fig = Figure(figsize=(figure_width, figure_height), dpi=300)
    canvas: Any = FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.14})
    ax_time: Axes = axes[0]
    ax_gap: Axes = axes[1]

    has_positive_times = False
    has_band = False
    all_points = [pt for pts in series_by_alg.values() for pt in pts]
    has_all_gaps = bool(all_points) and all("gap_mean" in pt for pt in all_points)
    quality_prefix = "gap" if has_all_gaps else "cost"
    effective_gap_ylabel = resolved_gap_ylabel if has_all_gaps else resolved_cost_ylabel

    def _map_xs(pts: list[dict[str, Any]]) -> np.ndarray:
      if x_axis_mode == "ordinal":
        return np.asarray([ordinal_x_by_problem[str(pt["problem"])] for pt in pts], dtype=float)
      return np.asarray([float(pt["n"]) for pt in pts], dtype=float)

    for style_idx, (alg_name, pts) in enumerate(series_by_alg.items()):
      series_style = _resolve_style(
        alg_name,
        style_idx,
        legend_labels=resolved_legend_labels,
        algorithm_styles=algorithm_styles,
      )

      time_pts = [pt for pt in pts if "time_mean" in pt and pt["time_mean"] > 0.0]
      if time_pts:
        has_positive_times = True
        xs_t = _map_xs(time_pts)
        ys_t = np.asarray([pt["time_mean"] for pt in time_pts], dtype=float)
        # Keep lower bound strictly positive so log-scale fill_between never emits warnings.
        lo_t = np.maximum(np.asarray([pt["time_lo"] for pt in time_pts], dtype=float), np.maximum(ys_t * 0.1, 1e-9))
        hi_t = np.maximum(np.asarray([pt["time_hi"] for pt in time_pts], dtype=float), ys_t)

        ax_time.plot(
          xs_t,
          ys_t,
          color=series_style["color"],
          marker=series_style["marker"],
          linestyle=series_style["linestyle"],
          linewidth=1.6,
          markersize=4.5,
          label=series_style["label"],
          zorder=3,
        )
        if xs_t.size >= 2:
          has_band = True
          ax_time.fill_between(xs_t, lo_t, hi_t, color=series_style["color"], alpha=0.16, linewidth=0.0, zorder=2)

      mean_key = f"{quality_prefix}_mean"
      lo_key = f"{quality_prefix}_lo"
      hi_key = f"{quality_prefix}_hi"
      gap_pts = [pt for pt in pts if mean_key in pt]
      if gap_pts:
        xs_g = _map_xs(gap_pts)
        ys_g = np.asarray([pt[mean_key] for pt in gap_pts], dtype=float)
        lo_g = np.asarray([pt[lo_key] for pt in gap_pts], dtype=float)
        if np.all(ys_g >= 0.0):
          lo_g = np.maximum(lo_g, 0.0)
        hi_g = np.asarray([pt[hi_key] for pt in gap_pts], dtype=float)

        ax_gap.plot(
          xs_g,
          ys_g,
          color=series_style["color"],
          marker=series_style["marker"],
          linestyle=series_style["linestyle"],
          linewidth=1.6,
          markersize=4.5,
          label=series_style["label"] if not time_pts else None,
          zorder=3,
        )
        if xs_g.size >= 2:
          has_band = True
          ax_gap.fill_between(xs_g, lo_g, hi_g, color=series_style["color"], alpha=0.16, linewidth=0.0, zorder=2)

    if has_positive_times:
      ax_time.set_yscale("log")

    if x_axis_mode == "continuous":
      if x_max > x_min:
        x_pad = max(15.0, 0.045 * (x_max - x_min))
        ax_gap.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
      else:
        ax_gap.set_xlim(max(0.0, x_min - 10.0), x_max + 10.0)
    elif ordinal_slots:
      ax_gap.set_xlim(-0.5, len(ordinal_slots) - 0.5)

    if resolved_show_size_guides:
      ordinal_sizes = [slot[0] for slot in ordinal_slots] if x_axis_mode == "ordinal" else None
      _draw_stratum_guides(
        ax_time,
        x_max,
        annotate=True,
        ordinal_sizes=ordinal_sizes,
        size_thresholds=resolved_size_thresholds,
      )
      _draw_stratum_guides(
        ax_gap,
        x_max,
        annotate=False,
        ordinal_sizes=ordinal_sizes,
        size_thresholds=resolved_size_thresholds,
      )

    ax_time.set_ylabel(resolved_time_ylabel)
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)

    handles, labels = ax_time.get_legend_handles_labels()
    if not handles:
      handles, labels = ax_gap.get_legend_handles_labels()
    if effective_band_label and has_band:
      handles.append(Patch(facecolor="#666666", alpha=0.22, edgecolor="none"))
      labels.append(effective_band_label)

    legend_obj = None
    if handles:
      ncol = 2 if len(handles) <= 4 else 3
      legend_obj = ax_time.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
      )

    ax_gap.set_ylabel(effective_gap_ylabel)
    ax_gap.set_xlabel(resolved_xlabel)
    if resolved_decimal_comma:
      ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.1f}".replace(".", ",")))
    ax_gap.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)

    suptitle_obj = fig.suptitle(resolved_title, y=0.98, fontsize=10) if resolved_title else None

    bottom_margin = 0.16 if (x_axis_mode == "ordinal" or x_tick_mode == "instances") else 0.11

    # Reserve vertical space between the legend above `ax_time` and `fig.suptitle`
    # so multi-row legends never collide with the figure title.
    if legend_obj is not None and suptitle_obj is not None:
      top_margin = 0.80
      fig.subplots_adjust(top=top_margin, bottom=bottom_margin)
      canvas.draw()
      renderer = canvas.get_renderer()
      inv_fig = fig.transFigure.inverted()
      leg_bbox = legend_obj.get_window_extent(renderer).transformed(inv_fig)
      title_bbox = suptitle_obj.get_window_extent(renderer).transformed(inv_fig)
      min_gap = 0.02
      if leg_bbox.y1 > title_bbox.y0 - min_gap:
        shift = (leg_bbox.y1 - title_bbox.y0) + min_gap + 0.005
        fig.subplots_adjust(top=max(0.55, top_margin - shift), bottom=bottom_margin)
        canvas.draw()
    elif legend_obj is not None:
      fig.subplots_adjust(top=0.84, bottom=bottom_margin)
    else:
      fig.subplots_adjust(bottom=bottom_margin)

    if x_axis_mode == "ordinal" and ordinal_slots:
      ax_gap.set_xticks(list(range(len(ordinal_slots))))
      ax_gap.set_xticklabels(
        [f"{prob_name} ({prob_size})" for prob_size, prob_name in ordinal_slots],
        rotation=45,
        ha="right",
        fontsize=7.2,
      )
      canvas.draw()
    elif x_axis_mode == "continuous" and x_tick_mode == "instances":
      _apply_continuous_instance_ticks(ax_gap, fig, canvas, unique_ns)

    for fmt, out_path in zip(normalized_formats, output_paths, strict=True):
      save_kwargs: dict[str, Any] = {"format": fmt, "bbox_inches": "tight"}
      if fmt == "png":
        save_kwargs["dpi"] = 300
      fig.savefig(out_path, **save_kwargs)

  logger.info("Generated scaling plots: %s", ", ".join(str(p) for p in output_paths))
  return output_paths


def generate_scaling_plots_from_checkpoints(
  checkpoint_dir: Path | str,
  output_dir: Path | str,
  problem_configs: Sequence[ProblemConfig] | None = None,
  filename_stem: str = "scaling_time_gap",
  *,
  style: ReportStyle | None = None,
  title: str | None = None,
  time_ylabel: str | None = None,
  gap_ylabel: str | None = None,
  cost_ylabel: str | None = None,
  xlabel: str | None = None,
  band_label: str | None = None,
  legend_labels: Mapping[str, str] | None = None,
  algorithm_order: Sequence[str] | None = None,
  algorithm_styles: Mapping[str, Mapping[str, str]] | None = None,
  formats: Sequence[str] = ("pdf", "svg"),
  figure_width: float = TEXT_WIDTH_INCHES,
  figure_height: float = DEFAULT_FIGURE_HEIGHT_INCHES,
  show_size_guides: bool | None = None,
  size_thresholds: tuple[int, int] | None = None,
  dispersion_metric: DispersionMetric = "iqr",
  x_axis_mode: XAxisMode = "continuous",
  x_tick_mode: XTickMode = "instances",
  decimal_comma: bool | None = None,
) -> tuple[Path, ...]:
  """Load checkpoints read-only and write scaling plots to ``output_dir``."""
  all_results = load_checkpoints_dir(checkpoint_dir)
  return generate_scaling_plots(
    all_results=all_results,
    problem_configs=problem_configs,
    output_dir=output_dir,
    filename_stem=filename_stem,
    style=style,
    title=title,
    time_ylabel=time_ylabel,
    gap_ylabel=gap_ylabel,
    cost_ylabel=cost_ylabel,
    xlabel=xlabel,
    band_label=band_label,
    legend_labels=legend_labels,
    algorithm_order=algorithm_order,
    algorithm_styles=algorithm_styles,
    formats=formats,
    figure_width=figure_width,
    figure_height=figure_height,
    show_size_guides=show_size_guides,
    size_thresholds=size_thresholds,
    dispersion_metric=dispersion_metric,
    x_axis_mode=x_axis_mode,
    x_tick_mode=x_tick_mode,
    decimal_comma=decimal_comma,
  )
