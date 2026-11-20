"""Unit tests for dual-panel scaling and trade-off visualization (`DP13-9`, `DP14-3`)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from benchmarking.analysis.report_generator import generate_result_tables
from benchmarking.analysis.scaling_plots import (
  TEXT_WIDTH_INCHES,
  _collect_algorithm_series,
  _summarize_metric,
  generate_scaling_plots,
  generate_scaling_plots_from_checkpoints,
  get_academic_plot_rc_params,
  load_results_from_checkpoints,
)
from benchmarking.runner.config_loader import ProblemConfig

if TYPE_CHECKING:
  from pathlib import Path


def _build_synthetic_results(
  *, include_large_cpu: bool = False
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[ProblemConfig]]:
  """Build synthetic benchmark results spanning small, medium, and large strata."""
  specs = [
    ("att48", 48, True),
    ("kroA100", 100, True),
    ("lin318", 318, True),
    ("pcb442", 442, True),
    ("rat783", 783, include_large_cpu),
    ("pr1002", 1002, include_large_cpu),
  ]
  configs = [ProblemConfig(name=name, optimal=1000.0, size=size) for name, size, _ in specs]
  all_results: dict[str, dict[str, dict[str, Any]]] = {}

  for name, size, has_cpu in specs:
    scale = (size / 50.0) ** 2
    entries: dict[str, dict[str, Any]] = {
      "HybridNaive": {
        "problem_size": size,
        "raw_times": [0.2 * scale, 0.22 * scale, 0.21 * scale],
        "raw_gaps": [0.4, 0.6, 0.5],
      },
      "HybridOptimized": {
        "problem_size": size,
        "raw_times": [0.08 * scale, 0.09 * scale, 0.085 * scale],
        "raw_gaps": [0.5, 0.7, 0.6],
      },
      "FullGPUModular": {
        "problem_size": size,
        "raw_times": [0.12 * scale, 0.13 * scale, 0.125 * scale],
        "raw_gaps": [0.3, 0.4, 0.35],
      },
    }
    if has_cpu:
      entries["CPU"] = {
        "problem_size": size,
        "raw_times": [2.0 * scale, 2.3 * scale, 2.15 * scale],
        "raw_gaps": [0.4, 0.5, 0.45],
      }
    all_results[name] = entries

  return all_results, configs


def test_generate_scaling_plots_writes_valid_pdf_and_svg_with_partial_cpu(tmp_path: Path) -> None:
  """WHAT: Generate scaling_time_gap.{pdf,svg} with partial CPU coverage (4/6 instances).

  WHY: Task 14.3 (DP14-3) — Default formats are vector PDF and SVG (never PNG),
  supporting partial cohorts without requiring every algorithm to cover all instances.
  EXPECTED: Non-empty valid PDF (%PDF) and SVG (<svg) files written to tmp_path; no PNG;
  CPU series has 4 points while GPU series each have 6 sorted points.
  """
  all_results, configs = _build_synthetic_results(include_large_cpu=False)

  series = _collect_algorithm_series(all_results, configs)
  assert [pt["n"] for pt in series["CPU"]] == [48.0, 100.0, 318.0, 442.0]
  assert [pt["n"] for pt in series["FullGPUModular"]] == [48.0, 100.0, 318.0, 442.0, 783.0, 1002.0]

  pdf_path, svg_path = generate_scaling_plots(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tmp_path,
  )

  assert pdf_path == tmp_path / "scaling_time_gap.pdf"
  assert svg_path == tmp_path / "scaling_time_gap.svg"
  assert not (tmp_path / "scaling_time_gap.png").exists()
  assert pdf_path.is_file() and pdf_path.stat().st_size > 1000
  assert svg_path.is_file() and svg_path.stat().st_size > 1000
  assert pdf_path.read_bytes()[:4] == b"%PDF"
  assert b"<svg" in svg_path.read_bytes()[:512]


def test_generate_scaling_plots_summary_only_fallback_and_report_integration(tmp_path: Path) -> None:
  """WHAT: Summary-only metrics (mean/std without raw_*) and report_generator integration.

  WHY: Callers passing aggregated dicts or tables output_dir='.../tables' must
  generate vector figures in sibling 'figures/' directory without PNGs.
  EXPECTED: tables_dir sibling 'figures/scaling_time_gap.{pdf,svg}' created and valid.
  """
  all_results: dict[str, dict[str, dict[str, Any]]] = {
    "berlin52": {
      "CPU": {
        "mean_time": 1.5,
        "std_time": 0.2,
        "mean_cost": 7551.0,
        "std_cost": 10.0,
        "mean_gap": 0.12,
        "std_gap": 0.05,
        "successful_runs": 10,
      },
      "FullGPUModular": {
        "mean_time": 0.2,
        "std_time": 0.03,
        "mean_cost": 7553.0,
        "std_cost": 8.0,
        "mean_gap": 0.15,
        "std_gap": 0.04,
        "successful_runs": 10,
      },
    },
    "kroA100": {
      "CPU": {
        "mean_time": 8.0,
        "std_time": 0.9,
        "mean_cost": 21367.0,
        "std_cost": 25.0,
        "mean_gap": 0.40,
        "std_gap": 0.10,
        "successful_runs": 10,
      },
      "FullGPUModular": {
        "mean_time": 0.5,
        "std_time": 0.05,
        "mean_cost": 21356.0,
        "std_cost": 20.0,
        "mean_gap": 0.35,
        "std_gap": 0.08,
        "successful_runs": 10,
      },
    },
  }
  configs = [
    ProblemConfig(name="berlin52", optimal=7542.0, size=52),
    ProblemConfig(name="kroA100", optimal=21282.0, size=100),
  ]

  tables_dir = tmp_path / "tables"
  generate_result_tables(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tables_dir,
    benchmark_name="scaling_integration",
    baseline_algorithm="CPU",
  )

  figures_dir = tmp_path / "figures"
  assert (figures_dir / "scaling_time_gap.pdf").read_bytes()[:4] == b"%PDF"
  assert b"<svg" in (figures_dir / "scaling_time_gap.svg").read_bytes()[:512]
  assert not (figures_dir / "scaling_time_gap.png").exists()


def test_load_and_generate_from_checkpoints_dir(tmp_path: Path) -> None:
  """WHAT: Read-only checkpoint loading resolves sizes without ProblemConfig and skips corrupt files."""
  ckpt_dir = tmp_path / "checkpoints"
  ckpt_dir.mkdir()
  out_dir = tmp_path / "figures"

  payloads = [
    {
      "problem_name": "att48",
      "problem_size": 48,
      "algorithm": "CPU",
      "raw_times": [1.0, 1.2, 1.1],
      "raw_gaps": [0.1, 0.2, 0.15],
    },
    {
      "problem_name": "lin318",
      "problem_size": 318,
      "algorithm": "FullGPUModular",
      "raw_times": [0.8, 0.9, 0.85],
      "raw_gaps": [0.5, 0.6, 0.55],
    },
  ]
  for idx, payload in enumerate(payloads):
    (ckpt_dir / f"ckpt_{idx}.json").write_text(json.dumps(payload), encoding="utf-8")

  # Malformed checkpoint should be skipped gracefully
  (ckpt_dir / "corrupt.json").write_text("{not valid json", encoding="utf-8")

  loaded = load_results_from_checkpoints(ckpt_dir)
  assert set(loaded.keys()) == {"att48", "lin318"}

  pdf_path, svg_path = generate_scaling_plots_from_checkpoints(ckpt_dir, out_dir)
  assert pdf_path.read_bytes()[:4] == b"%PDF"
  assert b"<svg" in svg_path.read_bytes()[:512]
  assert not (out_dir / "scaling_time_gap.png").exists()


def test_academic_plot_rc_params_abnt_width_and_custom_labels(tmp_path: Path) -> None:
  """WHAT: Verify Type 42 font embedding, ABNT width (6.3 in), Libertinus Serif config, and parameterized labels."""
  assert TEXT_WIDTH_INCHES == pytest.approx(6.3)
  rc = get_academic_plot_rc_params()
  assert rc["pdf.fonttype"] == 42
  assert rc["ps.fonttype"] == 42
  assert rc["font.family"] == "serif"
  assert rc["font.serif"][0] == "Libertinus Serif"
  assert rc["mathtext.fontset"] == "custom"

  all_results, configs = _build_synthetic_results(include_large_cpu=True)
  (pdf_path,) = generate_scaling_plots(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tmp_path,
    filename_stem="custom_ptbr",
    title="Escalonamento Teste",
    time_ylabel="Tempo (s)",
    gap_ylabel="Gap (%)",
    xlabel=r"Dimensão ($n$)",
    band_label="IC 95%",
    legend_labels={"CPU": "CPU", "FullGPUModular": "Totalmente em GPU"},
    algorithm_order=("CPU", "HybridNaive", "HybridOptimized", "FullGPUModular"),
    formats=("pdf",),
  )
  assert pdf_path.is_file()
  assert not (tmp_path / "custom_ptbr.svg").exists()
  assert not (tmp_path / "custom_ptbr.png").exists()

  pdffonts_bin = shutil.which("pdffonts")
  if pdffonts_bin is not None:
    fonts_out = subprocess.check_output([pdffonts_bin, str(pdf_path)], text=True)  # noqa: S603
    assert "Type 3" not in fonts_out


def test_summarize_metric_and_error_paths(tmp_path: Path) -> None:
  """WHAT: Validate 95% CI calculation and explicit ValueError on invalid inputs."""
  stat = _summarize_metric({"raw_times": [10.0, 12.0, 14.0]}, "raw_times", "mean_time", "std_time")
  assert stat is not None
  mean, lo, hi = stat
  assert mean == pytest.approx(12.0)
  assert lo < mean < hi

  with pytest.raises(ValueError, match="empty"):
    generate_scaling_plots({}, [], tmp_path)

  with pytest.raises(ValueError, match="No plottable"):
    generate_scaling_plots({"unknown": {"CPU": {"mean_time": 1.0}}}, [], tmp_path)

  all_results, configs = _build_synthetic_results(include_large_cpu=True)
  with pytest.raises(ValueError, match="at least one output format"):
    generate_scaling_plots(all_results, configs, tmp_path, formats=())

  with pytest.raises(ValueError, match="Unsupported plot output format"):
    generate_scaling_plots(all_results, configs, tmp_path, formats=("bmp",))


def test_scaling_plot_title_and_legend_never_overlap_and_libertinus_path(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: Multi-row pt-BR legend and long figure title never overlap vertically, and Libertinus path maps mathtext."""
  from matplotlib.figure import Figure

  import benchmarking.analysis.scaling_plots as sp

  # 1. Verify Libertinus Serif branch when font is reported as installed
  monkeypatch.setattr(sp, "_is_font_installed", lambda name: name == "Libertinus Serif")
  rc_lib = sp.get_academic_plot_rc_params()
  assert rc_lib["mathtext.rm"] == "Libertinus Serif"
  assert rc_lib["mathtext.it"] == "Libertinus Serif:italic"
  assert rc_lib["mathtext.bf"] == "Libertinus Serif:bold"
  monkeypatch.undo()

  # 2. Verify legend bounding box sits strictly below suptitle bounding box
  captured_gaps: list[float] = []
  orig_savefig = Figure.savefig

  def _inspect_savefig(self: Figure, *args: Any, **kwargs: Any) -> Any:
    renderer = self.canvas.get_renderer()  # type: ignore[attr-defined]
    inv = self.transFigure.inverted()
    leg = self.axes[0].get_legend()
    sup = self._suptitle  # type: ignore[attr-defined]
    assert leg is not None and sup is not None
    leg_bbox = leg.get_window_extent(renderer).transformed(inv)
    sup_bbox = sup.get_window_extent(renderer).transformed(inv)
    captured_gaps.append(float(sup_bbox.y0 - leg_bbox.y1))
    return orig_savefig(self, *args, **kwargs)

  monkeypatch.setattr(Figure, "savefig", _inspect_savefig)

  all_results, configs = _build_synthetic_results(include_large_cpu=True)
  generate_scaling_plots(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tmp_path,
    title="Escalonamento do Tempo de Execução e da Qualidade vs. Dimensão do Problema",
    band_label="IC 95% (t-Student, repetições)",
    legend_labels={
      "CPU": "CPU",
      "HybridNaive": "Híbrida Ingênua",
      "HybridOptimized": "Híbrida Otimizada",
      "FullGPUModular": "Totalmente em GPU",
    },
    formats=("pdf",),
  )
  assert captured_gaps and captured_gaps[0] > 0.015


def test_summarize_metric_dispersion_modes_and_invalid_options(tmp_path: Path) -> None:
  """WHAT: Validate IQR, SD, and CI dispersion calculations (raw & summary-only) plus invalid option guards."""
  import numpy as np

  skewed = {"raw_times": [1.0, 2.0, 3.0, 4.0, 100.0]}
  med, q25, q75 = _summarize_metric(skewed, "raw_times", "mean_time", "std_time", dispersion_metric="iqr") or (
    0.0,
    0.0,
    0.0,
  )
  assert med == pytest.approx(3.0)
  assert q25 == pytest.approx(2.0)
  assert q75 == pytest.approx(4.0)

  mean_sd, lo_sd, hi_sd = _summarize_metric(
    {"raw_times": [10.0, 12.0, 14.0]}, "raw_times", "mean_time", "std_time", dispersion_metric="std"
  ) or (0.0, 0.0, 0.0)
  assert mean_sd == pytest.approx(12.0)
  assert hi_sd - mean_sd == pytest.approx(2.0)
  assert mean_sd - lo_sd == pytest.approx(2.0)

  summary_only = {"mean_time": 10.0, "std_time": 2.0, "repetitions": 10}
  c_iqr, lo_iqr, hi_iqr = _summarize_metric(
    summary_only, "raw_times", "mean_time", "std_time", dispersion_metric="iqr"
  ) or (0.0, 0.0, 0.0)
  assert c_iqr == pytest.approx(10.0)
  assert hi_iqr - c_iqr == pytest.approx(0.67448975 * 2.0, rel=1e-4)
  assert c_iqr - lo_iqr == pytest.approx(0.67448975 * 2.0, rel=1e-4)

  with pytest.raises(ValueError, match="Unsupported dispersion_metric"):
    _summarize_metric(skewed, "raw_times", "mean_time", "std_time", dispersion_metric="invalid")  # type: ignore[arg-type]

  all_results, configs = _build_synthetic_results(include_large_cpu=True)
  with pytest.raises(ValueError, match="Unsupported dispersion_metric"):
    generate_scaling_plots(all_results, configs, tmp_path, dispersion_metric="bad")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="Unsupported x_axis_mode"):
    generate_scaling_plots(all_results, configs, tmp_path, x_axis_mode="log")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="Unsupported x_tick_mode"):
    generate_scaling_plots(all_results, configs, tmp_path, x_tick_mode="none")  # type: ignore[arg-type]

  # Single-instance cohort should render cleanly without division-by-zero in continuous tick placement
  single_results = {"att48": all_results["att48"]}
  single_configs = [np.asarray(configs)[0]]
  (single_pdf,) = generate_scaling_plots(
    single_results, list(single_configs), tmp_path, filename_stem="single", formats=("pdf",)
  )
  assert single_pdf.is_file() and single_pdf.read_bytes()[:4] == b"%PDF"


def _oriented_bbox(renderer: Any, text_obj: Any, angle_deg: float) -> list[tuple[float, float]]:
  """Compute 4 display-coordinate corners of a rotated ``rotation_mode='anchor'`` annotation."""
  import math

  w, h, _ = renderer.get_text_width_height_descent(text_obj.get_text(), text_obj.get_fontproperties(), ismath=False)
  pos_disp = text_obj.get_transform().transform(text_obj.get_position())
  rad = math.radians(angle_deg)
  ca, sa = math.cos(rad), math.sin(rad)
  corners: list[tuple[float, float]] = []
  # ha="right", va="top" anchor puts unrotated box at x in [-w, 0], y in [-h, 0]
  for lx, ly in ((-w, -h), (0.0, -h), (0.0, 0.0), (-w, 0.0)):
    corners.append((pos_disp[0] + lx * ca - ly * sa, pos_disp[1] + lx * sa + ly * ca))
  return corners


def _polygons_overlap(poly1: list[tuple[float, float]], poly2: list[tuple[float, float]]) -> bool:
  """Separating Axis Theorem check for convex 2D polygons."""
  for poly in (poly1, poly2):
    for idx in range(len(poly)):
      x1, y1 = poly[idx]
      x2, y2 = poly[(idx + 1) % len(poly)]
      nx, ny = -(y2 - y1), x2 - x1
      proj1 = [px * nx + py * ny for px, py in poly1]
      proj2 = [px * nx + py * ny for px, py in poly2]
      if max(proj1) <= min(proj2) or max(proj2) <= min(proj1):
        return False
  return True


def test_continuous_instance_ticks_no_overlap_across_15_cohort_and_ordinal_mode(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: Continuous numeric x-axis places all 15 exact instance n values without rotated label overlap.

  WHY: Task 14.6 (DP14-5) — Adjacent clusters (48/51/52, 96/100, 532/535) must remain readable on a
  continuous proportional n axis without colliding labels or colliding with xlabel.
  """
  from matplotlib.figure import Figure

  cohort = [
    ("att48", 48),
    ("eil51", 51),
    ("berlin52", 52),
    ("gr96", 96),
    ("kroA100", 100),
    ("ch150", 150),
    ("d198", 198),
    ("pr264", 264),
    ("lin318", 318),
    ("pcb442", 442),
    ("att532", 532),
    ("si535", 535),
    ("gr666", 666),
    ("rat783", 783),
    ("pr1002", 1002),
  ]
  configs = [ProblemConfig(name=name, optimal=1000.0, size=size) for name, size in cohort]
  all_results: dict[str, dict[str, dict[str, Any]]] = {}
  for name, size in cohort:
    scale = (size / 50.0) ** 2
    all_results[name] = {
      "CPU": {"problem_size": size, "raw_times": [1.8 * scale, 2.0 * scale, 2.3 * scale], "raw_gaps": [0.5, 0.7, 0.9]},
      "FullGPUModular": {
        "problem_size": size,
        "raw_times": [0.05 * scale, 0.06 * scale, 0.08 * scale],
        "raw_gaps": [0.4, 0.6, 0.8],
      },
    }

  captured_xticks: list[list[float]] = []
  captured_labels: list[str] = []
  overlap_pairs: list[tuple[str, str]] = []
  xlabel_clearance: list[float] = []
  orig_savefig = Figure.savefig

  def _inspect_savefig(self: Figure, *args: Any, **kwargs: Any) -> Any:
    ax_gap = self.axes[1]
    renderer = self.canvas.get_renderer()  # type: ignore[attr-defined]
    captured_xticks.append([float(x) for x in ax_gap.get_xticks()])
    tick_anns = [t for t in ax_gap.texts if t.get_text().isdigit()]
    captured_labels.extend(t.get_text() for t in tick_anns)
    obbs = [_oriented_bbox(renderer, t, 45.0) for t in tick_anns]
    for i in range(len(obbs)):
      for j in range(i + 1, len(obbs)):
        if _polygons_overlap(obbs[i], obbs[j]):
          overlap_pairs.append((tick_anns[i].get_text(), tick_anns[j].get_text()))
    min_tick_y = min(t.get_window_extent(renderer).y0 for t in tick_anns)
    xlabel_y1 = ax_gap.xaxis.label.get_window_extent(renderer).y1
    xlabel_clearance.append(float(min_tick_y - xlabel_y1))
    return orig_savefig(self, *args, **kwargs)

  monkeypatch.setattr(Figure, "savefig", _inspect_savefig)

  (pdf_path,) = generate_scaling_plots(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tmp_path,
    filename_stem="cohort15_continuous",
    dispersion_metric="iqr",
    x_axis_mode="continuous",
    x_tick_mode="instances",
    formats=("pdf",),
  )
  monkeypatch.undo()

  expected_ns = [size for _, size in cohort]
  assert pdf_path.is_file()
  assert captured_xticks[0] == pytest.approx([float(n) for n in expected_ns])
  assert [int(lbl) for lbl in captured_labels] == expected_ns
  assert overlap_pairs == []
  assert xlabel_clearance and xlabel_clearance[0] > 0.0

  # Also verify ordinal mode and auto tick mode render cleanly
  (ordinal_pdf,) = generate_scaling_plots(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tmp_path,
    filename_stem="cohort15_ordinal",
    dispersion_metric="std",
    x_axis_mode="ordinal",
    formats=("pdf",),
  )
  (auto_pdf,) = generate_scaling_plots(
    all_results=all_results,
    problem_configs=configs,
    output_dir=tmp_path,
    filename_stem="cohort15_auto",
    dispersion_metric="ci",
    x_axis_mode="continuous",
    x_tick_mode="auto",
    formats=("pdf",),
  )
  assert ordinal_pdf.is_file() and ordinal_pdf.read_bytes()[:4] == b"%PDF"
  assert auto_pdf.is_file() and auto_pdf.read_bytes()[:4] == b"%PDF"


def test_continuous_ticks_narrow_right_cluster_5digit_fallback_and_ordinal_ties(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """WHAT: Continuous ticks avoid collisions on 5-digit DejaVu fallback and right clusters; ordinal separates tied n.

  WHY: Fixed point constants collide on 5-digit DejaVu labels and single-sided left PAVA clamping overflows
  the right spine on narrow figures; ordinal mode keyed by size collapsed same-dimension instances onto slot 0.
  """
  import numpy as np
  from matplotlib.figure import Figure

  import benchmarking.analysis.scaling_plots as sp

  orig_savefig = Figure.savefig

  def _run_and_inspect(
    sizes: list[int],
    width: float,
    stem: str,
  ) -> tuple[list[tuple[str, str]], list[tuple[str, float]]]:
    configs = [ProblemConfig(name=f"p{s}", optimal=100.0, size=s) for s in sizes]
    all_res = {
      f"p{s}": {"A": {"problem_size": s, "raw_times": [1.0, 1.2], "raw_gaps": [0.1, 0.2]}} for s in sizes
    }
    overlaps: list[tuple[str, str]] = []
    right_overshoot: list[tuple[str, float]] = []

    def _spy(self: Figure, *args: Any, **kwargs: Any) -> Any:
      ax_gap = self.axes[1]
      renderer = self.canvas.get_renderer()  # type: ignore[attr-defined]
      anns = [t for t in ax_gap.texts if t.get_text().isdigit()]
      obbs = [_oriented_bbox(renderer, t, 45.0) for t in anns]
      for i in range(len(obbs)):
        for j in range(i + 1, len(obbs)):
          if _polygons_overlap(obbs[i], obbs[j]):
            overlaps.append((anns[i].get_text(), anns[j].get_text()))
      ax_bb = ax_gap.get_window_extent(renderer)
      for t, obb in zip(anns, obbs, strict=True):
        xs = [pt[0] for pt in obb]
        if max(xs) > ax_bb.x1 + 5.0:
          right_overshoot.append((t.get_text(), float(max(xs) - ax_bb.x1)))
      return orig_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy)
    generate_scaling_plots(all_res, configs, tmp_path, filename_stem=stem, figure_width=width, formats=("pdf",))
    monkeypatch.undo()
    return overlaps, right_overshoot

  # 1. Right-edge cluster on narrow 3.15 in canvas must stay inside right spine without overlap
  ov_right, overshoot_right = _run_and_inspect([48, 995, 998, 999, 1000, 1001, 1002], 3.15, "right_narrow")
  assert ov_right == []
  assert overshoot_right == []

  # 2. 5-digit cluster under DejaVu Serif fallback must measure real font glyphs and avoid overlap
  base_rc = sp.get_academic_plot_rc_params()
  monkeypatch.setattr(sp, "get_academic_plot_rc_params", lambda: {**base_rc, "font.serif": ["DejaVu Serif"]})
  ov_5d, overshoot_5d = _run_and_inspect([10000, 10001, 10002, 10003, 10050, 20000], 6.3, "dejavu_5d")
  monkeypatch.undo()
  assert ov_5d == []
  assert overshoot_5d == []

  # 3. Ordinal mode must map distinct instances sharing the same n (e.g. kroA100 & kroB100) to consecutive slots
  dup_configs = [
    ProblemConfig(name="kroA100", optimal=21282.0, size=100),
    ProblemConfig(name="kroB100", optimal=22141.0, size=100),
    ProblemConfig(name="ch150", optimal=6528.0, size=150),
  ]
  dup_results: dict[str, dict[str, dict[str, Any]]] = {
    "kroA100": {"CPU": {"raw_times": [4.0, 5.0], "raw_gaps": [0.4, 0.6]}},
    "kroB100": {"CPU": {"raw_times": [6.0, 7.0], "raw_gaps": [0.8, 1.0]}},
    "ch150": {"CPU": {"raw_times": [9.0, 10.0], "raw_gaps": [0.7, 0.9]}},
  }
  captured_xdata: list[list[float]] = []
  captured_tick_labels: list[list[str]] = []

  def _spy_ord(self: Figure, *args: Any, **kwargs: Any) -> Any:
    ax_gap = self.axes[1]
    captured_xdata.append(np.asarray(ax_gap.lines[0].get_xdata(), dtype=float).tolist())
    captured_tick_labels.append([t.get_text() for t in ax_gap.get_xticklabels()])
    return orig_savefig(self, *args, **kwargs)

  monkeypatch.setattr(Figure, "savefig", _spy_ord)
  generate_scaling_plots(
    dup_results,
    dup_configs,
    tmp_path,
    filename_stem="ordinal_dup_n",
    x_axis_mode="ordinal",
    formats=("pdf",),
  )
  monkeypatch.undo()
  assert captured_xdata[0] == pytest.approx([0.0, 1.0, 2.0])
  assert captured_tick_labels[0] == ["kroA100 (100)", "kroB100 (100)", "ch150 (150)"]
