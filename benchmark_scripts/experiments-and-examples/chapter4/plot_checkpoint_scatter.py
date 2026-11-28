"""Scatter diagnostics for one benchmark checkpoint: time vs generations vs stop reason.

Usage:
  path/to/python plot_checkpoint_scatter.py [checkpoint.json]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main(ckpt_path: Path) -> Path:
  payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
  times = np.asarray(payload["raw_times"], dtype=float)
  gens = np.asarray(payload["raw_generations"], dtype=float)
  reasons = np.asarray(payload["raw_stop_reasons"], dtype=object)
  seeds = np.asarray(payload.get("raw_seeds", np.arange(times.size)))
  if not (times.size == gens.size == reasons.size == seeds.size):
    raise ValueError(f"misaligned parallel arrays in {ckpt_path.name}")

  pearson = stats.pearsonr(gens, times)
  spearman = stats.spearmanr(gens, times)
  slope, intercept = np.polyfit(gens, times, 1)
  order = [reason for reason, _ in Counter(reasons).most_common()]
  cmap = plt.get_cmap("tab10")
  palette = {reason: cmap(idx / 9.0) for idx, reason in enumerate(order)}

  fig, (ax_s, ax_b) = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.4),
    gridspec_kw={"width_ratios": [2.2, 1.0]},
  )

  for reason in order:
    mask = reasons == reason
    ax_s.scatter(
      gens[mask],
      times[mask],
      s=58,
      alpha=0.85,
      color=palette[reason],
      edgecolor="black",
      linewidth=0.5,
      label=f"{reason} (n={int(mask.sum())})",
    )
    if mask.sum() <= 3:
      for g, t, seed in zip(gens[mask], times[mask], seeds[mask], strict=True):
        ax_s.annotate(
          f"seed {seed}",
          (g, t),
          textcoords="offset points",
          xytext=(6, 6),
          fontsize=8,
        )

  span = np.linspace(gens.min() * 0.97, gens.max() * 1.03, 100)
  ax_s.plot(
    span,
    slope * span + intercept,
    "k--",
    linewidth=1.2,
    label=f"OLS t = {intercept:.0f} + {slope:.2f}*g  (R2={pearson.statistic**2:.3f})",
  )
  ax_s.set_xlabel("generations  (raw_generations)")
  ax_s.set_ylabel("wall-clock time [s]  (raw_times)")
  ax_s.set_title(
    f"{ckpt_path.stem}: r={pearson.statistic:.3f} (p={pearson.pvalue:.1e}), "
    f"rho={spearman.statistic:.3f} (p={spearman.pvalue:.1e})",
    fontsize=10,
  )
  ax_s.legend(frameon=False, fontsize=8, loc="upper left")
  ax_s.grid(alpha=0.3)

  groups = [times[reasons == reason] for reason in order]
  ax_b.boxplot(
    groups,
    tick_labels=[f"{r}\n(n={len(d)})" for r, d in zip(order, groups, strict=True)],
    widths=0.45,
  )
  rng = np.random.default_rng(0)
  for idx, (reason, group) in enumerate(zip(order, groups, strict=True), start=1):
    ax_b.scatter(
      idx + rng.uniform(-0.08, 0.08, group.size),
      group,
      s=34,
      color=palette[reason],
      edgecolor="black",
      linewidth=0.4,
      zorder=3,
    )
  ax_b.set_ylabel("wall-clock time [s]")
  ax_b.set_title("time by stop reason", fontsize=10)
  ax_b.grid(alpha=0.3, axis="y")

  fig.tight_layout()
  out = ckpt_path.with_name(f"{ckpt_path.stem}_time_vs_generations.png")
  fig.savefig(out, dpi=200)

  per_gen = times / gens
  print(
    f"n={times.size}  r={pearson.statistic:.4f} (p={pearson.pvalue:.2e})  "
    f"rho={spearman.statistic:.4f} (p={spearman.pvalue:.2e})"
  )
  print(f"OLS: time = {intercept:.1f} + {slope:.4f} * generations  R2={pearson.statistic**2:.4f}")
  print(
    f"seconds/generation: min={per_gen.min():.3f} mean={per_gen.mean():.3f} "
    f"max={per_gen.max():.3f}; corr(generations, s/gen)="
    f"{stats.pearsonr(gens, per_gen).statistic:.3f}"
  )
  print(f"stop reasons: {dict(Counter(reasons))}")
  print(f"wrote {out}")
  return out


if __name__ == "__main__":
  default = Path(__file__).resolve().parent / "results" / "checkpoints" / "pr1002_FullGPUModular.json"
  main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
