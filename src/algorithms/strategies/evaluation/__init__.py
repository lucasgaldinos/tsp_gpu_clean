"""Evaluation strategies for routing algorithms."""

from .cost_evaluation import CPUCostEvaluation, GPUCostEvaluation

__all__: list[str] = ["CPUCostEvaluation", "GPUCostEvaluation"]
