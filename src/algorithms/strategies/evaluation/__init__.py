"""Evaluation strategies for routing algorithms."""

from .cost_evaluation import EvaluationCPUCost, EvaluationGPUCost

__all__: list[str] = ["EvaluationCPUCost", "EvaluationGPUCost"]
