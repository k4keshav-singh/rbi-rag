"""Evaluation: metrics, the faithfulness judge, calibration and the ablation."""

from rbi_rag.eval.ablation import ArmSpec, evaluate_generation, evaluate_retrieval
from rbi_rag.eval.calibration import calibrate
from rbi_rag.eval.dataset import load_eval_set, verified_only, write_eval_set
from rbi_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "ArmSpec",
    "calibrate",
    "evaluate_generation",
    "evaluate_retrieval",
    "load_eval_set",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "verified_only",
    "write_eval_set",
]
