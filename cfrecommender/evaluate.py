"""Evaluation metrics for a trained recommender's rating predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .data import Rating
from .model import MatrixFactorizationModel


@dataclass(frozen=True)
class EvaluationResult:
    rmse: float
    mae: float
    n_evaluated: int
    n_skipped_cold_start: int


def evaluate(
    model: MatrixFactorizationModel,
    test_ratings: Sequence[Rating],
    skip_cold_start: bool = False,
) -> EvaluationResult:
    """Compute RMSE and MAE of `model`'s predictions against `test_ratings`.

    If `skip_cold_start` is True, ratings whose user or item was never seen
    during training are excluded from the metric entirely (and counted in
    `n_skipped_cold_start`) instead of being scored against the model's
    global-mean fallback prediction.
    """
    known_users = set(model.known_users())
    known_items = set(model.known_items())

    sq_err_sum = 0.0
    abs_err_sum = 0.0
    n_evaluated = 0
    n_skipped = 0

    for r in test_ratings:
        is_cold = r.user not in known_users or r.item not in known_items
        if is_cold and skip_cold_start:
            n_skipped += 1
            continue
        pred = model.predict(r.user, r.item)
        err = r.rating - pred
        sq_err_sum += err * err
        abs_err_sum += abs(err)
        n_evaluated += 1

    if n_evaluated == 0:
        raise ValueError("No ratings were evaluated (test set empty, or all ratings were cold-start and skipped)")

    rmse = (sq_err_sum / n_evaluated) ** 0.5
    mae = abs_err_sum / n_evaluated
    return EvaluationResult(rmse=rmse, mae=mae, n_evaluated=n_evaluated, n_skipped_cold_start=n_skipped)
