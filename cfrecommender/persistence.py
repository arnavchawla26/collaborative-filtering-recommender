"""Save/load a trained MatrixFactorizationModel to/from a JSON file."""

from __future__ import annotations

import json
from typing import Any, Dict

from .model import MatrixFactorizationModel

_FORMAT_VERSION = 1


def model_to_dict(model: MatrixFactorizationModel) -> Dict[str, Any]:
    if not model._fitted:
        raise RuntimeError("Model has not been fit yet -- call fit() before saving")
    return {
        "format_version": _FORMAT_VERSION,
        "hyperparameters": {
            "n_factors": model.n_factors,
            "learning_rate": model.learning_rate,
            "regularization": model.regularization,
            "n_epochs": model.n_epochs,
            "init_std": model.init_std,
            "seed": model.seed,
        },
        "global_mean": model.global_mean,
        "user_bias": model.user_bias,
        "item_bias": model.item_bias,
        "user_factors": model.user_factors,
        "item_factors": model.item_factors,
        "train_rmse_history": model.train_rmse_history,
    }


def save_model(model: MatrixFactorizationModel, path: str) -> None:
    """Serialize a fitted model to a JSON file at `path`."""
    data = model_to_dict(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_model(path: str) -> MatrixFactorizationModel:
    """Load a model previously saved with save_model(). The returned model
    behaves identically to the original for predict()/recommend() calls.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"Unsupported model file format_version: {data.get('format_version')!r}")

    hp = data["hyperparameters"]
    model = MatrixFactorizationModel(
        n_factors=hp["n_factors"],
        learning_rate=hp["learning_rate"],
        regularization=hp["regularization"],
        n_epochs=hp["n_epochs"],
        init_std=hp["init_std"],
        seed=hp["seed"],
    )
    model.global_mean = data["global_mean"]
    model.user_bias = dict(data["user_bias"])
    model.item_bias = dict(data["item_bias"])
    model.user_factors = {k: list(v) for k, v in data["user_factors"].items()}
    model.item_factors = {k: list(v) for k, v in data["item_factors"].items()}
    model.train_rmse_history = list(data.get("train_rmse_history", []))
    model._fitted = True
    return model
