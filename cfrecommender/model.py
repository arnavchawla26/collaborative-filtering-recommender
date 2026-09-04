"""A dependency-free SGD-trained matrix-factorization recommender.

Implements a Funk-SVD-style model: predicted rating = global mean + per-user
bias + per-item bias + dot(user latent factors, item latent factors), trained
by stochastic gradient descent with L2 regularization. Built entirely from
plain Python floats, lists, and dicts -- no numpy, no pandas.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Rating


class MatrixFactorizationModel:
    """SGD matrix-factorization recommender with bias terms."""

    def __init__(
        self,
        n_factors: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.02,
        n_epochs: int = 20,
        init_std: float = 0.1,
        seed: int = 42,
    ) -> None:
        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        if n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.init_std = init_std
        self.seed = seed

        self.global_mean: float = 0.0
        self.user_bias: Dict[str, float] = {}
        self.item_bias: Dict[str, float] = {}
        self.user_factors: Dict[str, List[float]] = {}
        self.item_factors: Dict[str, List[float]] = {}
        self.train_rmse_history: List[float] = []
        self._fitted = False

    def fit(self, train_ratings: Sequence[Rating], verbose: bool = False) -> "MatrixFactorizationModel":
        """Train the model on `train_ratings` via SGD for `n_epochs` epochs,
        re-shuffling the training examples each epoch. Returns self.
        """
        if not train_ratings:
            raise ValueError("train_ratings must not be empty")

        rng = random.Random(self.seed)
        users = sorted({r.user for r in train_ratings})
        items = sorted({r.item for r in train_ratings})

        self.global_mean = sum(r.rating for r in train_ratings) / len(train_ratings)
        self.user_bias = {u: 0.0 for u in users}
        self.item_bias = {i: 0.0 for i in items}
        self.user_factors = {u: [rng.gauss(0, self.init_std) for _ in range(self.n_factors)] for u in users}
        self.item_factors = {i: [rng.gauss(0, self.init_std) for _ in range(self.n_factors)] for i in items}

        examples = list(train_ratings)
        self.train_rmse_history = []
        lr = self.learning_rate
        reg = self.regularization
        k_range = range(self.n_factors)

        for epoch in range(self.n_epochs):
            rng.shuffle(examples)
            sq_err_sum = 0.0
            for r in examples:
                pu = self.user_factors[r.user]
                qi = self.item_factors[r.item]
                bu = self.user_bias[r.user]
                bi = self.item_bias[r.item]

                dot = sum(pu[k] * qi[k] for k in k_range)
                pred = self.global_mean + bu + bi + dot
                err = r.rating - pred
                sq_err_sum += err * err

                self.user_bias[r.user] = bu + lr * (err - reg * bu)
                self.item_bias[r.item] = bi + lr * (err - reg * bi)

                # Compute both updates from the *pre-update* factor values so
                # the simultaneous-gradient-step semantics of SGD are preserved.
                new_pu = [pu[k] + lr * (err * qi[k] - reg * pu[k]) for k in k_range]
                new_qi = [qi[k] + lr * (err * pu[k] - reg * qi[k]) for k in k_range]
                self.user_factors[r.user] = new_pu
                self.item_factors[r.item] = new_qi

            epoch_rmse = (sq_err_sum / len(examples)) ** 0.5
            self.train_rmse_history.append(epoch_rmse)
            if verbose:
                print(f"epoch {epoch + 1}/{self.n_epochs}  train_rmse={epoch_rmse:.4f}")

        self._fitted = True
        return self

    def predict(self, user: str, item: str) -> float:
        """Predict a rating for (user, item). Falls back gracefully for a
        user or item unseen during training by using the global mean plus
        whichever bias term(s) are known (no crash on cold-start lookups).
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fit yet -- call fit() first")

        bu = self.user_bias.get(user, 0.0)
        bi = self.item_bias.get(item, 0.0)
        pu = self.user_factors.get(user)
        qi = self.item_factors.get(item)
        dot = sum(pu[k] * qi[k] for k in range(self.n_factors)) if (pu is not None and qi is not None) else 0.0
        return self.global_mean + bu + bi + dot

    def recommend(
        self,
        user: str,
        candidate_items: Sequence[str],
        exclude_items: Optional[Sequence[str]] = None,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return the top-N (item, predicted_rating) pairs for `user` among
        `candidate_items`, excluding any item in `exclude_items` (typically
        items the user has already rated). Ties broken by item id for
        deterministic output.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fit yet -- call fit() first")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        exclude = set(exclude_items or ())
        scored = [(item, self.predict(user, item)) for item in candidate_items if item not in exclude]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_n]

    def known_users(self) -> List[str]:
        return sorted(self.user_factors)

    def known_items(self) -> List[str]:
        return sorted(self.item_factors)
