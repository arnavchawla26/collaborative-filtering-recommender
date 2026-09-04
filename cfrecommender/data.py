"""Data structures, synthetic dataset generation, CSV I/O, and train/test
splitting for the collaborative-filtering recommender. No third-party
dependencies -- only the standard library.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Rating:
    """A single observed (user, item, rating) triple."""

    user: str
    item: str
    rating: float


def load_ratings_csv(path: str) -> List[Rating]:
    """Load ratings from a CSV file with a header row containing at least the
    columns 'user', 'item', 'rating' (any column order, extra columns ignored).
    """
    ratings: List[Rating] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"user", "item", "rating"}
        fieldnames = set(reader.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(f"CSV must have header columns {sorted(required)}, got {sorted(fieldnames)}")
        for row_num, row in enumerate(reader, start=2):
            user = (row["user"] or "").strip()
            item = (row["item"] or "").strip()
            rating_raw = (row["rating"] or "").strip()
            if not user or not item:
                raise ValueError(f"Row {row_num}: 'user' and 'item' must be non-empty")
            try:
                rating = float(rating_raw)
            except ValueError as exc:
                raise ValueError(f"Row {row_num}: rating {rating_raw!r} is not a number") from exc
            ratings.append(Rating(user=user, item=item, rating=rating))
    if not ratings:
        raise ValueError(f"No ratings found in {path!r}")
    return ratings


def save_ratings_csv(ratings: Sequence[Rating], path: str) -> None:
    """Write ratings to a CSV file with header 'user,item,rating'."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "item", "rating"])
        for r in ratings:
            writer.writerow([r.user, r.item, r.rating])


def generate_synthetic_ratings(
    n_users: int = 40,
    n_items: int = 25,
    n_latent: int = 4,
    density: float = 0.35,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    noise_std: float = 0.4,
    seed: int = 42,
) -> List[Rating]:
    """Generate a deterministic synthetic ratings dataset with a genuine
    low-rank structure: every user and item gets a hidden n_latent-dimensional
    preference vector (drawn from a seeded RNG), and the "true" rating is the
    dot product of those vectors plus per-user/per-item bias terms, clipped to
    [rating_min, rating_max] and perturbed by Gaussian noise. This gives a
    matrix-factorization model genuine correlation structure to learn, rather
    than pure noise.

    Every user and every item is guaranteed to appear in at least one rating.
    """
    if n_users < 1 or n_items < 1:
        raise ValueError("n_users and n_items must be >= 1")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if rating_min >= rating_max:
        raise ValueError("rating_min must be < rating_max")

    rng = random.Random(seed)
    user_ids = [f"u{idx}" for idx in range(n_users)]
    item_ids = [f"i{idx}" for idx in range(n_items)]

    user_vecs = {u: [rng.gauss(0, 1) for _ in range(n_latent)] for u in user_ids}
    item_vecs = {i: [rng.gauss(0, 1) for _ in range(n_latent)] for i in item_ids}
    user_bias = {u: rng.gauss(0, 0.5) for u in user_ids}
    item_bias = {i: rng.gauss(0, 0.5) for i in item_ids}
    global_mean = (rating_min + rating_max) / 2.0

    def _true_rating(u: str, i: str) -> float:
        dot = sum(a * b for a, b in zip(user_vecs[u], item_vecs[i]))
        raw = global_mean + user_bias[u] + item_bias[i] + dot
        return min(rating_max, max(rating_min, raw))

    ratings: List[Rating] = []
    for u in user_ids:
        for i in item_ids:
            if rng.random() > density:
                continue
            noisy = min(rating_max, max(rating_min, _true_rating(u, i) + rng.gauss(0, noise_std)))
            ratings.append(Rating(user=u, item=i, rating=round(noisy, 2)))

    # Backfill any user/item that got unlucky and drew zero ratings above, so
    # every declared user and item is guaranteed to appear at least once.
    seen_users = {r.user for r in ratings}
    seen_items = {r.item for r in ratings}
    for u in user_ids:
        if u not in seen_users:
            i = item_ids[rng.randrange(n_items)]
            ratings.append(Rating(user=u, item=i, rating=round(_true_rating(u, i), 2)))
    for i in item_ids:
        if i not in seen_items:
            u = user_ids[rng.randrange(n_users)]
            ratings.append(Rating(user=u, item=i, rating=round(_true_rating(u, i), 2)))

    return ratings


def train_test_split(
    ratings: Sequence[Rating],
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[Rating], List[Rating]]:
    """Split ratings into (train, test) sets.

    Guarantees that every user and every item appearing anywhere in `ratings`
    keeps at least one rating in the TRAIN set: a rating is only allowed to
    land in `test` if, after the random split, its user and item both still
    have train-set coverage from some other rating. This keeps evaluation
    honest (no accidental cold-start users/items in the held-out set) without
    needing a separate stratified-sampling pass.
    """
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")

    rng = random.Random(seed)
    shuffled = list(ratings)
    rng.shuffle(shuffled)

    train: List[Rating] = []
    test: List[Rating] = []
    for r in shuffled:
        if rng.random() < test_fraction:
            test.append(r)
        else:
            train.append(r)

    train_users = {r.user for r in train}
    train_items = {r.item for r in train}

    kept_test: List[Rating] = []
    for r in test:
        if r.user not in train_users or r.item not in train_items:
            train.append(r)
            train_users.add(r.user)
            train_items.add(r.item)
        else:
            kept_test.append(r)

    return train, kept_test
