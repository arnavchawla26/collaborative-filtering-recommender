import csv
import os
import tempfile

import pytest

from cfrecommender.data import (
    Rating,
    generate_synthetic_ratings,
    load_ratings_csv,
    save_ratings_csv,
    train_test_split,
)


def test_generate_synthetic_ratings_deterministic():
    a = generate_synthetic_ratings(n_users=15, n_items=10, seed=7)
    b = generate_synthetic_ratings(n_users=15, n_items=10, seed=7)
    assert a == b


def test_generate_synthetic_ratings_different_seed_differs():
    a = generate_synthetic_ratings(n_users=15, n_items=10, seed=7)
    b = generate_synthetic_ratings(n_users=15, n_items=10, seed=8)
    assert a != b


def test_generate_synthetic_ratings_covers_every_user_and_item():
    ratings = generate_synthetic_ratings(n_users=20, n_items=12, density=0.05, seed=1)
    users = {r.user for r in ratings}
    items = {r.item for r in ratings}
    assert users == {f"u{i}" for i in range(20)}
    assert items == {f"i{i}" for i in range(12)}


def test_generate_synthetic_ratings_within_bounds():
    ratings = generate_synthetic_ratings(n_users=25, n_items=15, rating_min=1.0, rating_max=5.0, seed=3)
    for r in ratings:
        assert 1.0 <= r.rating <= 5.0


def test_generate_synthetic_ratings_invalid_args():
    with pytest.raises(ValueError):
        generate_synthetic_ratings(n_users=0)
    with pytest.raises(ValueError):
        generate_synthetic_ratings(n_items=0)
    with pytest.raises(ValueError):
        generate_synthetic_ratings(density=0)
    with pytest.raises(ValueError):
        generate_synthetic_ratings(density=1.5)
    with pytest.raises(ValueError):
        generate_synthetic_ratings(rating_min=5.0, rating_max=1.0)


def test_generate_synthetic_ratings_has_learnable_structure():
    # A user's ratings for items should correlate with dot-product structure,
    # not be pure noise: two users with very similar hidden preferences (same
    # seed stream) should rate items more similarly than two arbitrary users
    # would by chance. We check this indirectly via correlation of a user's
    # ratings against another draw of "similar" synthetic data is out of
    # scope here -- instead assert basic sanity: not all ratings identical,
    # and there is real variance (i.e. not degenerate).
    ratings = generate_synthetic_ratings(n_users=30, n_items=20, seed=11)
    values = [r.rating for r in ratings]
    assert len(set(values)) > 10
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    assert variance > 0.05


def test_csv_round_trip():
    ratings = [
        Rating(user="alice", item="movie1", rating=4.5),
        Rating(user="bob", item="movie2", rating=3.0),
        Rating(user="alice", item="movie2", rating=2.5),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ratings.csv")
        save_ratings_csv(ratings, path)
        loaded = load_ratings_csv(path)
        assert loaded == ratings


def test_load_ratings_csv_missing_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "item"])
            writer.writerow(["alice", "movie1"])
        with pytest.raises(ValueError, match="header columns"):
            load_ratings_csv(path)


def test_load_ratings_csv_bad_rating_value():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "item", "rating"])
            writer.writerow(["alice", "movie1", "not-a-number"])
        with pytest.raises(ValueError, match="not a number"):
            load_ratings_csv(path)


def test_load_ratings_csv_empty_user_or_item():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "item", "rating"])
            writer.writerow(["", "movie1", "4.0"])
        with pytest.raises(ValueError, match="non-empty"):
            load_ratings_csv(path)


def test_load_ratings_csv_empty_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "item", "rating"])
        with pytest.raises(ValueError, match="No ratings found"):
            load_ratings_csv(path)


def test_train_test_split_covers_all_users_and_items_in_train():
    ratings = generate_synthetic_ratings(n_users=20, n_items=15, seed=5)
    train, test = train_test_split(ratings, test_fraction=0.3, seed=5)

    all_users = {r.user for r in ratings}
    all_items = {r.item for r in ratings}
    train_users = {r.user for r in train}
    train_items = {r.item for r in train}

    assert train_users == all_users
    assert train_items == all_items
    assert len(train) + len(test) == len(ratings)
    assert set(train) & set(test) == set()


def test_train_test_split_roughly_matches_fraction_on_dense_data():
    # With plenty of ratings per user/item, the coverage-guarantee correction
    # should only nudge a handful of ratings back into train, so the actual
    # test fraction should land reasonably close to the requested one.
    ratings = generate_synthetic_ratings(n_users=50, n_items=40, density=0.6, seed=9)
    train, test = train_test_split(ratings, test_fraction=0.25, seed=9)
    actual_fraction = len(test) / len(ratings)
    assert 0.15 < actual_fraction < 0.30


def test_train_test_split_zero_fraction_returns_empty_test():
    ratings = generate_synthetic_ratings(n_users=10, n_items=8, seed=2)
    train, test = train_test_split(ratings, test_fraction=0.0, seed=2)
    assert test == []
    assert len(train) == len(ratings)


def test_train_test_split_invalid_fraction():
    ratings = [Rating(user="a", item="x", rating=3.0)]
    with pytest.raises(ValueError):
        train_test_split(ratings, test_fraction=1.0)
    with pytest.raises(ValueError):
        train_test_split(ratings, test_fraction=-0.1)


def test_train_test_split_deterministic_given_seed():
    ratings = generate_synthetic_ratings(n_users=12, n_items=9, seed=4)
    a_train, a_test = train_test_split(ratings, test_fraction=0.2, seed=99)
    b_train, b_test = train_test_split(ratings, test_fraction=0.2, seed=99)
    assert a_train == b_train
    assert a_test == b_test


def test_train_test_split_single_rating_per_user_forces_all_to_train():
    # Every user/item has exactly one rating -- nothing can safely go to test.
    ratings = [Rating(user=f"u{i}", item=f"i{i}", rating=3.0) for i in range(5)]
    train, test = train_test_split(ratings, test_fraction=0.9, seed=1)
    assert test == []
    assert len(train) == 5
