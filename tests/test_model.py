import pytest

from cfrecommender.data import Rating, generate_synthetic_ratings, train_test_split
from cfrecommender.model import MatrixFactorizationModel


def _toy_ratings():
    # A tiny, fully-specified 3-user x 3-item ratings matrix so the model
    # can be checked against near-exact overfit predictions.
    return [
        Rating("alice", "matrix", 5.0),
        Rating("alice", "titanic", 1.0),
        Rating("alice", "up", 4.0),
        Rating("bob", "matrix", 1.0),
        Rating("bob", "titanic", 5.0),
        Rating("bob", "up", 2.0),
        Rating("carol", "matrix", 4.0),
        Rating("carol", "titanic", 2.0),
        Rating("carol", "up", 5.0),
    ]


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        MatrixFactorizationModel(n_factors=0)
    with pytest.raises(ValueError):
        MatrixFactorizationModel(learning_rate=0)
    with pytest.raises(ValueError):
        MatrixFactorizationModel(regularization=-1)
    with pytest.raises(ValueError):
        MatrixFactorizationModel(n_epochs=0)


def test_predict_before_fit_raises():
    model = MatrixFactorizationModel()
    with pytest.raises(RuntimeError):
        model.predict("alice", "matrix")


def test_recommend_before_fit_raises():
    model = MatrixFactorizationModel()
    with pytest.raises(RuntimeError):
        model.recommend("alice", ["matrix"])


def test_fit_empty_ratings_raises():
    model = MatrixFactorizationModel()
    with pytest.raises(ValueError):
        model.fit([])


def test_fit_overfits_small_matrix_closely():
    # With enough factors and epochs, the model should nearly memorize a
    # tiny fully-observed matrix.
    ratings = _toy_ratings()
    model = MatrixFactorizationModel(n_factors=5, learning_rate=0.05, regularization=0.0, n_epochs=400, seed=1)
    model.fit(ratings)

    for r in ratings:
        pred = model.predict(r.user, r.item)
        assert abs(pred - r.rating) < 0.3, f"{r.user},{r.item}: predicted {pred}, actual {r.rating}"


def test_train_rmse_decreases_over_training():
    ratings = _toy_ratings()
    model = MatrixFactorizationModel(n_factors=5, learning_rate=0.05, regularization=0.0, n_epochs=200, seed=1)
    model.fit(ratings)

    history = model.train_rmse_history
    assert len(history) == 200
    # Compare average of the first 10 epochs against the last 10: should be
    # a clear, large improvement even allowing for per-epoch SGD noise.
    early_avg = sum(history[:10]) / 10
    late_avg = sum(history[-10:]) / 10
    assert late_avg < early_avg * 0.5


def test_predict_unknown_user_falls_back_to_item_bias_and_mean():
    model = MatrixFactorizationModel(n_factors=3, n_epochs=10, seed=2)
    model.fit(_toy_ratings())
    pred = model.predict("dave", "matrix")  # dave never seen during training
    # Should not crash, and should equal global_mean + item bias for matrix
    # (no user bias/factors contribute for an unknown user).
    expected = model.global_mean + model.item_bias["matrix"]
    assert pred == pytest.approx(expected)


def test_predict_unknown_item_falls_back_to_user_bias_and_mean():
    model = MatrixFactorizationModel(n_factors=3, n_epochs=10, seed=2)
    model.fit(_toy_ratings())
    pred = model.predict("alice", "avatar")  # avatar never seen during training
    expected = model.global_mean + model.user_bias["alice"]
    assert pred == pytest.approx(expected)


def test_predict_completely_unknown_user_and_item_returns_global_mean():
    model = MatrixFactorizationModel(n_factors=3, n_epochs=10, seed=2)
    model.fit(_toy_ratings())
    pred = model.predict("zeynep", "brand-new-movie")
    assert pred == pytest.approx(model.global_mean)


def test_fit_is_deterministic_given_seed():
    ratings = _toy_ratings()
    m1 = MatrixFactorizationModel(n_factors=4, n_epochs=30, seed=123).fit(ratings)
    m2 = MatrixFactorizationModel(n_factors=4, n_epochs=30, seed=123).fit(ratings)
    for u in m1.known_users():
        assert m1.user_factors[u] == m2.user_factors[u]
        assert m1.user_bias[u] == m2.user_bias[u]
    for i in m1.known_items():
        assert m1.item_factors[i] == m2.item_factors[i]


def test_fit_different_seed_gives_different_factors():
    ratings = _toy_ratings()
    m1 = MatrixFactorizationModel(n_factors=4, n_epochs=5, seed=1).fit(ratings)
    m2 = MatrixFactorizationModel(n_factors=4, n_epochs=5, seed=2).fit(ratings)
    assert m1.user_factors["alice"] != m2.user_factors["alice"]


def test_recommend_excludes_already_rated_and_sorts_descending():
    model = MatrixFactorizationModel(n_factors=5, learning_rate=0.05, regularization=0.0, n_epochs=400, seed=1)
    model.fit(_toy_ratings())

    # Alice loved 'matrix' (5.0) and 'up' (4.0), disliked 'titanic' (1.0).
    # If we exclude 'matrix' (already rated) and ask for recommendations
    # among all three items, 'up' should outrank 'titanic'.
    recs = model.recommend("alice", ["matrix", "titanic", "up"], exclude_items=["matrix"], top_n=5)
    rec_items = [item for item, _ in recs]
    assert "matrix" not in rec_items
    assert rec_items[0] == "up"

    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_top_n_limits_results():
    model = MatrixFactorizationModel(n_factors=3, n_epochs=20, seed=1)
    model.fit(_toy_ratings())
    recs = model.recommend("alice", ["matrix", "titanic", "up"], top_n=2)
    assert len(recs) == 2


def test_recommend_invalid_top_n():
    model = MatrixFactorizationModel(n_factors=3, n_epochs=5, seed=1)
    model.fit(_toy_ratings())
    with pytest.raises(ValueError):
        model.recommend("alice", ["matrix"], top_n=0)


def test_recommend_tie_break_by_item_id():
    # Force a tie by predicting for an unknown user against two unknown
    # items -- both fall back to the same global-mean prediction.
    model = MatrixFactorizationModel(n_factors=3, n_epochs=5, seed=1)
    model.fit(_toy_ratings())
    recs = model.recommend("nobody", ["zzz", "aaa", "mmm"], top_n=3)
    items = [item for item, _ in recs]
    assert items == ["aaa", "mmm", "zzz"]


def test_known_users_and_items_sorted():
    model = MatrixFactorizationModel(n_factors=3, n_epochs=5, seed=1)
    model.fit(_toy_ratings())
    assert model.known_users() == sorted(model.known_users())
    assert model.known_items() == sorted(model.known_items())
    assert set(model.known_users()) == {"alice", "bob", "carol"}
    assert set(model.known_items()) == {"matrix", "titanic", "up"}


def test_fit_on_synthetic_data_beats_naive_mean_baseline():
    # Sanity end-to-end check: on a reasonably sized synthetic dataset with
    # real latent structure, the trained model's test RMSE should beat the
    # trivial "always predict the global mean" baseline.
    ratings = generate_synthetic_ratings(n_users=60, n_items=40, seed=21)
    train, test = train_test_split(ratings, test_fraction=0.2, seed=21)

    model = MatrixFactorizationModel(n_factors=8, learning_rate=0.02, regularization=0.02, n_epochs=30, seed=21)
    model.fit(train)

    model_sq_err = sum((model.predict(r.user, r.item) - r.rating) ** 2 for r in test)
    model_rmse = (model_sq_err / len(test)) ** 0.5

    baseline_mean = sum(r.rating for r in train) / len(train)
    baseline_sq_err = sum((baseline_mean - r.rating) ** 2 for r in test)
    baseline_rmse = (baseline_sq_err / len(test)) ** 0.5

    assert model_rmse < baseline_rmse
