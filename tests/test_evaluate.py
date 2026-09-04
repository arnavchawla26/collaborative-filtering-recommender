import pytest

from cfrecommender.data import Rating
from cfrecommender.evaluate import evaluate
from cfrecommender.model import MatrixFactorizationModel


def _hand_checkable_model():
    # A model with a fixed, hand-computable state: no latent factors
    # contribute (n_factors=1 with factors forced to zero after fit isn't
    # directly settable, so instead we fit briefly then override state
    # directly for a fully deterministic hand-checked prediction surface).
    model = MatrixFactorizationModel(n_factors=1, n_epochs=1, seed=0)
    model.fit([Rating("a", "x", 3.0)])
    model.global_mean = 3.0
    model.user_bias = {"alice": 1.0, "bob": -1.0}
    model.item_bias = {"m1": 0.5, "m2": -0.5}
    model.user_factors = {"alice": [0.0], "bob": [0.0]}
    model.item_factors = {"m1": [0.0], "m2": [0.0]}
    return model


def test_evaluate_hand_computed_rmse_and_mae():
    model = _hand_checkable_model()
    # predict(alice, m1) = 3.0 + 1.0 + 0.5 = 4.5
    # predict(alice, m2) = 3.0 + 1.0 - 0.5 = 3.5
    # predict(bob, m1)   = 3.0 - 1.0 + 0.5 = 2.5
    test_ratings = [
        Rating("alice", "m1", 5.0),  # err = 0.5
        Rating("alice", "m2", 3.0),  # err = -0.5
        Rating("bob", "m1", 2.0),  # err = -0.5
    ]
    result = evaluate(model, test_ratings)

    expected_mae = (0.5 + 0.5 + 0.5) / 3
    expected_rmse = ((0.5**2 + 0.5**2 + 0.5**2) / 3) ** 0.5

    assert result.mae == pytest.approx(expected_mae)
    assert result.rmse == pytest.approx(expected_rmse)
    assert result.n_evaluated == 3
    assert result.n_skipped_cold_start == 0


def test_evaluate_perfect_predictions_give_zero_error():
    model = _hand_checkable_model()
    test_ratings = [
        Rating("alice", "m1", 4.5),
        Rating("bob", "m1", 2.5),
    ]
    result = evaluate(model, test_ratings)
    assert result.rmse == pytest.approx(0.0)
    assert result.mae == pytest.approx(0.0)


def test_evaluate_cold_start_not_skipped_by_default():
    model = _hand_checkable_model()
    test_ratings = [Rating("stranger", "unknown-item", 3.0)]
    result = evaluate(model, test_ratings)
    # Falls back to global_mean = 3.0, so err = 0 exactly here.
    assert result.n_evaluated == 1
    assert result.n_skipped_cold_start == 0
    assert result.rmse == pytest.approx(0.0)


def test_evaluate_cold_start_skipped_when_requested():
    model = _hand_checkable_model()
    test_ratings = [
        Rating("alice", "m1", 5.0),
        Rating("stranger", "unknown-item", 1.0),
    ]
    result = evaluate(model, test_ratings, skip_cold_start=True)
    assert result.n_evaluated == 1
    assert result.n_skipped_cold_start == 1


def test_evaluate_empty_test_set_raises():
    model = _hand_checkable_model()
    with pytest.raises(ValueError, match="No ratings"):
        evaluate(model, [])


def test_evaluate_all_cold_start_skipped_raises():
    model = _hand_checkable_model()
    test_ratings = [Rating("stranger", "unknown-item", 1.0)]
    with pytest.raises(ValueError, match="No ratings"):
        evaluate(model, test_ratings, skip_cold_start=True)
