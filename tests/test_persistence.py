import os
import tempfile

import pytest

from cfrecommender.data import generate_synthetic_ratings, train_test_split
from cfrecommender.model import MatrixFactorizationModel
from cfrecommender.persistence import load_model, save_model


def test_save_unfitted_model_raises():
    model = MatrixFactorizationModel()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError):
            save_model(model, os.path.join(tmp, "model.json"))


def test_save_and_load_round_trip_predictions_identical():
    ratings = generate_synthetic_ratings(n_users=25, n_items=15, seed=6)
    train, test = train_test_split(ratings, test_fraction=0.2, seed=6)

    model = MatrixFactorizationModel(n_factors=6, n_epochs=15, seed=6)
    model.fit(train)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.json")
        save_model(model, path)
        loaded = load_model(path)

        for r in test:
            original_pred = model.predict(r.user, r.item)
            loaded_pred = loaded.predict(r.user, r.item)
            assert original_pred == pytest.approx(loaded_pred)

        assert loaded.known_users() == model.known_users()
        assert loaded.known_items() == model.known_items()
        assert loaded.global_mean == pytest.approx(model.global_mean)
        assert loaded.n_factors == model.n_factors
        assert loaded.train_rmse_history == pytest.approx(model.train_rmse_history)


def test_load_model_bad_format_version_raises():
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.json")
        with open(path, "w") as f:
            json.dump({"format_version": 999}, f)
        with pytest.raises(ValueError, match="format_version"):
            load_model(path)


def test_loaded_model_can_recommend():
    ratings = generate_synthetic_ratings(n_users=20, n_items=12, seed=8)
    model = MatrixFactorizationModel(n_factors=5, n_epochs=10, seed=8)
    model.fit(ratings)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.json")
        save_model(model, path)
        loaded = load_model(path)

        user = loaded.known_users()[0]
        recs = loaded.recommend(user, loaded.known_items(), top_n=3)
        assert len(recs) == 3
