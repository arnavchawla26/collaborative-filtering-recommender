import csv
import json
import os
import subprocess
import sys
import tempfile

import pytest

from cfrecommender.cli import main


def _write_ratings_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "item", "rating"])
        writer.writerows(rows)


def test_demo_command_runs_and_prints_metrics(capsys):
    rc = main(["demo", "--users", "20", "--items", "12", "--seed", "3", "--top-n", "3"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Generated" in captured.out
    assert "Test RMSE" in captured.out
    assert "Sample top 3 recommendations" in captured.out


def test_demo_command_deterministic(capsys):
    main(["demo", "--seed", "5"])
    first = capsys.readouterr().out
    main(["demo", "--seed", "5"])
    second = capsys.readouterr().out
    assert first == second


def test_train_command_with_small_csv(capsys):
    rows = []
    for u in range(6):
        for i in range(5):
            rows.append((f"u{u}", f"i{i}", 1 + (u + i) % 5))

    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "ratings.csv")
        _write_ratings_csv(data_path, rows)
        model_path = os.path.join(tmp, "model.json")

        rc = main(
            [
                "train",
                "--data-file",
                data_path,
                "--factors",
                "4",
                "--epochs",
                "15",
                "--seed",
                "1",
                "--save",
                model_path,
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "Final train RMSE" in captured.out
        assert "Test RMSE" in captured.out
        assert os.path.exists(model_path)

        with open(model_path) as f:
            saved = json.load(f)
        assert saved["format_version"] == 1


def test_train_command_too_few_ratings(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "ratings.csv")
        _write_ratings_csv(data_path, [("alice", "m1", 4.0)])
        rc = main(["train", "--data-file", data_path])
        captured = capsys.readouterr()
        assert rc == 1
        assert "at least 2 ratings" in captured.err


def test_train_then_recommend_end_to_end(capsys):
    rows = []
    for u in range(8):
        # Leave the last two items unrated by each user so there is always
        # something left to recommend after excluding already-rated items.
        for i in range(4):
            rows.append((f"u{u}", f"i{i}", 1 + (u * 2 + i) % 5))
    # A couple of ratings on i4/i5 so those items are known to the model too.
    rows.append(("u1", "i4", 3.0))
    rows.append(("u2", "i5", 2.0))

    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "ratings.csv")
        _write_ratings_csv(data_path, rows)
        model_path = os.path.join(tmp, "model.json")

        rc = main(["train", "--data-file", data_path, "--epochs", "10", "--save", model_path])
        capsys.readouterr()
        assert rc == 0

        rc = main(["recommend", "--model", model_path, "--user", "u0", "--data-file", data_path, "--top-n", "3"])
        captured = capsys.readouterr()
        assert rc == 0
        # u0 only has 2 unrated items left (i4, i5), so top-n caps at 2.
        assert "Top 2 recommendations for u0" in captured.out
        assert "i4" in captured.out
        assert "i5" in captured.out


def test_recommend_unknown_user(capsys):
    rows = [(f"u{u}", f"i{i}", 3.0) for u in range(3) for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "ratings.csv")
        _write_ratings_csv(data_path, rows)
        model_path = os.path.join(tmp, "model.json")
        main(["train", "--data-file", data_path, "--epochs", "5", "--save", model_path])
        capsys.readouterr()

        rc = main(["recommend", "--model", model_path, "--user", "ghost", "--data-file", data_path])
        captured = capsys.readouterr()
        assert rc == 1
        assert "cold-start" in captured.err


def test_no_command_prints_help(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_subprocess_module_invocation():
    # End-to-end sanity check that `python -m cfrecommender.cli demo` works
    # as a real subprocess, not just via the in-process main() call.
    result = subprocess.run(
        [sys.executable, "-m", "cfrecommender.cli", "demo", "--users", "10", "--items", "8", "--seed", "1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "Test RMSE" in result.stdout
