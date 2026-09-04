# collaborative-filtering-recommender

A dependency-free collaborative-filtering recommender system, trained with
stochastic gradient descent from scratch in plain Python. No numpy, no
pandas, no scikit-learn -- just the standard library.

## What it does

Given a set of `(user, item, rating)` observations, the model learns a
low-rank factorization of the ratings matrix (a Funk-SVD-style approach):
each user and each item gets a learned latent-factor vector, plus a
per-user and per-item bias term, and a predicted rating is

```
predicted_rating(user, item) = global_mean + user_bias + item_bias + dot(user_factors, item_factors)
```

Training minimizes squared error against the observed ratings via SGD with
L2 regularization, updating all parameters example-by-example over several
epochs. The result is a model that can predict how a user would rate an item
they haven't seen, and rank the best unseen items for them.

Includes:
- `cfrecommender.model.MatrixFactorizationModel` -- the SGD trainer/predictor, built entirely from Python floats/lists/dicts.
- `cfrecommender.data` -- CSV loading, a deterministic synthetic ratings generator with genuine latent structure (not pure noise), and a coverage-safe train/test splitter.
- `cfrecommender.evaluate` -- RMSE / MAE evaluation, with optional cold-start skipping.
- `cfrecommender.persistence` -- save/load a trained model to/from JSON.
- `cfrecommender.cli` -- the `cfrec` command-line tool (`train`, `recommend`, `demo`).

## Tech stack

Python 3.9+, standard library only (`csv`, `json`, `random`, `argparse`,
`dataclasses`). Dev/test dependency: `pytest`.

## Install

```bash
pip install -e ".[dev]"
```

This installs the `cfrec` console script.

## Usage

### Quick demo (no data file needed)

Generates a synthetic ratings dataset with real latent structure, trains a
model, and prints metrics plus a sample recommendation list:

```bash
$ cfrec demo
Generated 340 synthetic ratings across 40 users and 25 items
Split: 282 train / 58 test
Final train RMSE: 0.6206
Test RMSE: 1.1672
Test MAE:  0.8957

Sample top 5 recommendations for u0:
  1. i18  (predicted rating: 4.77)
  2. i7  (predicted rating: 4.71)
  3. i10  (predicted rating: 4.64)
  4. i20  (predicted rating: 4.50)
  5. i4  (predicted rating: 4.25)
```

### Train on your own data

Ratings CSV format (header required, extra columns ignored):

```csv
user,item,rating
alice,matrix,5.0
alice,titanic,1.5
...
```

A small example is included at `examples/sample_ratings.csv` (4 users, 6
movies). Training it and saving the model:

```bash
$ cfrec train --data-file examples/sample_ratings.csv --factors 4 --epochs 200 --lr 0.05 --seed 7 --save /tmp/model.json
Loaded 19 ratings (13 train / 6 test)
Final train RMSE: 0.0255
Test RMSE: 1.8209
Test MAE:  1.6105
Evaluated 6 ratings (0 skipped cold-start)
Model saved to /tmp/model.json
```

(Test RMSE is high here purely because the example dataset is tiny -- 19
ratings total -- so the held-out set is only a handful of points; see the
`demo` output above for how the model behaves on a realistically sized
dataset.)

### Get recommendations from a trained model

```bash
$ cfrec recommend --model /tmp/model.json --user alice --data-file examples/sample_ratings.csv --top-n 2
Top 2 recommendations for alice:
  1. moana  (predicted rating: 4.34)
  2. frozen  (predicted rating: 2.82)
```

`recommend` automatically excludes items the user has already rated (per the
CSV) and only considers items seen during training.

### All options

```bash
cfrec train --help
cfrec recommend --help
cfrec demo --help
```

## How the model is trained

For each observed rating `(u, i, r)`, SGD computes the prediction error
`err = r - predicted_rating(u, i)` and updates, for each latent dimension:

```
b_u  += lr * (err - reg * b_u)
b_i  += lr * (err - reg * b_i)
p_u  += lr * (err * q_i - reg * p_u)   # user factor vector
q_i  += lr * (err * p_u - reg * q_i)   # item factor vector, using the PRE-update p_u
```

Both factor-vector updates are computed from the same pre-update values (the
standard simultaneous-gradient-step formulation), not chained. Training
reshuffles the example order every epoch.

The train/test split guarantees that every user and item appearing anywhere
in the dataset keeps at least one rating in the training set -- ratings only
land in the held-out test set if their user and item both still have other
training coverage. This avoids accidentally evaluating on a completely
unseen (cold-start) user or item, which the model can't meaningfully score.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

52 tests covering: synthetic-data determinism and structure, CSV
load/save/validation, train/test split coverage guarantees, SGD convergence
(overfitting a tiny hand-checkable matrix, RMSE decreasing over epochs,
beating the naive global-mean baseline on synthetic data), cold-start
fallback behavior for unknown users/items, recommendation ranking/exclusion/
tie-breaking, model save/load round-trips, and the CLI end-to-end (including
a real subprocess invocation of `python -m cfrecommender.cli`).

## Current status

v1, functional and tested. Implemented: SGD matrix factorization with bias
terms, CSV I/O, deterministic synthetic data generation, coverage-safe
train/test splitting, RMSE/MAE evaluation (with optional cold-start
skipping), JSON model persistence, and a three-subcommand CLI (`train`,
`recommend`, `demo`).

Possible future extensions (not yet built): implicit-feedback support
(clicks/views instead of explicit ratings), a simple item-item cosine/
Pearson-similarity baseline to compare against the factorization model, and
adaptive learning-rate schedules / early stopping on validation RMSE.

## License

MIT
