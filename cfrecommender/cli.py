"""Command-line interface for the collaborative-filtering recommender.

Subcommands:
  train      Train a model on a ratings CSV, report train/test RMSE and MAE,
             optionally save the trained model to JSON.
  recommend  Load a saved model and print the top-N recommended items for a
             given user.
  demo       Generate a synthetic dataset, train a model, and print sample
             metrics and recommendations end-to-end -- no input files needed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .data import generate_synthetic_ratings, load_ratings_csv, train_test_split
from .evaluate import evaluate
from .model import MatrixFactorizationModel
from .persistence import load_model, save_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfrec",
        description="Dependency-free matrix-factorization collaborative filtering recommender.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a model on a ratings CSV and report metrics.")
    train_p.add_argument("--data-file", required=True, help="Path to a CSV file with columns user,item,rating.")
    train_p.add_argument("--factors", type=int, default=10, help="Number of latent factors (default: 10).")
    train_p.add_argument("--epochs", type=int, default=20, help="Number of SGD epochs (default: 20).")
    train_p.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01).")
    train_p.add_argument("--reg", type=float, default=0.02, help="Regularization strength (default: 0.02).")
    train_p.add_argument(
        "--test-fraction", type=float, default=0.2, help="Fraction of ratings held out for evaluation (default: 0.2)."
    )
    train_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    train_p.add_argument("--save", help="Path to save the trained model as JSON.")
    train_p.add_argument("--verbose", action="store_true", help="Print per-epoch training RMSE.")

    rec_p = subparsers.add_parser("recommend", help="Get top-N item recommendations for a user from a saved model.")
    rec_p.add_argument("--model", required=True, help="Path to a model JSON file saved by 'train'.")
    rec_p.add_argument("--user", required=True, help="User ID to recommend items for.")
    rec_p.add_argument(
        "--data-file",
        required=True,
        help="Ratings CSV used to determine which items the user already rated (excluded from results) and the full item catalog.",
    )
    rec_p.add_argument("--top-n", type=int, default=5, help="Number of recommendations to return (default: 5).")

    demo_p = subparsers.add_parser("demo", help="Run an end-to-end demo on a generated synthetic dataset.")
    demo_p.add_argument("--users", type=int, default=40, help="Number of synthetic users (default: 40).")
    demo_p.add_argument("--items", type=int, default=25, help="Number of synthetic items (default: 25).")
    demo_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    demo_p.add_argument("--top-n", type=int, default=5, help="Number of sample recommendations to show (default: 5).")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    ratings = load_ratings_csv(args.data_file)
    if len(ratings) < 2:
        print("error: need at least 2 ratings to train and evaluate a model", file=sys.stderr)
        return 1

    train, test = train_test_split(ratings, test_fraction=args.test_fraction, seed=args.seed)
    if not train:
        print("error: train split is empty -- lower --test-fraction or provide more data", file=sys.stderr)
        return 1

    model = MatrixFactorizationModel(
        n_factors=args.factors,
        learning_rate=args.lr,
        regularization=args.reg,
        n_epochs=args.epochs,
        seed=args.seed,
    )
    model.fit(train, verbose=args.verbose)

    print(f"Loaded {len(ratings)} ratings ({len(train)} train / {len(test)} test)")
    print(f"Final train RMSE: {model.train_rmse_history[-1]:.4f}")

    if test:
        result = evaluate(model, test)
        print(f"Test RMSE: {result.rmse:.4f}")
        print(f"Test MAE:  {result.mae:.4f}")
        print(f"Evaluated {result.n_evaluated} ratings ({result.n_skipped_cold_start} skipped cold-start)")
    else:
        print("No test ratings were held out (dataset too small or --test-fraction 0); skipping evaluation.")

    if args.save:
        save_model(model, args.save)
        print(f"Model saved to {args.save}")

    return 0


def _run_recommend(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    ratings = load_ratings_csv(args.data_file)

    if args.user not in model.known_users():
        print(
            f"error: user {args.user!r} was not seen during training (cold-start users are not supported)",
            file=sys.stderr,
        )
        return 1

    all_items = sorted({r.item for r in ratings})
    already_rated = {r.item for r in ratings if r.user == args.user}

    recs = model.recommend(args.user, all_items, exclude_items=already_rated, top_n=args.top_n)
    if not recs:
        print(f"No recommendations available for user {args.user!r} (all known items already rated).")
        return 0

    print(f"Top {len(recs)} recommendations for {args.user}:")
    for rank, (item, score) in enumerate(recs, start=1):
        print(f"  {rank}. {item}  (predicted rating: {score:.2f})")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    ratings = generate_synthetic_ratings(n_users=args.users, n_items=args.items, seed=args.seed)
    train, test = train_test_split(ratings, test_fraction=0.2, seed=args.seed)

    model = MatrixFactorizationModel(
        n_factors=8, learning_rate=0.02, regularization=0.02, n_epochs=25, seed=args.seed
    )
    model.fit(train)

    print(f"Generated {len(ratings)} synthetic ratings across {args.users} users and {args.items} items")
    print(f"Split: {len(train)} train / {len(test)} test")
    print(f"Final train RMSE: {model.train_rmse_history[-1]:.4f}")

    result = evaluate(model, test)
    print(f"Test RMSE: {result.rmse:.4f}")
    print(f"Test MAE:  {result.mae:.4f}")

    demo_user = model.known_users()[0]
    all_items = sorted({r.item for r in ratings})
    already_rated = {r.item for r in ratings if r.user == demo_user}
    recs = model.recommend(demo_user, all_items, exclude_items=already_rated, top_n=args.top_n)

    print(f"\nSample top {len(recs)} recommendations for {demo_user}:")
    for rank, (item, score) in enumerate(recs, start=1):
        print(f"  {rank}. {item}  (predicted rating: {score:.2f})")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _run_train(args)
    if args.command == "recommend":
        return _run_recommend(args)
    if args.command == "demo":
        return _run_demo(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
