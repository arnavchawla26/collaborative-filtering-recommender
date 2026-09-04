"""collaborative-filtering-recommender: a dependency-free (no numpy/pandas)
SGD-trained matrix-factorization recommender system, implemented with plain
Python floats and lists.
"""

from .data import Rating, generate_synthetic_ratings, load_ratings_csv, save_ratings_csv, train_test_split
from .evaluate import EvaluationResult, evaluate
from .model import MatrixFactorizationModel
from .persistence import load_model, save_model

__version__ = "0.1.0"

__all__ = [
    "Rating",
    "generate_synthetic_ratings",
    "load_ratings_csv",
    "save_ratings_csv",
    "train_test_split",
    "EvaluationResult",
    "evaluate",
    "MatrixFactorizationModel",
    "load_model",
    "save_model",
    "__version__",
]
