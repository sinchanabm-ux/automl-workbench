import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# Classification libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Keeps training inside Render free-tier CPU/RAM limits
MAX_ROWS = 10000      # sample larger datasets before cross-validation
SMALL_DATA_ROWS = 2000  # use 5 folds up to this size, 3 folds above it


def _sample_rows(X, y):
    """Randomly sample down to MAX_ROWS (works for DataFrame/Series, arrays, sparse)."""
    if X.shape[0] <= MAX_ROWS:
        return X, y
    idx = np.random.RandomState(42).choice(X.shape[0], MAX_ROWS, replace=False)
    X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    return X, y


def _choose_folds(y, problem_type):
    folds = 5 if len(y) <= SMALL_DATA_ROWS else 3
    if problem_type == "classification":
        # Stratified CV fails if any class has fewer rows than there are folds
        smallest_class = pd.Series(y).value_counts().min()
        if smallest_class < 2:
            raise ValueError(
                "Each class in the target needs at least 2 rows. "
                "Choose a different target column or clean the data."
            )
        folds = min(folds, int(smallest_class))
    return folds


def train_models(X, y, problem_type):
    X, y = _sample_rows(X, y)
    folds = _choose_folds(y, problem_type)

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=50, max_depth=12, n_jobs=1, random_state=42
            ),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=50, max_depth=12, n_jobs=1, random_state=42
            ),
        }

    results = {}
    for name, model in models.items():
        # cross_val_score fits clones of the model internally, so no extra
        # model.fit(X, y) is needed (the fitted models were never used).
        scores = cross_val_score(model, X, y, cv=folds)
        results[name] = scores.mean()

    return results