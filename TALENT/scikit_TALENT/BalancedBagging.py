from __future__ import annotations

import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


def _safe_row_index(X, idx: np.ndarray):
    """Index rows for numpy/pandas/sparse etc."""
    if hasattr(X, "iloc"):  # pandas
        return X.iloc[idx]
    return X[idx]


def _resolve_max_samples(max_samples, n_samples: int) -> int:
    if isinstance(max_samples, float):
        if not (0.0 < max_samples <= 1.0):
            raise ValueError("If float, max_samples must be in (0, 1].")
        return max(1, int(np.floor(max_samples * n_samples)))
    if isinstance(max_samples, int):
        if not (1 <= max_samples <= n_samples):
            raise ValueError("If int, max_samples must be in [1, n_samples].")
        return max_samples
    raise TypeError("max_samples must be int or float.")


def _fit_estimator_maybe_with_cats(est, X_sub, y_sub, categorical_indicator):
    """
    Try to call est.fit with categorical indicators if provided.
    We do NOT assume sklearn-style signatures; we try a couple common patterns.
    """
    if categorical_indicator is None:
        est.fit(X_sub, y_sub)
        return

    # 1) common keyword form: fit(X, y, categorical_indicator=...)
    try:
        est.fit(X_sub, y_sub, categorical_indicator=categorical_indicator)
        return
    except TypeError:
        pass

    # 2) some libs accept third positional arg: fit(X, y, categorical_indicator)
    try:
        est.fit(X_sub, y_sub, categorical_indicator)
        return
    except TypeError:
        pass

    # 3) fallback: ignore categorical indicator
    est.fit(X_sub, y_sub)


class UnderSamplingBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Bagging + undersampling (BalancedBagging-like):
      1) bootstrap sample rows (bagging)
      2) undersample majority classes within that bootstrap sample
      3) fit base estimator
      4) hard-vote predict

    Base estimator requirements:
      - must implement fit(X, y) and predict(X)
      - NO need for get_params / set_params / classes_ / predict_proba

    Categorical indicators:
      - You can pass `categorical_indicator` to fit(...).
      - If provided, it will be forwarded to base_estimator.fit when possible.

    IMPORTANT:
      - If base_estimator is NOT sklearn-cloneable, pass `estimator_factory`
        that returns a *fresh* estimator each time.
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        random_state=None,
        *,
        estimator_factory=None,          # callable -> new estimator each time
        bootstrap: bool = True,
        max_samples=1.0,                 # int or float fraction
        bootstrap_max_attempts: int = 10,
        bootstrap_minority: bool = False, # allow replacement when class count == min_count
        categorical_indicator=None,       # optional default; can be overridden in fit()
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_factory = estimator_factory
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.bootstrap_max_attempts = bootstrap_max_attempts
        self.bootstrap_minority = bootstrap_minority
        self.categorical_indicator = categorical_indicator

    def _new_estimator(self):
        if self.estimator_factory is not None:
            est = self.estimator_factory()
            if est is None:
                raise ValueError("estimator_factory() returned None; it must return a new estimator instance.")
            return est

        if self.base_estimator is None:
            raise ValueError("Provide either base_estimator or estimator_factory.")

        try:
            return copy.deepcopy(self.base_estimator)
        except Exception as e:
            raise TypeError(
                "base_estimator is not sklearn-cloneable and could not be deepcopy'ed.\n"
                "Provide `estimator_factory` that returns a fresh estimator each time, e.g.\n"
                "  UnderSamplingBaggingClassifier(estimator_factory=lambda: TabFlex(...), n_estimators=...)"
            ) from e

    def fit(self, X, y, categorical_indicator=None):
        # if user didn't pass it to fit, fall back to the default from __init__
        if categorical_indicator is None:
            categorical_indicator = self.categorical_indicator

        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()

        if y.size == 0:
            raise ValueError("Empty y.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")

        # Minimal validation: integer-like labels for voting
        if not np.issubdtype(y.dtype, np.integer):
            if np.all(np.isfinite(y)) and np.all(y == np.round(y)):
                y = y.astype(int)
            else:
                raise ValueError("y must contain integer class labels (already encoded).")

        self.classes_ = np.unique(y)  # sorted
        if self.classes_.size < 2:
            raise ValueError("Need at least 2 classes to fit a classifier.")
        self.n_classes_ = int(self.classes_.size)

        rng = check_random_state(self.random_state)
        n_samples = int(y.shape[0])
        boot_size = _resolve_max_samples(self.max_samples, n_samples)

        # Full-data per-class indices (fallback if a bootstrap ends up single-class)
        full_class_to_idx = {c: np.flatnonzero(y == c) for c in self.classes_}
        full_counts = np.array([len(full_class_to_idx[c]) for c in self.classes_], dtype=int)
        full_min_count = int(full_counts.min())

        self.estimators_ = []

        for _ in range(self.n_estimators):
            # 1) Bootstrap sample of rows
            if self.bootstrap:
                boot_idx = None
                for _attempt in range(max(1, int(self.bootstrap_max_attempts))):
                    cand = rng.choice(n_samples, size=boot_size, replace=True)
                    if np.unique(y[cand]).size >= 2:
                        boot_idx = cand
                        break
                if boot_idx is None:
                    boot_idx = np.arange(n_samples, dtype=int)  # fallback
            else:
                boot_idx = rng.choice(n_samples, size=boot_size, replace=False)

            # 2) Undersample majority classes inside the bootstrap sample
            y_boot = y[boot_idx]
            present_classes, present_counts = np.unique(y_boot, return_counts=True)

            if present_classes.size < 2:
                # fallback to balanced sample from full data
                parts = []
                for c in self.classes_:
                    idx_c = full_class_to_idx[c]
                    if len(idx_c) > full_min_count:
                        pick = rng.choice(idx_c, size=full_min_count, replace=False)
                    else:
                        pick = (
                            rng.choice(idx_c, size=full_min_count, replace=True)
                            if self.bootstrap_minority
                            else idx_c
                        )
                    parts.append(pick)
                sampled_idx = np.concatenate(parts)
            else:
                min_count = int(present_counts.min())
                parts = []
                for c in present_classes:
                    idx_c_in_boot = boot_idx[np.flatnonzero(y_boot == c)]
                    if len(idx_c_in_boot) > min_count:
                        pick = rng.choice(idx_c_in_boot, size=min_count, replace=False)
                    else:
                        pick = (
                            rng.choice(idx_c_in_boot, size=min_count, replace=True)
                            if self.bootstrap_minority
                            else idx_c_in_boot
                        )
                    parts.append(pick)
                sampled_idx = np.concatenate(parts)

            rng.shuffle(sampled_idx)

            X_sub = _safe_row_index(X, sampled_idx)
            y_sub = y[sampled_idx]

            est = self._new_estimator()
            _fit_estimator_maybe_with_cats(est, X_sub, y_sub, categorical_indicator)
            self.estimators_.append(est)

        return self

    def predict(self, X):
        if not hasattr(self, "estimators_") or len(self.estimators_) == 0:
            raise ValueError("This UnderSamplingBaggingClassifier instance is not fitted yet.")

        pred_indices = []
        for est in self.estimators_:
            p = np.asarray(est.predict(X))

            if not np.issubdtype(p.dtype, np.integer):
                if np.all(np.isfinite(p)) and np.all(p == np.round(p)):
                    p = p.astype(int)
                else:
                    raise ValueError("Base estimator predict() must return integer class labels.")

            # map labels -> indices via searchsorted on sorted self.classes_
            idx = np.searchsorted(self.classes_, p)
            bad = (idx < 0) | (idx >= self.n_classes_) | (self.classes_[idx] != p)
            if np.any(bad):
                raise ValueError("Base estimator predicted a class label not seen during fit().")

            pred_indices.append(idx)

        pred_indices = np.stack(pred_indices, axis=0)  # (n_estimators, n_samples)

        n_samples = pred_indices.shape[1]
        out_idx = np.empty(n_samples, dtype=int)

        for j in range(n_samples):
            votes = np.bincount(pred_indices[:, j], minlength=self.n_classes_)
            out_idx[j] = int(np.argmax(votes))  # tie -> lowest index

        return self.classes_[out_idx]


if __name__ == "__main__":
    # Smoke test on a standard sklearn dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=8,
        n_redundant=2,
        weights=[0.93, 0.07],
        flip_y=0.01,
        random_state=0,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    clf = UnderSamplingBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=25,
        random_state=0,
        bootstrap=True,
        max_samples=1.0,
    )

    # Example categorical_indicator usage (here: none; replace with your real indicator)
    categorical_indicator = None

    clf.fit(X_train, y_train, categorical_indicator=categorical_indicator)
    y_pred = clf.predict(X_test)

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
