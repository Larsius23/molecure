"""PLS regression with a fold-safe preprocessing recipe.

Everything that is LEARNED from data (which columns survive the sigma filter, the column
scales, the means, the PLS components) happens inside `fit_pls`, so calling it on the
training part of a cross-validation fold means nothing leaks from the held-out compounds.

PLS1 (one activity column) by the classical NIPALS algorithm, written out here (~25 lines)
rather than taken from scikit-learn because we also need:
  - predictions for EVERY number of components from a single fit (for the q2-vs-components curve),
  - the coefficients rotated back to the original columns (for contour maps), and
  - a model that can be saved and re-applied to screening molecules later.
A test checks this implementation against scikit-learn.
"""

from dataclasses import dataclass

import numpy as np

SCALINGS = ("none", "autoscale", "block")


class ModelError(Exception):
    """The model cannot be fitted with this data and these settings."""


@dataclass(frozen=True, eq=False)
class Recipe:
    """How to prepare the field columns before PLS."""

    min_sigma: float  # drop columns whose standard deviation is below this (kcal/mol)
    scaling: str  # none | autoscale (unit variance per column) | block (equal weight per field)
    blocks: np.ndarray  # field-block index of every column, shape (n_columns,)

    def __post_init__(self) -> None:
        if self.scaling not in SCALINGS:
            raise ModelError(f"scaling must be one of {SCALINGS}, not {self.scaling!r}")


@dataclass(frozen=True, eq=False)
class FittedPLS:
    keep: np.ndarray  # (p,) bool: columns that survived the sigma filter
    scale: np.ndarray  # (p_kept,) multiplicative column scale
    x_mean: np.ndarray  # (p_kept,)
    y_mean: float
    weights: np.ndarray  # W (p_kept, A)
    loadings: np.ndarray  # P (p_kept, A)
    y_loadings: np.ndarray  # q (A,)

    @property
    def n_components(self) -> int:
        return self.weights.shape[1]

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float)[:, self.keep] - self.x_mean) * self.scale

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Predictions for 1, 2, ..., A components: array of shape (n, A)."""
        Xa = self._prepare(X)
        out = np.empty((len(Xa), self.n_components))
        running = np.full(len(Xa), self.y_mean)
        for a in range(self.n_components):
            t = Xa @ self.weights[:, a]
            running = running + self.y_loadings[a] * t
            out[:, a] = running
            Xa = Xa - np.outer(t, self.loadings[:, a])
        return out

    def predict(self, X: np.ndarray, n_components: int) -> np.ndarray:
        k = min(n_components, self.n_components)
        return self.predict_all(X)[:, k - 1]

    def coefficients(self, n_components: int) -> tuple[np.ndarray, float]:
        """The conventional QSAR equation in ORIGINAL column units: (b, intercept).

        y_hat = intercept + X @ b ; columns removed by the filter get coefficient 0.
        This is the vector that is contoured in a CoMFA coefficient map.
        """
        k = min(n_components, self.n_components)
        W, P, q = self.weights[:, :k], self.loadings[:, :k], self.y_loadings[:k]
        b_scaled = W @ np.linalg.solve(P.T @ W, q)
        b_kept = self.scale * b_scaled
        b = np.zeros(len(self.keep))
        b[self.keep] = b_kept
        return b, float(self.y_mean - self.x_mean @ b_kept)

    # ---- saving / loading (numpy arrays only, no pickle) ----
    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "keep": self.keep,
            "scale": self.scale,
            "x_mean": self.x_mean,
            "y_mean": np.array(self.y_mean),
            "weights": self.weights,
            "loadings": self.loadings,
            "y_loadings": self.y_loadings,
        }

    @classmethod
    def from_arrays(cls, arrays) -> "FittedPLS":
        return cls(
            keep=np.asarray(arrays["keep"], dtype=bool),
            scale=np.asarray(arrays["scale"]),
            x_mean=np.asarray(arrays["x_mean"]),
            y_mean=float(arrays["y_mean"]),
            weights=np.asarray(arrays["weights"]),
            loadings=np.asarray(arrays["loadings"]),
            y_loadings=np.asarray(arrays["y_loadings"]),
        )


def fit_pls(X: np.ndarray, y: np.ndarray, recipe: Recipe, max_components: int) -> FittedPLS:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y):
        raise ModelError(f"X has shape {X.shape} but y has {len(y)} values")
    if len(recipe.blocks) != X.shape[1]:
        raise ModelError("the recipe describes a different number of columns than X has")
    if len(y) < 3:
        raise ModelError("at least 3 compounds are needed to fit a model")

    # 1. column filter: constant columns always go; so does anything below min_sigma
    sigma = X.std(axis=0, ddof=1)
    keep = (sigma >= recipe.min_sigma) & (sigma > 1e-12)
    if not keep.any():
        raise ModelError(
            f"no column has a standard deviation of at least {recipe.min_sigma}; "
            "lower pls.min_sigma or check the fields"
        )
    Xk = X[:, keep]
    sigma_k = sigma[keep]

    # 2. centring and scaling (all learned from THIS training set only)
    x_mean = Xk.mean(axis=0)
    scale = np.ones(Xk.shape[1])
    if recipe.scaling == "autoscale":
        scale = 1.0 / sigma_k
    elif recipe.scaling == "block":
        blocks = recipe.blocks[keep]
        for b in np.unique(blocks):
            in_block = blocks == b
            scale[in_block] = 1.0 / np.sqrt(np.sum(sigma_k[in_block] ** 2))  # equal total variance
    Xa = (Xk - x_mean) * scale
    y_mean = float(y.mean())
    yc = y - y_mean

    # 3. PLS1 by NIPALS
    W, P, q = [], [], []
    for _ in range(max_components):
        w = Xa.T @ yc
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            break
        w = w / norm
        t = Xa @ w
        tt = float(t @ t)
        if tt < 1e-12:
            break
        p = Xa.T @ t / tt
        W.append(w)
        P.append(p)
        q.append(float(yc @ t) / tt)
        Xa = Xa - np.outer(t, p)
    if not W:
        raise ModelError("PLS found no component: the fields do not vary with activity")
    return FittedPLS(
        keep=keep,
        scale=scale,
        x_mean=x_mean,
        y_mean=y_mean,
        weights=np.column_stack(W),
        loadings=np.column_stack(P),
        y_loadings=np.array(q),
    )
