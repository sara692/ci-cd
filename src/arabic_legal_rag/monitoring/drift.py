import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def centroid_cosine(ref: np.ndarray, live: np.ndarray) -> tuple[float, float]:
    c = ref.mean(axis=0)
    c /= np.linalg.norm(c)
    return float((ref @ c).mean()), float((live @ c).mean())  # vectors are already L2-normalised


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    d = ((z[:, None] - z[None]) ** 2).sum(-1)
    gamma = 1.0 / np.median(d[d > 0])  # median heuristic

    def k(a, b):
        return np.exp(-gamma * ((a[:, None] - b[None]) ** 2).sum(-1))

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def domain_auc(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    return float(
        cross_val_score(LogisticRegression(max_iter=1000), z, lab, cv=5, scoring="roc_auc").mean()
    )
