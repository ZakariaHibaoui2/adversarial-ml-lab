"""Small NumPy classifiers with analytic gradients w.r.t. both parameters and *inputs* (needed for attacks)."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


class MLP:
    """x -> Dense(hidden) -> ReLU -> Dense(classes) -> softmax. hidden=0 gives softmax regression."""

    def __init__(self, n_in: int, n_classes: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        if hidden:
            self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, n_classes))
        else:
            self.W2 = rng.normal(0, 0.01, (n_in, n_classes))
        self.b2 = np.zeros(n_classes)
        self.k = n_classes

    # forward / backward ----------------------------------------------------------------------------
    def _forward(self, x):
        if self.hidden:
            a1 = x @ self.W1 + self.b1
            h = np.maximum(a1, 0)
        else:
            a1, h = None, x
        return a1, h, softmax(h @ self.W2 + self.b2)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[2]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(x)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    def gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
        """Mean cross-entropy gradients for parameters, and per-example gradient w.r.t. the input."""
        a1, h, p = self._forward(x)
        n = len(x)
        d_logits = (p - one_hot(y, self.k)) / n
        grads = {"W2": h.T @ d_logits, "b2": d_logits.sum(0)}
        d_h = d_logits @ self.W2.T
        if self.hidden:
            d_a1 = d_h * (a1 > 0)
            grads["W1"] = x.T @ d_a1
            grads["b1"] = d_a1.sum(0)
            d_x = d_a1 @ self.W1.T
        else:
            d_x = d_h
        return grads, d_x * n          # per-example input gradient (undo the batch mean)

    def input_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.gradients(x, y)[1]

    # training --------------------------------------------------------------------------------------
    def params(self) -> dict:
        names = ["W1", "b1", "W2", "b2"] if self.hidden else ["W2", "b2"]
        return {n: getattr(self, n) for n in names}

    def fit(self, x, y, epochs: int = 30, lr: float = 1e-2, batch: int = 64, seed: int = 0,
            adversary=None, adv_ratio: float = 0.5) -> MLP:
        """Adam. If ``adversary(model, x, y)`` is given, each batch is partly replaced by adversarial examples."""
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.params().items()}
        v = {k: np.zeros_like(v) for k, v in self.params().items()}
        t = 0
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for i in range(0, len(x), batch):
                idx = order[i:i + batch]
                xb, yb = x[idx], y[idx]
                if adversary is not None:
                    k = int(len(xb) * adv_ratio)
                    if k:
                        xb = xb.copy()
                        xb[:k] = adversary(self, xb[:k], yb[:k])
                grads, _ = self.gradients(xb, yb)
                t += 1
                for name, p in self.params().items():
                    g = grads[name]
                    m[name] = 0.9 * m[name] + 0.1 * g
                    v[name] = 0.999 * v[name] + 0.001 * g * g
                    mh, vh = m[name] / (1 - 0.9 ** t), v[name] / (1 - 0.999 ** t)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self

    def accuracy(self, x, y) -> float:
        return float((self.predict(x) == y).mean())
