"""Defenses: adversarial training and feature-squeezing detection (Xu et al., 2018)."""
from __future__ import annotations

from functools import partial

import numpy as np

from .attacks import pgd
from .models import MLP


def adversarially_trained(x, y, eps: float, hidden: int = 64, epochs: int = 30, steps: int = 7, seed: int = 0) -> MLP:
    """Madry-style training: half of every batch is replaced by PGD examples generated on the fly."""
    adversary = partial(_pgd_adv, eps=eps, steps=steps)
    return MLP(x.shape[1], int(y.max()) + 1, hidden, seed).fit(x, y, epochs=epochs, adversary=adversary, seed=seed)


def _pgd_adv(model, xb, yb, eps, steps):
    return pgd(model, xb, yb, eps=eps, steps=steps)


def squeeze_bits(x: np.ndarray, bits: int = 2) -> np.ndarray:
    levels = 2 ** bits - 1
    return np.round(x * levels) / levels


def median_smooth(x: np.ndarray, side: int = 8) -> np.ndarray:
    imgs = x.reshape(-1, side, side)
    padded = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    return np.median(windows, axis=(-2, -1)).reshape(len(x), -1)


class ManifoldDetector:
    """Two complementary signals, OR-combined, each calibrated on held-out clean data:

    * **instability** - fraction of predictions that change under small random noise. Weak adversarial
      examples sit just across a decision boundary, so noise often flips them back.
    * **off-manifold distance** - PCA reconstruction error (MagNet-style). Strong perturbations push inputs
      away from the low-dimensional space where natural digits live.
    """

    def __init__(self, model, components: int = 30, sigma: float = 0.1, samples: int = 20, seed: int = 0):
        self.model, self.k, self.sigma, self.samples, self.seed = model, components, sigma, samples, seed
        self.mean = self.basis = None
        self.t_rec = self.t_inst = np.inf

    def fit(self, x_train: np.ndarray, x_calib: np.ndarray, fpr: float = 0.05) -> ManifoldDetector:
        self.mean = x_train.mean(axis=0)
        _, _, vt = np.linalg.svd(x_train - self.mean, full_matrices=False)
        self.basis = vt[: self.k]
        per_signal = 1 - (1 - fpr) ** 0.5            # split the false-positive budget between two tests
        self.t_rec = float(np.quantile(self.reconstruction_error(x_calib), 1 - per_signal))
        self.t_inst = float(np.quantile(self.instability(x_calib), 1 - per_signal))
        return self

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) @ self.basis.T
        return np.linalg.norm(x - (z @ self.basis + self.mean), axis=1)

    def instability(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        base = self.model.predict(x)
        flips = np.zeros(len(x))
        for _ in range(self.samples):
            flips += self.model.predict(np.clip(x + rng.normal(0, self.sigma, x.shape), 0, 1)) != base
        return flips / self.samples

    def is_adversarial(self, x: np.ndarray) -> np.ndarray:
        return (self.reconstruction_error(x) > self.t_rec) | (self.instability(x) > self.t_inst)


class SqueezeDetector:
    """Flags inputs whose prediction changes a lot when the input is 'squeezed' (bit-depth / median filter).

    Legitimate inputs are robust to squeezing; adversarial perturbations live in the fine detail that
    squeezing removes, so the model's output jumps. Threshold chosen on clean data for a target FPR.
    """

    def __init__(self, model, bits: int = 2, side: int = 8):
        self.model, self.bits, self.side = model, bits, side
        self.threshold = 1.0

    def score(self, x: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(x)
        d1 = np.abs(p - self.model.predict_proba(squeeze_bits(x, self.bits))).sum(axis=1)
        d2 = np.abs(p - self.model.predict_proba(median_smooth(x, self.side))).sum(axis=1)
        return np.maximum(d1, d2)

    def calibrate(self, x_clean: np.ndarray, fpr: float = 0.05) -> SqueezeDetector:
        self.threshold = float(np.quantile(self.score(x_clean), 1 - fpr))
        return self

    def is_adversarial(self, x: np.ndarray) -> np.ndarray:
        return self.score(x) > self.threshold
