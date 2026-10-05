"""White-box evasion attacks under an L-infinity budget, on inputs scaled to [0, 1]."""
from __future__ import annotations

import numpy as np


def fgsm(model, x: np.ndarray, y: np.ndarray, eps: float, target: np.ndarray | None = None) -> np.ndarray:
    """Fast Gradient Sign Method (Goodfellow et al., 2015). Targeted: step *down* the loss of the target class."""
    if target is not None:
        step = -np.sign(model.input_gradient(x, target))
    else:
        step = np.sign(model.input_gradient(x, y))
    return np.clip(x + eps * step, 0.0, 1.0)


def pgd(model, x: np.ndarray, y: np.ndarray, eps: float, alpha: float | None = None, steps: int = 20,
        random_start: bool = True, target: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Projected Gradient Descent (Madry et al., 2018): iterated FGSM projected back into the eps-ball."""
    alpha = alpha or 2.5 * eps / steps
    rng = np.random.default_rng(seed)
    adv = x + (rng.uniform(-eps, eps, x.shape) if random_start else 0.0)
    adv = np.clip(adv, 0.0, 1.0)
    for _ in range(steps):
        if target is not None:
            adv = adv - alpha * np.sign(model.input_gradient(adv, target))
        else:
            adv = adv + alpha * np.sign(model.input_gradient(adv, y))
        adv = np.clip(np.clip(adv, x - eps, x + eps), 0.0, 1.0)     # project to the ball, then the valid range
    return adv


def random_noise(model, x: np.ndarray, y: np.ndarray, eps: float, seed: int = 0) -> np.ndarray:
    """Baseline: random +-eps noise. Shows that adversarial perturbations are *not* just noise."""
    rng = np.random.default_rng(seed)
    return np.clip(x + eps * rng.choice([-1.0, 1.0], x.shape), 0.0, 1.0)


def attack_success_rate(model, x_clean: np.ndarray, x_adv: np.ndarray, y: np.ndarray) -> float:
    """Among examples the model got right, the fraction the attack flipped."""
    ok = model.predict(x_clean) == y
    if not ok.any():
        return 0.0
    return float((model.predict(x_adv[ok]) != y[ok]).mean())
