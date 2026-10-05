"""Reproducible experiments on the 8x8 handwritten digits dataset (scikit-learn, 1,797 images)."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .attacks import attack_success_rate, fgsm, pgd, random_noise
from .defenses import ManifoldDetector, SqueezeDetector, adversarially_trained
from .models import MLP

EPSILONS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


def load_data(seed: int = 0):
    d = load_digits()
    x = d.data / 16.0                                        # pixels to [0, 1]
    return train_test_split(x, d.target, test_size=0.3, stratify=d.target, random_state=seed)


def robustness_curve(model, x, y, epsilons=EPSILONS) -> dict[str, list[float]]:
    out = {"eps": list(epsilons), "noise": [], "fgsm": [], "pgd": []}
    for e in epsilons:
        out["noise"].append(model.accuracy(random_noise(model, x, y, e), y) if e else model.accuracy(x, y))
        out["fgsm"].append(model.accuracy(fgsm(model, x, y, e), y) if e else model.accuracy(x, y))
        out["pgd"].append(model.accuracy(pgd(model, x, y, e), y) if e else model.accuracy(x, y))
    return out


def run_all(seed: int = 0, train_eps: float = 0.1, epochs: int = 30) -> dict:
    x_tr, x_te, y_tr, y_te = load_data(seed)
    standard = MLP(64, 10, 64, seed).fit(x_tr, y_tr, epochs=epochs, seed=seed)
    robust = adversarially_trained(x_tr, y_tr, eps=train_eps, epochs=epochs, seed=seed)

    # targeted attack: make every test digit read as "0"
    target = np.zeros_like(y_te)
    mask = y_te != 0
    adv_t = pgd(standard, x_te[mask], y_te[mask], eps=0.2, target=target[mask], steps=40)
    targeted_rate = float((standard.predict(adv_t) == 0).mean())

    # detection: calibrate on a held-out slice of the training data (never on the test set)
    x_fit, x_cal = x_tr[: int(0.75 * len(x_tr))], x_tr[int(0.75 * len(x_tr)):]
    squeeze = SqueezeDetector(standard).calibrate(x_cal, fpr=0.05)
    manifold = ManifoldDetector(standard).fit(x_fit, x_cal, fpr=0.05)
    detection = {"eps": [], "manifold_tpr": [], "squeeze_tpr": []}
    for e in (0.05, 0.1, 0.15, 0.2, 0.3):
        a = pgd(standard, x_te, y_te, eps=e)
        detection["eps"].append(e)
        detection["manifold_tpr"].append(float(manifold.is_adversarial(a).mean()))
        detection["squeeze_tpr"].append(float(squeeze.is_adversarial(a).mean()))
    detection["manifold_fpr"] = float(manifold.is_adversarial(x_te).mean())
    detection["squeeze_fpr"] = float(squeeze.is_adversarial(x_te).mean())
    adv = pgd(standard, x_te, y_te, eps=train_eps)
    return {
        "detection": detection,
        "standard": robustness_curve(standard, x_te, y_te),
        "adversarial_training": robustness_curve(robust, x_te, y_te),
        "pgd_success_rate_standard": attack_success_rate(standard, x_te, adv, y_te),
        "targeted_to_zero_rate": targeted_rate,
        "train_eps": train_eps,
    }
