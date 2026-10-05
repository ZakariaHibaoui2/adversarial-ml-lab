# Adversarial ML Lab — Attacks & Defenses from Scratch

> How to fool a neural network, and how to fight back, implemented in **pure NumPy** so every gradient is visible. The lab includes gradient-checked models, **FGSM** and **PGD** attacks (untargeted and targeted), **adversarial training**, and two detectors compared honestly, all reproducible in about 10 seconds on a laptop CPU.

[![CI](https://github.com/ZakariaHibaoui2/adversarial-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ZakariaHibaoui2/adversarial-ml-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy&logoColor=white)
![AI security](https://img.shields.io/badge/AI%20security-adversarial%20ML-purple)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Results (`advlab`, seed 0)

Dataset: scikit-learn **8×8 handwritten digits** (1,797 images, 70/30 split). Model: MLP 64→64→10, Adam, 30 epochs. Attacks use an L∞ budget ε on pixels in [0, 1].

**Test accuracy under attack**

| ε | Standard: random noise | Standard: FGSM | Standard: PGD-20 | Adv. trained (ε=0.1): FGSM | Adv. trained: PGD-20 |
|---|---|---|---|---|---|
| 0.00 | 98.0% | 98.0% | 98.0% | 97.2% | 97.2% |
| 0.05 | 96.9% | 82.4% | 81.5% | 92.6% | 92.2% |
| 0.10 | 96.3% | 40.6% | **35.0%** | 78.7% | **76.9%** |
| 0.15 | 94.3% | 12.0% | 8.0% | 53.7% | 46.5% |
| 0.20 | 91.3% | 0.9% | 0.7% | 27.6% | 17.6% |
| 0.30 | 76.3% | 0.0% | 0.0% | 1.7% | 0.2% |

- **Adversarial ≠ random.** Random ±0.1 noise costs 1.7 points, while a *crafted* ±0.1 perturbation costs 63.
- **PGD ≥ FGSM** at every budget: iterating and projecting finds stronger examples than a single step.
- **Adversarial training** (half of each batch replaced by PGD-7 examples) more than doubles robust accuracy at ε=0.1 (35% → 77%), for 0.8 points of clean accuracy: the classic robustness/accuracy trade-off. Robustness doesn't transfer to budgets far above the training ε.
- **Targeted PGD** (ε=0.2) turns 58.6% of non-zero digits into a "0" chosen by the attacker.

**Detecting adversarial inputs** (true-positive rate on PGD examples; thresholds calibrated on held-out clean training images)

| ε | Manifold detector | Feature squeezing |
|---|---|---|
| 0.05 | 31.1% | 4.4% |
| 0.10 | 59.6% | 6.1% |
| 0.15 | 63.5% | 8.0% |
| 0.20 | 91.1% | 6.3% |
| 0.30 | **100%** | 5.9% |
| FPR on clean test images | 10.4% | 2.6% |

- **Feature squeezing** (bit-depth reduction + median filter, Xu et al. 2018) **does not work here**: detection stays at chance level. These images have only 17 grey levels and 8×8 pixels, so there is little "fine detail" for squeezing to remove. It is kept as a documented negative result.
- The **manifold detector** ORs two complementary signals:
  - *instability* under random noise, which catches weak attacks sitting just across a decision boundary;
  - *PCA reconstruction error* (MagNet-style), which catches strong attacks pushed off the natural-digit manifold.
- Its FPR (10.4%) exceeds the 5% target because the calibration set is small (≈ 300 images). A larger calibration set tightens it.
- Neither detector has been evaluated against an **adaptive attacker** who knows the defence. That is the standard caveat (Carlini & Wagner, 2017): detection results like these are upper bounds.

## What's implemented

| Module | Contents |
|---|---|
| `models.py` | MLP / softmax regression with analytic gradients for parameters **and inputs**; Adam; optional adversary hook for adversarial training |
| `attacks.py` | FGSM, PGD (random start, projection to the ε-ball, targeted mode), random-noise baseline, attack success rate |
| `defenses.py` | Madry-style adversarial training, `ManifoldDetector` (instability + PCA), `SqueezeDetector` (bit depth + median) |
| `experiments.py` | Robustness curves, targeted attack, detection study |

Correctness is checked, not assumed: tests compare the hand-written input and parameter gradients against **finite differences**, and verify that perturbations stay inside the ε-ball and the valid pixel range.

## Usage

```bash
pip install -e ".[dev]"
advlab                         # full experiment suite (~10 s)
advlab --train-eps 0.15 --json results.json
pytest -q                      # 12 tests incl. gradient checks
```

```python
from advlab.experiments import load_data
from advlab.models import MLP
from advlab.attacks import pgd

x_tr, x_te, y_tr, y_te = load_data()
model = MLP(64, 10).fit(x_tr, y_tr)
x_adv = pgd(model, x_te, y_te, eps=0.1)
print(model.accuracy(x_te, y_te), model.accuracy(x_adv, y_te))
```

## References

Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2015) · Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018) · Xu et al., *Feature Squeezing* (2018) · Meng & Chen, *MagNet* (2017) · Carlini & Wagner, *Adversarial Examples Are Not Easily Detected* (2017)

## License

[MIT](LICENSE)
