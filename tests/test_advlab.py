import numpy as np
import pytest

from advlab.attacks import attack_success_rate, fgsm, pgd, random_noise
from advlab.cli import main
from advlab.defenses import ManifoldDetector, SqueezeDetector, adversarially_trained, median_smooth, squeeze_bits
from advlab.experiments import load_data
from advlab.models import MLP


@pytest.fixture(scope="module")
def data():
    return load_data(0)


@pytest.fixture(scope="module")
def model(data):
    x_tr, _, y_tr, _ = data
    return MLP(64, 10, 64, 0).fit(x_tr, y_tr, epochs=20)


@pytest.mark.parametrize("hidden", [0, 16])
def test_input_gradient_matches_finite_differences(hidden):
    rng = np.random.default_rng(1)
    m = MLP(12, 4, hidden, seed=2)
    x, y = rng.uniform(0.1, 0.9, (3, 12)), np.array([0, 2, 3])
    g = m.input_gradient(x, y)
    h = 1e-5
    for i in range(3):
        for j in range(12):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += h
            xm[i, j] -= h
            num = (m.loss(xp[i:i + 1], y[i:i + 1]) - m.loss(xm[i:i + 1], y[i:i + 1])) / (2 * h)
            assert g[i, j] == pytest.approx(num, rel=1e-3, abs=1e-6)


def test_parameter_gradient_finite_difference():
    rng = np.random.default_rng(3)
    m = MLP(6, 3, 5, seed=4)
    x, y = rng.uniform(0, 1, (8, 6)), rng.integers(0, 3, 8)
    grads, _ = m.gradients(x, y)
    h = 1e-6
    w = m.W1
    old = w[2, 1]
    w[2, 1] = old + h
    lp = m.loss(x, y)
    w[2, 1] = old - h
    lm = m.loss(x, y)
    w[2, 1] = old
    assert grads["W1"][2, 1] == pytest.approx((lp - lm) / (2 * h), rel=1e-4)


def test_model_learns(model, data):
    _, x_te, _, y_te = data
    assert model.accuracy(x_te, y_te) > 0.95


def test_attacks_respect_budget_and_range(model, data):
    _, x, _, y = data
    for adv in (fgsm(model, x, y, 0.1), pgd(model, x, y, 0.1)):
        assert np.abs(adv - x).max() <= 0.1 + 1e-9
        assert adv.min() >= 0 and adv.max() <= 1


def test_adversarial_beats_random_noise_and_pgd_beats_fgsm(model, data):
    _, x, _, y = data
    acc_noise = model.accuracy(random_noise(model, x, y, 0.15), y)
    acc_fgsm = model.accuracy(fgsm(model, x, y, 0.15), y)
    acc_pgd = model.accuracy(pgd(model, x, y, 0.15), y)
    assert acc_noise > 0.85 and acc_fgsm < 0.4 and acc_pgd <= acc_fgsm


def test_targeted_attack(model, data):
    _, x, _, y = data
    mask = y != 0
    adv = pgd(model, x[mask], y[mask], 0.3, target=np.zeros(mask.sum(), int), steps=40)
    assert (model.predict(adv) == 0).mean() > 0.5


def test_attack_success_rate(model, data):
    _, x, _, y = data
    assert attack_success_rate(model, x, x, y) == 0.0
    assert attack_success_rate(model, x, pgd(model, x, y, 0.2), y) > 0.9


def test_adversarial_training_improves_robustness(model, data):
    x_tr, x_te, y_tr, y_te = data
    robust = adversarially_trained(x_tr, y_tr, eps=0.1, epochs=15)
    std_rob = model.accuracy(pgd(model, x_te, y_te, 0.1), y_te)
    adv_rob = robust.accuracy(pgd(robust, x_te, y_te, 0.1), y_te)
    assert adv_rob > std_rob + 0.2
    assert robust.accuracy(x_te, y_te) > 0.93            # small clean-accuracy cost


def test_squeezers():
    x = np.array([[0.0, 0.4, 0.6, 1.0]])
    assert np.allclose(squeeze_bits(x, 1), [[0, 0, 1, 1]])
    img = np.zeros((1, 64))
    img[0, 27] = 1.0                                       # single bright pixel
    assert median_smooth(img).max() == 0.0


def test_manifold_detector_outperforms_squeezing(model, data):
    x_tr, x_te, y_tr, y_te = data
    fit, cal = x_tr[:900], x_tr[900:]
    man = ManifoldDetector(model).fit(fit, cal, fpr=0.05)
    sq = SqueezeDetector(model).calibrate(cal, fpr=0.05)
    adv = pgd(model, x_te, y_te, 0.3)
    assert man.is_adversarial(adv).mean() > 0.9
    assert man.is_adversarial(adv).mean() > sq.is_adversarial(adv).mean() + 0.5
    assert man.is_adversarial(x_te).mean() < 0.2


def test_cli(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["--epochs", "8", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "PGD-20" in text and "manifold" in text and out.exists()
