# Architecture optimum for the erf-combo committee

This note explains what **“arch optimum”** (architecture optimum) means in [`erf_combo_commette_machine.py`](erf_combo_commette_machine.py), how it is computed, and how to interpret it relative to the teacher and to gradient-descent training.

---

## Setup

**Input:** \(x \in \mathbb{R}^d\), currently \(d = 1\).

**Teacher (supervisor target):**

\[
y(x) = \mathrm{erf}(x_1) - 2\,\mathrm{erf}\!\left(\frac{x_1}{2}\right)
\]

**Student (committee network):** same class as `CommitteeStudent` in [`simple_commette_machine.py`](simple_commette_machine.py):

\[
f(x;\,W) = \frac{1}{\sqrt{P}} \sum_{p=1}^{P} \mathrm{erf}(w_p \cdot x)
\]

where \(P =\) `N_HIDDEN`, and each row \(w_p \in \mathbb{R}^d\) is a hidden weight vector. In \(d=1\), each \(w_p\) is a scalar pre-activation scale inside the erf.

**Training data distribution:** \(x \sim \mathcal{N}(0, I)\) (standard normal on each coordinate).

**Loss:** population (or empirical) mean squared error,

\[
\mathcal{L}(W) = \mathbb{E}_x\big[(f(x; W) - y(x))^2\big].
\]

---

## Three different “targets” (do not confuse them)

When we ask whether training “converged to the theoretical result,” there are **three** distinct reference points.

| Name | What it is | Typical MSE (P=10, d=1) |
|------|------------|-------------------------|
| **1. Teacher** | The function \(y(x)\) itself | 0 by definition |
| **2. Exact readout reference** | Same formula as teacher, but interpreted as *two erf neurons with free output weights* \(+1\) and \(-2\) | 0 exactly |
| **3. Arch optimum** | Best fit **within the equal-readout student class** | \(\approx 10^{-5}\)–\(10^{-6}\) (numerical) |
| **4. Trained student** | Weights found by Adam on fixed offline data from init | Often \(\gg\) arch optimum (e.g. \(\sim 0.02\)–\(0.06\) in our runs) |

The **arch optimum** answers:

> *Given that every hidden unit must contribute with the same coefficient \(1/\sqrt{P}\), what is the smallest MSE this architecture can achieve on Gaussian inputs?*

It does **not** assume the student found that solution during training.

---

## Why the teacher and the student are different classes

The teacher is a **weighted** sum of two erfs:

\[
y(x) = \underbrace{1}_{\text{coeff}} \cdot \mathrm{erf}(1 \cdot x) + \underbrace{(-2)}_{\text{coeff}} \cdot \mathrm{erf}\!\left(\tfrac{1}{2} x\right).
\]

The student is an **equal-weight** sum:

\[
f(x) = \frac{1}{\sqrt{P}} \sum_{p=1}^{P} \mathrm{erf}(w_p x).
\]

Every neuron has the same readout \(1/\sqrt{P}\). You cannot set one coefficient to \(+1\) and another to \(-2\) unless \(P=1\) and you absorb constants into \(w\) (which you cannot, because erf is nonlinear in its argument scale).

So:

- **Exact readout** (free coefficients): MSE \(= 0\) with two terms.
- **Equal-readout committee**: must approximate the teacher by choosing **many** \(w_p\) and letting them interfere. With enough flexibility in \(\{w_p\}\), MSE can still become very small—but not via the same two-term closed form.

The arch optimum quantifies how small “very small” can get **for your actual architecture**.

---

## Formal definition of the architecture optimum

Fix width \(P\) and dimension \(d\). The **architecture optimum** is (conceptually):

\[
W^\star \in \arg\min_{W \in \mathbb{R}^{P \times d}} \; \mathbb{E}_{x \sim \mathcal{N}(0,I)}\!\left[\left(\frac{1}{\sqrt{P}} \sum_{p=1}^{P} \mathrm{erf}(w_p \cdot x) - y(x)\right)^2\right].
\]

The corresponding **architecture-optimum function** is:

\[
f_{\mathrm{arch}}(x) = \frac{1}{\sqrt{P}} \sum_{p=1}^{P} \mathrm{erf}(w_p^\star \cdot x).
\]

The **architecture-optimum MSE** is \(\mathcal{L}(W^\star)\).

Important properties:

1. **Same hypothesis class as the student** (equal readout, erf activation).
2. **Optimization is over \(W\) only**—no separate training dynamics, no cached init, no unit-norm constraint at optimum (see below).
3. **Population loss on Gaussian \(x\)**—matches the data-generating process used in training.

This is a **numerical** object: we do not have a closed-form formula for \(W^\star\). We approximate it by direct optimization.

---

## How the code computes it

Implemented in `fit_theoretical_equal_weight_committee()`:

```python
def fit_theoretical_equal_weight_committee(n_hidden, n_samples=10_000, steps=5000, lr=1e-2, seed=0):
    x = torch.randn(n_samples, DIMENSION)   # Monte Carlo samples for E_x[·]
    y = teacher_erf_combo(x)

    w = nn.Parameter(torch.randn(n_hidden, DIMENSION))  # random init
    optimizer = optim.Adam([w], lr=lr)

    for _ in range(steps):
        pred = (1/sqrt(P)) * erf(x @ w.T).sum(-1)
        loss = ((pred - y)**2).mean()
        loss.backward()
        optimizer.step()

    return final_mse, w.detach()
```

Steps in plain language:

1. Draw \(10{,}000\) points \(x \sim \mathcal{N}(0,1)\).
2. Initialize all hidden scales \(w_p\) randomly.
3. Run **5,000 steps of Adam** (lr \(= 10^{-2}\)) to minimize empirical MSE.
4. Return the final MSE and weights \(w_p\).

That pair \((\mathcal{L}, W)\) is what we call **arch optimum** in plots and logs:

- Green dashed curve in `theory_curves.png` → \(f_{\mathrm{arch}}(x)\) on a grid.
- `arch_opt_population_mse` → MSE on the Gaussian Monte Carlo batch used for fitting.
- `arch_opt_grid_mse` → MSE on a uniform grid \(x \in [-3,3]\) (sanity check).
- “arch optimum” series in `weights_distribution.png` → histogram of fitted \(w_p\).

---

## What the arch optimum is **not**

| Not this | Why |
|----------|-----|
| **An analytic theorem** | We do not prove the infimum is \(10^{-5}\); we **estimate** it by optimization. |
| **Guaranteed global minimum** | The problem is non-convex; Adam from one random init may miss a better solution. |
| **The same as training** | Training uses fixed offline data, `CommitteeStudent` init (unit-norm rows at \(t=0\)), lr \(10^{-4}\), 10k epochs on 10k points. The fitter uses lr \(10^{-2}\), no unit-norm constraint during fit, and optimizes population MSE directly on fresh Gaussian samples. |
| **The teacher** | Teacher allows coefficients \(+1, -2\); arch optimum respects \(1/\sqrt{P}\) on every unit. |
| **NNGP / kernel regression** | NNGP uses the infinite-width kernel at initialization; it is unrelated to this finite-\(P\) equal-weight class (and is degenerate at \(d=1\)). |

---

## Typical numerical values (P = 10, d = 1)

Re-running the fitter gives stable results on the order of:

| Fit steps | `arch_opt_population_mse` |
|-----------|---------------------------|
| 2,000 | \(\approx 1.2 \times 10^{-5}\) |
| 5,000 | \(\approx 7.7 \times 10^{-6}\) |
| 10,000 | \(\approx 7.0 \times 10^{-6}\) |

So for **10 hidden units**, the equal-readout architecture can represent the erf-combo teacher to **~\(10^{-5}\) MSE** on Gaussian inputs—essentially a perfect fit at the population level.

By contrast, a **trained student** in the same script (same \(P=10\), Adam from init) may sit at **test MSE \(\sim 0.05\)** while the curve on \([-3,3]\) still looks wrong. That gap is reported as:

\[
\texttt{trained\_minus\_arch\_opt} = \text{trained\_test\_mse} - \text{arch\_opt\_population\_mse}.
\]

Large positive values mean: *the architecture **could** fit the teacher, but this training run **did not** find those weights.*

---

## What the fitted weights look like

The exact teacher wants two logical components:

- \(\mathrm{erf}(x)\) → scale \(w \approx 1\), coefficient \(+1\)
- \(\mathrm{erf}(x/2)\) → scale \(w \approx 0.5\), coefficient \(-2\)

The arch optimum **cannot** assign \(+1\) and \(-2\) to individual neurons. Instead, with \(P=10\) equal coefficients, the fitter typically spreads mass across several \(w_p\) values clustered near **\(\pm 1\)** and **\(\pm 0.5\)** (see dotted lines in the weight histogram). Many units cooperate to emulate the \(-2\) weight on the \(\mathrm{erf}(x/2)\) term.

The **trained** student’s weights are often **not** clustered that way—another sign it has not reached the arch-optimum configuration.

---

## Why the teacher has no linear term (context)

Near \(x=0\):

\[
\mathrm{erf}(x) - 2\,\mathrm{erf}(x/2) = -\frac{1}{2\sqrt{\pi}}\,x^3 + O(x^5).
\]

The linear pieces cancel: \(y'(0)=0\). So the network must learn **nonlinear** erf structure, not a slope near the origin. The arch optimum shows this **is possible** in the equal-weight class; training dynamics are the bottleneck.

---

## How to use this in interpretation

1. **Teacher curve (blue)** — ground-truth target \(y(x)\).
2. **Arch optimum (green dashed)** — best equal-weight approximation at fixed \(P\); if this matches the teacher, the **architecture is expressive enough**.
3. **Trained student (orange)** — what your optimizer actually learned.
4. If orange \(\neq\) green but green \(\approx\) blue → **optimization / init problem**, not wrong target or wrong model class.
5. Compare **`trained_minus_arch_opt`**, not just raw test MSE vs zero predictor.

---

## Reproducing the arch optimum

From the repo root:

```bash
python -c "
from erf_combo_commette_machine import fit_theoretical_equal_weight_committee, N_HIDDEN
mse, w = fit_theoretical_equal_weight_committee(N_HIDDEN)
print('arch_opt_population_mse', mse)
print('w_p', w.squeeze().tolist())
"
```

Or run full training; `analyze_convergence_to_theory()` calls the fitter automatically at the end and writes plots to `simple-committee-machine-erf-combo/`.

---

## Summary

The **architecture optimum** is the best-fit equal-readout erf committee at width \(P\), found by directly minimizing population MSE against the erf-combo teacher. It is the correct theoretical benchmark **for your network class**: it separates “can this architecture represent the teacher?” (yes, to ~\(10^{-5}\) for \(P=10\)) from “did Adam find that representation?” (often no, in current runs).
