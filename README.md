# 🌌 Astrophysics — Dark Matter Estimation via Normalizing Flows & the Jeans Equation

> A machine learning pipeline for inferring dark matter distribution in dwarf galaxies from projected stellar kinematics, without assuming a parametric profile shape.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat-square&logo=tensorflow)
![Platform](https://img.shields.io/badge/Training-Puhti%20Supercomputer%20(CSC)-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow?style=flat-square)

---

## Overview

A long-standing open question in near-field cosmology is whether dark matter halos in dwarf galaxies have **cores** (flat central density) or **cusps** (rising central density). Cold dark matter simulations universally predict cusps, but observations frequently suggest cores — a tension that remains unresolved.

This project implements and evaluates **JFlow**, a method introduced by Lim et al. (2025), which uses a **Normalizing Flow neural network** as a data-driven density estimator for stellar kinematics. Combined with the **spherical Jeans equation**, the pipeline recovers the dark matter mass profile directly from 2D projected observations — no parametric assumptions required.

The pipeline was validated on mock datasets and then applied to a **real observational dataset of 468 stars** for the first time.

> Conducted at **Sendai National College of Technology**, Japan, under the supervision of Professor Kohei Hayashi. Training was performed on the **Puhti supercomputer at CSC Finland** using NVIDIA V100 GPUs.
<img width="457" height="234" alt="{81A4496F-2BE7-45B3-9516-5E56E0A5B44F}" src="https://github.com/user-attachments/assets/6e7d5572-6a0d-45b9-9bc5-bd8cba4bad44" />

---

## The Problem

Given only 2D projected stellar positions (X, Y) and line-of-sight velocities (V_Z), recover:

| Target | Symbol |
|---|---|
| 3D stellar number density | n(r) |
| Radial velocity dispersion | σ²_r(r) |
| Enclosed dark matter mass | M(r) |
| Dark matter mass density | ρ(r) |

---

## Dataset

| Property | Details |
|---|---|
| Mock data source | GaiaChallenge simulation suite |
| Mock dataset types | Core (γ = 0) and Cusp (γ = 1) DM profiles × 3 sizes each |
| Stars per mock dataset | ~1,000 simulated stars |
| Observables | Projected positions (X, Y) + line-of-sight velocity (V_Z) |
| Real dataset | 468 observed stars |

Dark matter halos follow a **generalized NFW profile**:

```
ρ_DM(r) = ρ₀ / [ (r/r_s)^γ · (1 + (r/r_s)^α)^((β−γ)/α) ]
```

- **Core** (γ = 0): flat central density
- **Cusp** (γ = 1): density rises toward centre

---

## Pipeline Architecture

```
Observed stars (X, Y, V_Z)
        │
        ▼
┌───────────────────────┐
│   Normalizing Flow    │  ← 12 coupling layers, 256 hidden units
│  (Neural Density      │     Learns full 3D phase-space f(X, Y, V_Z)
│   Estimator)          │     Trained by maximizing log-likelihood
└───────────┬───────────┘
            │ Sample 100,000 synthetic 3D stars
            ▼
┌───────────────────────┐
│  Abel Inversion       │  ← Projected Σ(R) → 3D density n(r)
│  + Spline Fitting     │     25 log-spaced projected bins
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Spherical Jeans Eq.  │  ← Infer enclosed DM mass M(r)
│                       │     Two anisotropy values: β=0, β=−0.5
└───────────┬───────────┘
            │
            ▼
    ρ(r), M(r), n(r), σ²_r(r)
    with calibrated uncertainty bands
```

### Model Architecture

| Component | Details |
|---|---|
| Architecture | 12 affine coupling layers |
| Hidden units | 256 per layer |
| Input dimensions | 3 (X, Y, V_Z) |
| Activations | tanh (scale clipped to [−5, 5]) |
| Optimizer | Adam with gradient clipping (max norm = 1.0) |
| Training runs | 20 independent runs per dataset (different random seeds) |
| Max epochs | 100 per run |
| LR schedule | Halved after 5 eval intervals with no improvement |

### Uncertainty Quantification

20 independent runs provide calibrated uncertainty bands computed in log₁₀ space:

| Band | Interval | Meaning |
|---|---|---|
| 🟢 Green | 68% (1σ) | [p16, p84] |
| 🟡 Yellow | 95% (2σ) | [p2.5, p97.5] |
| ⬜ Grey shading | — | Low-statistics zones: r < 0.1 kpc or r > 10.0 kpc |

Agreement with ground truth is quantified via a **pull plot**:
```
Pull = (log₁₀(estimate) − log₁₀(truth)) / σ_band
```
Pull ≈ 0 means the estimate matches truth within uncertainty. |Pull| > 2 indicates systematic bias.

---

## Results Mock data
## Core Profiles
<img width="975" height="333" alt="image" src="https://github.com/user-attachments/assets/52ce92a3-7dd2-4593-96f8-bc41154b3e06" />

## Cusp Profiles
<img width="975" height="333" alt="image" src="https://github.com/user-attachments/assets/da91e8dc-0c80-45dc-889c-5abf8b119ec4" />

Both core (γ = 0) and cusp (γ = 1) profiles are recovered within the reliable radial range of **0.1–10 kpc**:

- **n(r)** and **σ²_r(r)** recovered accurately, with the 1σ band consistently containing the true analytic curve
- **M(r)** recovered within 1–2σ across the reliable range
- **ρ(r)** shows wider uncertainty due to numerical differentiation, but remains consistent within 2σ

> ⚠️ A known artifact in cusp profiles: near r ≈ 0.3–1 kpc, spline differentiation of M(r) occasionally produces negative ρ values (set to NaN). This is a numerical limitation of the differentiation step, not a flow failure.

### Real Observational Data (468 stars)
<img width="1049" height="323" alt="image" src="https://github.com/user-attachments/assets/a44b19fb-38f0-46da-b568-c22d779ac0b8" />

| Finding | Detail |
|---|---|
| Stellar density | Consistent across all 6 trained models |
| Core vs Cusp separation | Visible in σ²_r and ρ(r) at r ~ 0.2–0.8 kpc |
| Enclosed mass estimate | ~10⁸–10⁸·⁵ M☉ at r ~ 1 kpc (robust across both families) |
| Core vs Cusp distinction | Inconclusive — 468 stars insufficient to definitively separate profiles at this precision |

---

## Tech Stack

| Tool | Role |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Normalizing flow implementation |
| SciPy `UnivariateSpline` | Spline fitting for Abel inversion |
| NumPy | Numerical integration & profile recovery |
| Puhti (CSC Finland) | GPU training via SLURM job array |
| NVIDIA V100 | ~10–15 min per training run |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/zignut1035/Dark-Matter-Galaxy.git
```
---

## Future Work

- **Larger observational samples** — More stars would provide the statistical power needed to conclusively distinguish core vs cusp profiles
- **Additional kinematic tracers** — Breaking the mass-anisotropy degeneracy inherent in the Jeans equation requires complementary data (e.g., proper motions)
- **Automated preprocessing** — Standardizing unit conversion and systemic velocity subtraction for arbitrary observational inputs

---

## Reference

> Lim, K.H. et al. (2025). *JFlow: Dark matter estimation via normalizing flows and the Jeans equation.* arXiv:2503.00763

> GaiaChallenge mock data: https://gaiachallenges.gitlab.io/

---

## Acknowledgements

This work was carried out during an internship at **Sendai National College of Technology**, Japan, under the supervision of **Professor Kohei Hayashi**. Computational resources were provided by **CSC Finland** (Puhti supercomputer).
