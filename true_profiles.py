"""
true_profiles.py
================
Generates 3-panel plots for each dataset comparing:
  Panel 1 – Stellar number density     : NFlow v2 estimate  vs  6D true data
  Panel 2 – Radial velocity dispersion : NFlow v2 estimate  vs  true Jeans (analytical DM)
  Panel 3 – Enclosed DM mass           : NFlow v2 estimate  vs  analytical M(r)

Usage:
    python true_profiles.py
    (run from the same folder as your observables_*.dat files and *_v2.weights.* files)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
import glob


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DM profile parameters (from professor)
#     rho(r) = rho_DM * (r/rs)^(-gamma) * (1 + (r/rs)^alpha)^((gamma-beta)/alpha)
#     NOTE: beta here is the OUTER SLOPE of the DM profile (=3),
#           NOT the velocity anisotropy.
# ══════════════════════════════════════════════════════════════════════════════
DM_PARAMS = {
    'core':   dict(rho_DM=0.064e9,  rs_DM=1.0,  alpha=1.0, beta_dm=3.0, gamma=0.0),
    'core02': dict(rho_DM=0.6e9,    rs_DM=0.45, alpha=1.0, beta_dm=3.0, gamma=0.0),
    'core03': dict(rho_DM=1.5e9,    rs_DM=0.27, alpha=1.0, beta_dm=3.0, gamma=0.0),
    'cusp':   dict(rho_DM=0.064e9,  rs_DM=1.0,  alpha=1.0, beta_dm=3.0, gamma=1.0),
    'cusp02': dict(rho_DM=0.002e9,  rs_DM=5.0,  alpha=1.0, beta_dm=3.0, gamma=1.0),
    'cusp03': dict(rho_DM=0.0004e9, rs_DM=13.0, alpha=1.0, beta_dm=3.0, gamma=1.0),
}

G_CONST = 4.30091e-6   # (km/s)² kpc / M☉
N_SAMPLE = 300_000


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Analytical DM profile functions
# ══════════════════════════════════════════════════════════════════════════════
def dm_density(r, rho_DM, rs_DM, alpha, beta_dm, gamma):
    """Generalised Hernquist-Zhao (core or cusp) density."""
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    out = np.zeros_like(r)
    m = r > 0
    x = r[m] / rs_DM
    out[m] = rho_DM * x**(-gamma) * (1.0 + x**alpha)**((gamma - beta_dm) / alpha)
    return float(out[0]) if scalar else out


def enclosed_mass(r, params, r_min=1e-4):
    """M(<r) = integral_r_min^r  4 pi r'^2 rho(r') dr'"""
    if r <= r_min:
        return 0.0
    val, _ = quad(
        lambda rp: 4.0 * np.pi * rp**2 * dm_density(rp, **params),
        r_min, r, limit=300, epsrel=1e-5)
    return max(val, 0.0)


def true_jeans_sigma(r_grid, n_spl, params, r_outer):
    """
    Isotropic spherical Jeans equation (beta_aniso = 0):
        sigma_r^2(r) = (1/n(r)) * integral_r^r_outer  n(r') * G*M(r') / r'^2  dr'
    n_spl  : log-log UnivariateSpline of stellar 3D number density n(r)
    params : DM profile parameter dict
    """
    # Pre-compute M(r) on a fine grid to speed up the inner integral
    r_M    = np.logspace(np.log10(r_grid[0] * 0.5),
                         np.log10(r_outer * 1.1), 80)
    M_vals = np.array([enclosed_mass(r, params) for r in r_M])
    M_spl  = UnivariateSpline(
        np.log(r_M), np.log(np.clip(M_vals, 1e-30, None)),
        s=0, k=3, ext=3)

    sigma2 = []
    for r in r_grid:
        def integrand(rp):
            nr = float(np.exp(n_spl(np.log(rp))))
            Mr = float(np.exp(M_spl(np.log(rp))))
            return nr * G_CONST * Mr / rp**2

        try:
            val, _ = quad(integrand, r, r_outer, limit=200, epsrel=1e-4)
            nr = float(np.exp(n_spl(np.log(r))))
            sigma2.append(val / nr if nr > 1e-30 else np.nan)
        except Exception:
            sigma2.append(np.nan)

    return np.array(sigma2)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NFlow model  (v2 architecture — must match training)
# ══════════════════════════════════════════════════════════════════════════════
class CouplingLayer(keras.layers.Layer):
    def __init__(self, hidden=256, **kwargs):
        super().__init__(**kwargs)

        def _net(out_dim):
            return keras.Sequential([
                keras.layers.Dense(hidden, activation='swish'),
                keras.layers.Dense(hidden, activation='swish'),
                keras.layers.Dense(hidden, activation='swish'),
                keras.layers.Dense(out_dim,
                                   kernel_initializer='zeros',
                                   bias_initializer='zeros'),
            ])

        self.net_s = _net(2)
        self.net_t = _net(2)

    def call(self, x, reverse=False):
        x0, x1 = x[:, :1], x[:, 1:]
        s = tf.clip_by_value(self.net_s(x0), -5.0, 5.0)
        t = self.net_t(x0)
        if not reverse:
            return tf.concat([x0, x1 * tf.exp(s) + t], axis=1), \
                   tf.reduce_sum(s, axis=1)
        else:
            return tf.concat([x0, (x1 - t) * tf.exp(-s)], axis=1)


class NFlow(keras.Model):
    def __init__(self, n_layers=12, hidden=256, **kwargs):
        super().__init__(**kwargs)
        self.coupling_layers = [CouplingLayer(hidden) for _ in range(n_layers)]
        self.perms     = [tf.constant([1, 2, 0] if i % 2 == 0 else [2, 0, 1])
                          for i in range(n_layers)]
        self.inv_perms = [tf.argsort(p) for p in self.perms]

    def log_prob(self, x):
        log_det = tf.zeros(tf.shape(x)[0])
        for layer, perm in zip(self.coupling_layers, self.perms):
            x = tf.gather(x, perm, axis=1)
            x, ld = layer(x, reverse=False)
            log_det += ld
        log_base = -0.5 * tf.reduce_sum(
            x**2 + tf.cast(tf.math.log(2.0 * np.pi), tf.float32), axis=1)
        return log_base + log_det

    def sample(self, n):
        z = tf.random.normal((n, 3))
        for layer, inv_perm in zip(reversed(self.coupling_layers),
                                   reversed(self.inv_perms)):
            z = layer(z, reverse=True)
            z = tf.gather(z, inv_perm, axis=1)
        return z.numpy()

    def call(self, x):
        return self.log_prob(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Profile computation from NFlow samples  (same as v2 training script)
# ══════════════════════════════════════════════════════════════════════════════
MIN_STARS = 50

def compute_nflow_profiles(samples, mean, std, n_bins=50):
    """
    Returns r_grid, n_r, n_spl, sigma_r2_b0, sigma_r2_bm5, M_nf
    n_spl  : log-log spline of estimated stellar n(r) (for Jeans comparison)
    M_nf   : enclosed mass estimated from Jeans mass estimator
    """
    s    = samples * std + mean
    R_s  = np.sqrt(s[:, 0]**2 + s[:, 1]**2)
    VZ_s = s[:, 2]

    lo, hi = np.percentile(R_s, 1.0), np.percentile(R_s, 99.0)
    mask   = (R_s > lo) & (R_s < hi)
    R_s, VZ_s = R_s[mask], VZ_s[mask]

    R_bins = np.logspace(np.log10(R_s.min() * 1.01),
                         np.log10(R_s.max() * 0.99), n_bins + 1)
    R_mid  = np.sqrt(R_bins[:-1] * R_bins[1:])

    counts, _ = np.histogram(R_s, bins=R_bins)
    area       = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)
    Sigma      = counts / area

    sigma_los2 = np.array([
        VZ_s[(R_s >= R_bins[k]) & (R_s < R_bins[k+1])].var()
        for k in range(n_bins)
    ])

    good = (counts >= MIN_STARS) & (sigma_los2 > 0) & (Sigma > 0)
    if good.sum() < 10:
        raise ValueError(f"Only {good.sum()} good bins — try increasing N_SAMPLE.")

    Rm     = R_mid[good]
    Sig    = Sigma[good]
    sl2    = sigma_los2[good]
    n_good = good.sum()
    sv     = float(n_good) * 4.0
    sl     = float(n_good) * 0.8

    # Abel inversion: Sigma(R) → n(r)
    log_Sig_spl = UnivariateSpline(np.log(Rm), np.log(Sig), s=sv, k=3)

    def dSigma_dR(R):
        lnR = np.log(R)
        return (np.exp(log_Sig_spl(lnR)) / R
                * float(log_Sig_spl.derivative()(lnR)))

    R_max = Rm[-1]

    def abel_n(r):
        if r >= R_max:
            return 1e-30
        val, _ = quad(lambda R: dSigma_dR(R) / np.sqrt(R**2 - r**2),
                      r * 1.001, R_max, limit=150)
        return max(-val / np.pi, 1e-30)

    skip    = max(3, int(0.15 * n_good))
    r_inner = Rm[skip]
    r_outer = Rm[-skip - 1]
    if r_inner >= r_outer:
        r_inner, r_outer = Rm[2], Rm[-3]

    r_grid = np.logspace(np.log10(r_inner), np.log10(r_outer), 35)
    n_r    = np.clip(np.array([abel_n(r) for r in r_grid]), 1e-30, None)
    n_spl  = UnivariateSpline(np.log(r_grid), np.log(n_r), s=sl, k=3)

    # Jeans inversion: Sigma*sigma_los^2 → sigma_r^2(r)
    SS         = Sig * sl2
    good2      = SS > 0
    log_SS_spl = UnivariateSpline(
        np.log(Rm[good2]), np.log(SS[good2]), s=sv, k=3)

    def dSS_dR(R):
        lnR = np.log(R)
        return (np.exp(log_SS_spl(lnR)) / R
                * float(log_SS_spl.derivative()(lnR)))

    def jeans_sr2(r, beta=0.0):
        if r >= R_max:
            return 1e-30
        try:
            val, _ = quad(
                lambda R: dSS_dR(R) * (1.0 - beta * (r/R)**2)
                          / np.sqrt(R**2 - r**2),
                r * 1.001, R_max, limit=150)
        except Exception:
            return 1e-30
        nr = float(np.exp(n_spl(np.log(r))))
        return max(-val / (np.pi * nr), 1e-30) if nr > 1e-30 else 1e-30

    sigma_r2_b0  = np.array([jeans_sr2(r, 0.0)  for r in r_grid])
    sigma_r2_bm5 = np.array([jeans_sr2(r, -0.5) for r in r_grid])

    # Jeans mass estimator M(r) from NFlow
    dln_n_dln_r  = n_spl.derivative()(np.log(r_grid))
    log_s2_spl   = UnivariateSpline(
        np.log(r_grid), np.log(np.clip(sigma_r2_b0, 1e-30, None)),
        s=sl, k=3)
    dln_s2_dln_r = log_s2_spl.derivative()(np.log(r_grid))
    M_nf = -(r_grid * sigma_r2_b0 / G_CONST) * (dln_n_dln_r + dln_s2_dln_r)

    return r_grid, n_r, n_spl, sigma_r2_b0, sigma_r2_bm5, M_nf


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Ground-truth loader (6D data)
# ══════════════════════════════════════════════════════════════════════════════
def load_ground_truth(input_filename, n_obs):
    gt_filename = (input_filename
                   .replace('3D_data', '6D_data')
                   .replace('observables_', 'Mock_isotropic_'))
    try:
        df = pd.read_csv(gt_filename, sep=r'\s+', comment='#',
                         names=['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
    except FileNotFoundError:
        print(f"  [Warning] GT not found: {gt_filename}")
        return None

    r_6d  = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    vr_6d = ((df['X']*df['VX'] + df['Y']*df['VY']
               + df['Z']*df['VZ']) / r_6d)

    gt_bins  = np.logspace(np.log10(r_6d.min()*1.01),
                           np.log10(r_6d.max()*0.99), 40)
    gt_r_mid = np.sqrt(gt_bins[:-1] * gt_bins[1:])

    counts, _ = np.histogram(r_6d, bins=gt_bins)
    vol        = (4/3) * np.pi * (gt_bins[1:]**3 - gt_bins[:-1]**3)
    n_true     = counts / vol

    # ── Fix: Scale True 6D density down to match observable counts ──
    n_true *= (n_obs / len(df))

    sigma2_true = np.array([
        vr_6d[(r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])].var()
        if ((r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])).sum() > 1
        else np.nan
        for k in range(len(gt_bins) - 1)
    ])

    # Spline of 6D stellar n(r) — used for true Jeans calculation
    good = (n_true > 0) & np.isfinite(n_true)
    n_spl_6d = UnivariateSpline(
        np.log(gt_r_mid[good]), np.log(n_true[good]),
        s=float(good.sum()) * 0.5, k=3, ext=3)

    return gt_r_mid, n_true, sigma2_true, n_spl_6d, r_6d.min(), r_6d.max()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Helper: match filename → DM profile key
# ══════════════════════════════════════════════════════════════════════════════
def get_profile_key(filename):
    bn = os.path.basename(filename).lower()
    # Check longer keys first to avoid 'core' matching 'core02'
    for key in ['core02', 'core03', 'cusp02', 'cusp03', 'cusp', 'core']:
        if key in bn:
            return key
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Plotting — 3 panels
# ══════════════════════════════════════════════════════════════════════════════
def make_plot(base_name, profile_key,
              r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_nf,
              r_true, M_true,
              r_jeans, sigma_true_jeans,
              gt):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Stellar number density ──────────────────────────────────────
    if gt is not None:
        gt_r_mid, n_true, *_ = gt
        axes[0].loglog(gt_r_mid, n_true, 'k-', lw=2, alpha=0.7,
                       label='True 6D Physics')
    axes[0].loglog(r_grid, n_r, 'g--', lw=2, label='Estimated, NFlow v2')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel('n(r) [kpc⁻³]')
    axes[0].set_title('Stellar number density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: Velocity dispersion ─────────────────────────────────────────
    if gt is not None:
        gt_r_mid, _, sigma2_true, *_ = gt
        valid = sigma2_true > 0
        axes[1].semilogx(gt_r_mid[valid], np.log(sigma2_true[valid]),
                         'k-', lw=2, alpha=0.7, label='True 6D Physics')

    # True Jeans with analytical DM profile
    valid_j = np.isfinite(sigma_true_jeans) & (sigma_true_jeans > 0)
    if valid_j.any():
        axes[1].semilogx(r_jeans[valid_j], np.log(sigma_true_jeans[valid_j]),
                         'r-', lw=2, alpha=0.9,
                         label=f'True Jeans ({profile_key})')

    v0 = sigma_r2_b0  > 1e-20
    v5 = sigma_r2_bm5 > 1e-20
    if v0.any():
        axes[1].semilogx(r_grid[v0], np.log(sigma_r2_b0[v0]),
                         'g--', lw=2, label=r'NFlow v2 $\beta=0$')
    if v5.any():
        axes[1].semilogx(r_grid[v5], np.log(sigma_r2_bm5[v5]),
                         'b:', lw=2, label=r'NFlow v2 $\beta=-0.5$')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'log $\sigma_r^2$ [(km/s)²]')
    axes[1].set_title('Radial velocity dispersion')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Enclosed DM mass ─────────────────────────────────────────────
    axes[2].loglog(r_true, M_true, 'r-', lw=2, alpha=0.9,
                   label=f'True M(r) ({profile_key})')
    pos = M_nf > 0
    if pos.any():
        axes[2].loglog(r_grid[pos], M_nf[pos], 'g--', lw=2,
                       label=r'NFlow v2 $M(r)$ ($\beta=0$)')
    axes[2].set_xlabel('r [kpc]')
    axes[2].set_ylabel(r'M(r) [$M_\odot$]')
    axes[2].set_title('Dark Matter Enclosed Mass')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f'NFlow v2 vs True DM profile  [{profile_key}]\n({base_name})',
        fontsize=11)
    plt.tight_layout()
    out_png = f'{base_name}_true_profiles.png'
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Saved → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
DATA_FILES = sorted(glob.glob('3D_data/observables_*.dat'))
if not DATA_FILES:
    raise FileNotFoundError("No data files found in 3D_data/")

print(f"Found {len(DATA_FILES)} data files.")

for input_filename in DATA_FILES:
    base_name   = os.path.splitext(os.path.basename(input_filename))[0]
    profile_key = get_profile_key(input_filename)

    print(f"\n{'='*60}")
    print(f"File       : {input_filename}")
    print(f"DM profile : {profile_key}")
    print(f"{'='*60}")

    if profile_key is None or profile_key not in DM_PARAMS:
        print("  [Skip] Could not match a DM profile key from filename.")
        continue

    params = DM_PARAMS[profile_key]

    # ── Load 3D observables (needed for normalisation) ────────────────────────
    df       = pd.read_csv(input_filename, sep=r'\s+', comment='#',
                           names=['X', 'Y', 'V_Z'])
    obs      = df[['X', 'Y', 'V_Z']].values.astype(np.float32)
    mean_obs = obs.mean(0)
    std_obs  = obs.std(0)
    obs_norm = (obs - mean_obs) / std_obs
    print(f"  Stars loaded : {len(obs)}")

    # ── Load NFlow v2 weights ─────────────────────────────────────────────────
    weights_index = f'{base_name}_v2.weights.index'
    if not os.path.exists(weights_index):
        print(f"  [Skip] Weights not found: {base_name}_v2.weights.*")
        continue

    model = NFlow(n_layers=12, hidden=256)
    _     = model.log_prob(tf.constant(obs_norm[:10]))     # build graph
    ckpt  = tf.train.Checkpoint(model=model)
    ckpt.restore(f'{base_name}_v2.weights').expect_partial()
    print("  Weights loaded.")

    # ── Sample & compute NFlow profiles ──────────────────────────────────────
    print(f"  Sampling {N_SAMPLE:,} points from NFlow...")
    samples_norm = model.sample(N_SAMPLE)
    try:
        r_grid, n_r, n_spl_nf, sigma_r2_b0, sigma_r2_bm5, M_nf = \
            compute_nflow_profiles(samples_norm, mean_obs, std_obs)
    except Exception as e:
        print(f"  [Warning] NFlow profile computation failed: {e}")
        continue

    # ── Load ground truth (6D data) ───────────────────────────────────────────
    gt = load_ground_truth(input_filename, len(obs))
    
    # ── Analytical true M(r) ─────────────────────────────────────────────────
    print("  Computing analytical M(r)...")
    r_true = np.logspace(np.log10(r_grid[0] * 0.5),
                         np.log10(r_grid[-1] * 2.0), 60)
    M_true = np.array([enclosed_mass(r, params) for r in r_true])

    # ── True Jeans σ_r²(r) using 6D stellar n(r) + analytical DM ────────────
    r_jeans           = r_grid.copy()
    sigma_true_jeans  = np.full(len(r_jeans), np.nan)

    if gt is not None:
        _, _, _, n_spl_6d, r_min_6d, r_max_6d = gt
        # only compute where 6D data has coverage
        mask_cov = (r_jeans > r_min_6d * 1.05) & (r_jeans < r_max_6d * 0.9)
        if mask_cov.sum() > 0:
            print("  Computing true Jeans σ_r² (this may take a minute)...")
            sj = true_jeans_sigma(
                r_jeans[mask_cov], n_spl_6d, params,
                r_outer=r_max_6d * 0.9)
            sigma_true_jeans[mask_cov] = sj
    else:
        print("  [Note] No 6D data → skipping true Jeans σ_r².")

    # ── Plot ──────────────────────────────────────────────────────────────────
    make_plot(base_name, profile_key,
              r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_nf,
              r_true, M_true,
              r_jeans, sigma_true_jeans,
              gt)

print(f"\n{'='*60}")
print("All done!  One PNG saved per data file.")
print(f"{'='*60}")