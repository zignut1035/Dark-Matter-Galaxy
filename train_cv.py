"""
train_cv.py  –  Run the normalizing flow N_RUNS times (default 20)
               on every data file, each with a different random seed.
               Produces publication-quality plots matching Lim et al. (2025)
               arXiv:2503.00763.

FIXES vs previous version:
  1. DM_PARAMS keys renamed to match actual filenames:
       'core_data1', 'core_data2', 'core_data3',
       'cusp_data1', 'cusp_data2', 'cusp_data3'
  2. add_panel_pair(): plots log[quantity] on LINEAR y-axis (not loglog),
     matching the professor's figure exactly.
  3. Grey "Low statistics" shaded regions added at r < R_LOW and r > R_HIGH.
  4. True unprojected data shown as grey points with Poisson error bars.
  5. Pull panel interpolation restricted to r_true domain (no flat extrapolation).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS   = 20
N_LAYERS = 12
HIDDEN   = 256
EPOCHS   = 100
BATCH    = 256
N_SAMPLE = 100_000

G_CONST  = 4.30091e-6   # kpc (km/s)^2 / M_sun

# Low-statistics mask thresholds (same as professor's grey bands)
R_LOW  = 0.1   # kpc  — mask below this
R_HIGH = 10.0  # kpc  — mask above this

# ── File selection ────────────────────────────────────────────────────────────
ALL_FILES = [
    '3D_data/observables_core_data1.dat',
    '3D_data/observables_core_data2.dat',
    '3D_data/observables_core_data3.dat',
    '3D_data/observables_cusp_data1.dat',
    '3D_data/observables_cusp_data2.dat',
    '3D_data/observables_cusp_data3.dat',
]

slurm_idx = os.environ.get('SLURM_FILE_IDX')
if slurm_idx is not None:
    idx = int(slurm_idx)
    DATA_FILES = [ALL_FILES[idx]]
    print(f"SLURM array task {idx}: processing {DATA_FILES[0]}")
else:
    DATA_FILES = [f for f in ALL_FILES if os.path.exists(f)]
    if not DATA_FILES:
        raise FileNotFoundError("No data files found — check 3D_data/ directory")
    print(f"Local mode: processing {len(DATA_FILES)} files: {DATA_FILES}")


# ── DM parameters (generalized NFW) ──────────────────────────────────────────
DM_PARAMS = {
    'core_data1': dict(rho_DM=0.064e9,  rs_DM=1.0,  alpha=1.0, beta_dm=3.0, gamma=0.0),
    'core_data2': dict(rho_DM=0.6e9,    rs_DM=0.45, alpha=1.0, beta_dm=3.0, gamma=0.0),
    'core_data3': dict(rho_DM=1.5e9,    rs_DM=0.27, alpha=1.0, beta_dm=3.0, gamma=0.0),
    'cusp_data1': dict(rho_DM=0.064e9,  rs_DM=1.0,  alpha=1.0, beta_dm=3.0, gamma=1.0),
    'cusp_data2': dict(rho_DM=0.002e9,  rs_DM=5.0,  alpha=1.0, beta_dm=3.0, gamma=1.0),
    'cusp_data3': dict(rho_DM=0.0004e9, rs_DM=13.0, alpha=1.0, beta_dm=3.0, gamma=1.0),
}


def rho_true(r, rho_DM, rs_DM, alpha, beta_dm, gamma):
    x = np.maximum(r / rs_DM, 1e-10)
    return rho_DM / (x**gamma * (1.0 + x**alpha)**((beta_dm - gamma) / alpha))


def M_true(r, **params):
    if r <= 0:
        return 0.0
    val, _ = quad(
        lambda rp: 4.0 * np.pi * rp**2 * rho_true(rp, **params),
        0.0, r, limit=200, epsabs=1e-6, epsrel=1e-6
    )
    return val


def get_dm_params(base_name):
    for key in sorted(DM_PARAMS.keys(), key=len, reverse=True):
        if key in base_name:
            return DM_PARAMS[key]
    return None


def compute_true_dm_profiles(r_grid, params):
    rho_arr = np.array([rho_true(r, **params) for r in r_grid])
    M_arr   = np.array([M_true(r,  **params) for r in r_grid])
    return rho_arr, M_arr


# ── Coupling layer ────────────────────────────────────────────────────────────
class CouplingLayer(keras.layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.net_s = keras.Sequential([
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(2, kernel_initializer='zeros', bias_initializer='zeros')
        ])
        self.net_t = keras.Sequential([
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(2, kernel_initializer='zeros', bias_initializer='zeros')
        ])

    def call(self, x, reverse=False):
        x0, x1 = x[:, :1], x[:, 1:]
        s = tf.clip_by_value(self.net_s(x0), -5.0, 5.0)
        t = self.net_t(x0)
        if not reverse:
            y1      = x1 * tf.exp(s) + t
            log_det = tf.reduce_sum(s, axis=1)
            return tf.concat([x0, y1], axis=1), log_det
        else:
            return tf.concat([x0, (x1 - t) * tf.exp(-s)], axis=1)


# ── Normalizing flow ──────────────────────────────────────────────────────────
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
            x**2 + tf.cast(tf.math.log(2 * np.pi), tf.float32), axis=1)
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


# ── Profile computation ───────────────────────────────────────────────────────
def compute_profiles(samples, mean, std):
    s    = samples * std + mean
    X_s, Y_s, VZ_s = s[:, 0], s[:, 1], s[:, 2]
    R_s  = np.sqrt(X_s**2 + Y_s**2)

    lo, hi = np.percentile(R_s, 1), np.percentile(R_s, 99)
    mask   = (R_s > lo) & (R_s < hi)
    R_s, VZ_s = R_s[mask], VZ_s[mask]

    R_bins = np.logspace(np.log10(R_s.min()*1.01), np.log10(R_s.max()*0.99), 25)
    R_mid  = 0.5 * (R_bins[:-1] + R_bins[1:])

    counts, _ = np.histogram(R_s, bins=R_bins)
    area       = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)
    Sigma      = counts / area

    sigma_los2 = np.array([
        VZ_s[(R_s >= R_bins[k]) & (R_s < R_bins[k+1])].var()
        for k in range(len(R_bins) - 1)
    ])

    log_Sigma_spl = UnivariateSpline(np.log(R_mid), np.log(Sigma + 1e-30), s=1)
    dSigma_dR = lambda R: (
        np.exp(log_Sigma_spl(np.log(R))) / R
        * log_Sigma_spl.derivative()(np.log(R))
    )

    def abel_density(r, R_max=None):
        if R_max is None: R_max = R_mid[-1]
        if r >= R_max: return 0.0
        val, _ = quad(lambda R: dSigma_dR(R) / np.sqrt(R**2 - r**2),
                      r * 1.001, R_max)
        return -val / np.pi

    r_grid = np.logspace(np.log10(R_mid[3]), np.log10(R_mid[-2]), 30)
    n_r    = np.array([abel_density(r) for r in r_grid])

    Sigma_slos2 = Sigma * sigma_los2
    log_SS_spl  = UnivariateSpline(np.log(R_mid), np.log(Sigma_slos2 + 1e-30), s=1)
    dSS_dR = lambda R: (
        np.exp(log_SS_spl(np.log(R))) / R
        * log_SS_spl.derivative()(np.log(R))
    )
    n_spl = UnivariateSpline(np.log(r_grid), np.log(n_r + 1e-30), s=1)

    def jeans_sigma_r2(r, beta=0.0, R_max=None):
        if R_max is None: R_max = R_mid[-1]
        if r >= R_max: return 0.0
        val, _ = quad(
            lambda R: dSS_dR(R) * (1 - beta*(r/R)**2) / np.sqrt(R**2 - r**2),
            r * 1.001, R_max)
        nr = np.exp(n_spl(np.log(r)))
        return -val / (np.pi * nr) if nr > 0 else 0.0

    sigma_r2_b0  = np.array([jeans_sigma_r2(r, beta=0.0)  for r in r_grid])
    sigma_r2_bm5 = np.array([jeans_sigma_r2(r, beta=-0.5) for r in r_grid])

    dln_n_dln_r = n_spl.derivative()(np.log(r_grid))

    good = sigma_r2_b0 > 0
    if good.sum() >= 4:
        log_sig2_spl_b0 = UnivariateSpline(
            np.log(r_grid[good]), np.log(sigma_r2_b0[good]), s=3, ext=3)
        dln_sig2_dln_r_b0 = log_sig2_spl_b0.derivative()(np.log(r_grid))
    else:
        dln_sig2_dln_r_b0 = np.zeros_like(r_grid)

    M_r_b0_raw = -(r_grid * sigma_r2_b0 / G_CONST) * (dln_n_dln_r + dln_sig2_dln_r_b0)
    M_r_b0     = np.where(M_r_b0_raw > 0, M_r_b0_raw, np.nan)

    valid = np.isfinite(M_r_b0) & (M_r_b0 > 0)
    if valid.sum() >= 4:
        log_M_spl = UnivariateSpline(
            np.log(r_grid[valid]), np.log(M_r_b0[valid]), s=3, ext=3)
        dM_dr = np.exp(log_M_spl(np.log(r_grid))) / r_grid \
                * log_M_spl.derivative()(np.log(r_grid))
        rho_r = dM_dr / (4 * np.pi * r_grid**2)
        rho_r = np.where(rho_r > 0, rho_r, np.nan)
    else:
        rho_r = np.full_like(r_grid, np.nan)

    return r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_r_b0, rho_r


# ── Ground truth loader ───────────────────────────────────────────────────────
def load_ground_truth(input_filename):
    """
    Load 6D file.
    Returns (gt_r_mid, n_true, n_err, sigma2_true, sigma2_err)
    where n_err uses sqrt(N)/V Poisson errors and sigma2_err = sigma/sqrt(2*(N-1)).
    """
    gt_filename = (input_filename
                   .replace('3D_data', '6D_data')
                   .replace('observables_', 'Mock_isotropic_'))
    try:
        df_6d = pd.read_csv(gt_filename, sep=r'\s+', comment='#',
                            names=['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
    except FileNotFoundError:
        print(f"  [Warning] Ground truth not found: {gt_filename}")
        return None

    r_6d  = np.sqrt(df_6d['X']**2 + df_6d['Y']**2 + df_6d['Z']**2)
    vr_6d = ((df_6d['X']*df_6d['VX'] + df_6d['Y']*df_6d['VY']
              + df_6d['Z']*df_6d['VZ']) / r_6d)

    gt_bins  = np.logspace(np.log10(r_6d.min()*1.01),
                           np.log10(r_6d.max()*0.99), 30)
    gt_r_mid = np.sqrt(gt_bins[:-1] * gt_bins[1:])

    counts_6d, _ = np.histogram(r_6d, bins=gt_bins)
    vol_6d        = (4/3) * np.pi * (gt_bins[1:]**3 - gt_bins[:-1]**3)
    n_true        = counts_6d / vol_6d
    # Poisson error: delta_n = sqrt(N) / V
    n_err         = np.sqrt(np.maximum(counts_6d, 1)) / vol_6d

    sigma2_true = []
    sigma2_err  = []
    for k in range(len(gt_bins) - 1):
        m  = (r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])
        Nk = m.sum()
        if Nk > 1:
            s2 = vr_6d[m].var()
            sigma2_true.append(s2)
            # Standard error of variance ~ sigma^2 * sqrt(2/(N-1))
            sigma2_err.append(s2 * np.sqrt(2.0 / (Nk - 1)))
        else:
            sigma2_true.append(np.nan)
            sigma2_err.append(np.nan)

    return gt_r_mid, n_true, n_err, np.array(sigma2_true), np.array(sigma2_err)


# ── Single training run ───────────────────────────────────────────────────────
def train_one_run(obs_norm, seed, run_id):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    dataset = (tf.data.Dataset
               .from_tensor_slices(obs_norm)
               .shuffle(len(obs_norm), seed=seed)
               .batch(BATCH).prefetch(1))

    model     = NFlow(n_layers=N_LAYERS, hidden=HIDDEN)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(model.log_prob(x))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_nll, patience = np.inf, 0
    for epoch in range(EPOCHS):
        for x in dataset:
            train_step(x)
        if epoch % 20 == 0:
            nll = -tf.reduce_mean(model.log_prob(tf.constant(obs_norm))).numpy()
            lr  = float(optimizer.learning_rate)
            print(f"    [run {run_id:02d}] Epoch {epoch:3d}  NLL={nll:.4f}  LR={lr:.2e}")
            if nll < best_nll - 1e-3:
                best_nll, patience = nll, 0
            else:
                patience += 1
                if patience >= 5:
                    optimizer.learning_rate.assign(max(lr * 0.5, 1e-5))
                    patience = 0

    return model, best_nll


# ── percentile_bands ──────────────────────────────────────────────────────────
def percentile_bands(arr_list):
    """
    Returns statistics in LOG10 space: (mu, p16, p84, p2p5, p97p5).
    All values are log10(quantity) — kept in log space for the new plot style.
    """
    arr = np.array(arr_list, dtype=float)
    arr = np.where(arr > 0, arr, np.nan)
    log_arr = np.where(np.isfinite(arr), np.log10(arr), np.nan)

    mu    = np.nanmean(log_arr,             axis=0)
    p16   = np.nanpercentile(log_arr, 16,   axis=0)
    p84   = np.nanpercentile(log_arr, 84,   axis=0)
    p2p5  = np.nanpercentile(log_arr,  2.5, axis=0)
    p97p5 = np.nanpercentile(log_arr, 97.5, axis=0)
    return mu, p16, p84, p2p5, p97p5


# ── Panel pair: log-linear upper + Pull lower ─────────────────────────────────
def add_panel_pair(fig, gs_top, gs_bot,
                   r_grid,
                   mu_b0, p16, p84, p2p5, p97p5,
                   mu_bm5,
                   r_true, val_true,          # analytic smooth curve
                   r_data, val_data, err_data, # binned data points w/ error bars
                   ylabel, title,
                   r_low=R_LOW, r_high=R_HIGH):
    """
    Upper panel: log10(quantity) on a LINEAR y-axis, log x-axis.
      — Matches professor's figure (log[] on y-label, linear tick spacing).
    Lower panel: Pull = (estimate - true) / (0.5*(p84-p16)).

    Grey shading marks low-statistics regions r < r_low and r > r_high.
    """
    YELLOW = '#FFD700'
    GREEN  = '#228B22'
    GREY   = '#AAAAAA'
    LW     = 2.0

    ax  = fig.add_subplot(gs_top)
    axp = fig.add_subplot(gs_bot, sharex=ax)

    xmin = r_grid[0] * 0.8
    xmax = r_grid[-1] * 1.2

    # ── Grey low-statistics shading ───────────────────────────────────────────
    for panel in (ax, axp):
        panel.axvspan(xmin,   r_low,  color=GREY, alpha=0.45, zorder=0)
        panel.axvspan(r_high, xmax,   color=GREY, alpha=0.45, zorder=0)

    # ── Uncertainty bands (in log space on linear axis) ───────────────────────
    valid_band = np.isfinite(p2p5) & np.isfinite(p97p5)
    if valid_band.any():
        ax.fill_between(r_grid,
                        np.where(np.isfinite(p2p5),  p2p5,  np.nan),
                        np.where(np.isfinite(p97p5), p97p5, np.nan),
                        color=YELLOW, alpha=1.0, zorder=1)
        ax.fill_between(r_grid,
                        np.where(np.isfinite(p16), p16, np.nan),
                        np.where(np.isfinite(p84), p84, np.nan),
                        color=GREEN,  alpha=1.0, zorder=2)

    # ── Mean estimate lines ───────────────────────────────────────────────────
    valid_b0  = np.isfinite(mu_b0)
    valid_bm5 = np.isfinite(mu_bm5)
    if valid_b0.any():
        ax.semilogx(r_grid[valid_b0],  mu_b0[valid_b0],
                    color='green', lw=LW, ls='--', zorder=3,
                    label=r'Estimated, $\beta=0$')
    if valid_bm5.any():
        ax.semilogx(r_grid[valid_bm5], mu_bm5[valid_bm5],
                    color='black', lw=LW, ls=':',  zorder=3,
                    label=r'Estimated, $\beta=-0.5$')

    # ── Analytic true curve ───────────────────────────────────────────────────
    if r_true is not None and val_true is not None:
        mask_t = np.isfinite(val_true) & (val_true > 0)
        ax.semilogx(r_true[mask_t], np.log10(val_true[mask_t]),
                    color='red', lw=LW, ls='-', zorder=4,
                    label='True: analytic')

    # ── Binned data points with error bars ────────────────────────────────────
    if r_data is not None and val_data is not None:
        mask_d = np.isfinite(val_data) & (val_data > 0)
        log_val  = np.log10(val_data[mask_d])
        log_yerr = np.where(
            err_data[mask_d] > 0,
            err_data[mask_d] / (val_data[mask_d] * np.log(10)),
            0.0
        )
        ax.errorbar(r_data[mask_d], log_val,
                    yerr=log_yerr,
                    fmt='o', color='grey', ms=4, lw=1.0,
                    zorder=5, label='True: unprojected data')

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.2)
    ax.set_xlim(xmin, xmax)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ── Pull panel ────────────────────────────────────────────────────────────
    # sigma_band in log10 space: half of (p84 - p16)
    sigma_log = np.where(
        np.isfinite(p84) & np.isfinite(p16),
        np.maximum((p84 - p16) / 2.0, 1e-10),
        np.nan
    )

    axp.fill_between(r_grid, -2, 2, color=YELLOW, alpha=1.0, zorder=1)
    axp.fill_between(r_grid, -1, 1, color=GREEN,  alpha=1.0, zorder=2)
    axp.axhline(0, color='black', ls='--', lw=1.0, zorder=3)

    # Analytic true curve pull
    if r_true is not None and val_true is not None:
        in_range    = (r_grid >= r_true.min()) & (r_grid <= r_true.max())
        true_interp = np.full_like(r_grid, np.nan)
        mask_t2     = np.isfinite(val_true) & (val_true > 0)
        if mask_t2.sum() >= 2:
            true_interp[in_range] = np.interp(
                r_grid[in_range], r_true[mask_t2], np.log10(val_true[mask_t2])
            )
        pull_true = (true_interp - mu_b0) / sigma_log
        pull_b0   = (mu_b0  - true_interp) / sigma_log
        pull_bm5  = (mu_bm5 - true_interp) / sigma_log

        axp.semilogx(r_grid, pull_b0,   color='green', lw=LW, ls='--', zorder=4)
        axp.semilogx(r_grid, pull_bm5,  color='black', lw=LW, ls=':',  zorder=4)
        axp.semilogx(r_grid, pull_true, color='red',   lw=LW, ls='-',  zorder=5)

    # Binned data point pulls
    if r_data is not None and val_data is not None:
        mask_d = np.isfinite(val_data) & (val_data > 0)
        mu_at_data = np.interp(r_data[mask_d], r_grid, mu_b0,
                               left=np.nan, right=np.nan)
        sig_at_data = np.interp(r_data[mask_d], r_grid, sigma_log,
                                left=np.nan, right=np.nan)
        pull_data = (np.log10(val_data[mask_d]) - mu_at_data) / sig_at_data
        axp.errorbar(r_data[mask_d], pull_data,
                     fmt='o', color='grey', ms=4, lw=1.0, zorder=6)

    axp.set_ylim(-3.5, 3.5)
    axp.set_ylabel('Pull', fontsize=10)
    axp.set_xlabel('r [kpc]', fontsize=11)
    axp.grid(True, which='both', alpha=0.2)
    axp.set_xlim(xmin, xmax)


# ── 4-column plot ─────────────────────────────────────────────────────────────
def make_lim_style_plot(base_name, r_grid,
                        all_n_r, all_sigma_b0, all_sigma_bm5,
                        all_M_r, all_rho_r,
                        gt, n_runs):
    """
    Four-column figure: n(r) | sigma_r^2 | rho(r) | M(r), each with Pull.
    Y-axes show log10(quantity) on a LINEAR scale, matching Lim et al. (2025).
    """
    mu_n,   p16_n,   p84_n,   p2p5_n,   p97p5_n   = percentile_bands(all_n_r)
    mu_sb0, p16_sb0, p84_sb0, p2p5_sb0, p97p5_sb0 = percentile_bands(all_sigma_b0)
    mu_sbm5 = np.nanmean(
        np.where(np.array(all_sigma_bm5) > 0,
                 np.log10(np.maximum(np.array(all_sigma_bm5), 1e-30)),
                 np.nan),
        axis=0)
    mu_M,   p16_M,   p84_M,   p2p5_M,   p97p5_M   = percentile_bands(all_M_r)
    mu_rho, p16_rho, p84_rho, p2p5_rho, p97p5_rho = percentile_bands(all_rho_r)

    mu_M_bm5   = mu_M
    mu_rho_bm5 = mu_rho

    # Ground truth n(r) and sigma_r^2(r) from 6D file
    if gt is not None:
        gt_r, n_true, n_err, sigma2_true, sigma2_err = gt
    else:
        gt_r = n_true = n_err = sigma2_true = sigma2_err = None

    # Analytic DM truth
    dm_params = get_dm_params(base_name)
    if dm_params is not None:
        print(f"  Computing analytic DM truth for '{base_name}'...")
        rho_true_arr, M_true_arr = compute_true_dm_profiles(r_grid, dm_params)
    else:
        print(f"  [Warning] No DM_PARAMS match for '{base_name}' — DM truth omitted")
        rho_true_arr = M_true_arr = None

    fig = plt.figure(figsize=(24, 7))
    outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.40)

    panels = [
        # title, ylabel,
        # mu_b0, p16, p84, p2p5, p97p5, mu_bm5,
        # r_true_curve, val_true_curve,
        # r_data, val_data, err_data
        ('Stellar number density',
         r'$\log\,[n(r)\,/\,\mathrm{kpc}^{-3}]$',
         mu_n,   p16_n,   p84_n,   p2p5_n,   p97p5_n,   mu_n,
         gt_r, n_true,
         gt_r, n_true, n_err),

        ('Radial velocity dispersion',
         r'$\log\,[\sigma_r^2\,/\,(\mathrm{km\,s}^{-1})^2]$',
         mu_sb0, p16_sb0, p84_sb0, p2p5_sb0, p97p5_sb0, mu_sbm5,
         gt_r, sigma2_true,
         gt_r, sigma2_true, sigma2_err),

        ('DM mass density',
         r'$\log\,[\rho(r)\,/\,\mathrm{M}_\odot\,\mathrm{kpc}^{-3}]$',
         mu_rho, p16_rho, p84_rho, p2p5_rho, p97p5_rho, mu_rho_bm5,
         r_grid, rho_true_arr,
         None, None, None),

        ('Enclosed mass',
         r'$\log\,[M(r)\,/\,\mathrm{M}_\odot]$',
         mu_M,   p16_M,   p84_M,   p2p5_M,   p97p5_M,   mu_M_bm5,
         r_grid, M_true_arr,
         None, None, None),
    ]

    for col, (title, ylabel,
              mu_b0, p16, p84, p2p5, p97p5, mu_bm5,
              r_tc, val_tc,
              r_dp, val_dp, err_dp) in enumerate(panels):

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3, 1], hspace=0.05)
        add_panel_pair(fig, inner[0], inner[1],
                       r_grid,
                       mu_b0, p16, p84, p2p5, p97p5,
                       mu_bm5,
                       r_tc, val_tc,
                       r_dp, val_dp, err_dp,
                       ylabel, title)

    fig.suptitle(
        f'{base_name}  |  {n_runs} independent NFlow runs  '
        r'(green = 1$\sigma$,  yellow = 2$\sigma$)',
        fontsize=12, y=1.02)

    out_png = f'{base_name}_cv{n_runs}_lim_style.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
for input_filename in DATA_FILES:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {input_filename}  (CV with {N_RUNS} runs)")
    print(f"{'='*60}")

    df       = pd.read_csv(input_filename, sep=r'\s+', comment='#',
                           names=['X', 'Y', 'V_Z'])
    obs      = df[['X', 'Y', 'V_Z']].values.astype(np.float32)
    mean_obs = obs.mean(0);  std_obs = obs.std(0)
    obs_norm = (obs - mean_obs) / std_obs
    print(f"  Stars loaded: {len(obs)}")

    all_n_r       = []
    all_sigma_b0  = []
    all_sigma_bm5 = []
    all_M_r       = []
    all_rho_r     = []
    all_nll       = []

    for run in range(N_RUNS):
        seed = 42 + run
        print(f"\n  -- Run {run+1}/{N_RUNS}  (seed={seed}) --")

        model, best_nll = train_one_run(obs_norm, seed, run + 1)
        all_nll.append(best_nll)

        print(f"  Sampling {N_SAMPLE:,} points...")
        samples_norm = model.sample(N_SAMPLE)
        r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_r_b0, rho_r = compute_profiles(
            samples_norm, mean_obs, std_obs)

        all_n_r.append(n_r)
        all_sigma_b0.append(sigma_r2_b0)
        all_sigma_bm5.append(sigma_r2_bm5)
        all_M_r.append(M_r_b0)
        all_rho_r.append(rho_r)

        run_weights = f'{base_name}_run{run+1:02d}.weights'
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.write(run_weights)
        print(f"  Weights saved -> {run_weights}")

    nll_arr = np.array(all_nll)
    print(f"\n  NLL across {N_RUNS} runs: "
          f"mean={nll_arr.mean():.4f}  std={nll_arr.std():.4f}")

    np.savez(f'{base_name}_cv{N_RUNS}.npz',
             r_grid        = r_grid,
             n_r_all       = np.array(all_n_r),
             sigma_b0_all  = np.array(all_sigma_b0),
             sigma_bm5_all = np.array(all_sigma_bm5),
             M_r_all       = np.array(all_M_r,   dtype=object),
             rho_r_all     = np.array(all_rho_r, dtype=object),
             nll_all       = nll_arr,
             mean_obs      = mean_obs,
             std_obs       = std_obs)
    print(f"  All profiles saved -> {base_name}_cv{N_RUNS}.npz")

    gt = load_ground_truth(input_filename)
    make_lim_style_plot(base_name, r_grid,
                        all_n_r, all_sigma_b0, all_sigma_bm5,
                        all_M_r, all_rho_r,
                        gt, N_RUNS)

print(f"\n{'='*60}")
print(f"Done!  {N_RUNS}-run CV complete for all data files.")
print(f"{'='*60}")