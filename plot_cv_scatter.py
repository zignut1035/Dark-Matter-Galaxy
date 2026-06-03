"""
plot_cv_scatter.py  —  Reproduce scatter plots and DM profile plots
                       matching Lim et al. (2025) arXiv:2503.00763.

FIXES vs previous version:
  1. DM_PARAMS keys renamed to match actual filenames.
  2. make_profile_plot(): log10(quantity) plotted on LINEAR y-axis (semilogx),
     matching the professor's figure style.
  3. Grey "Low statistics" bands added at r < R_LOW and r > R_HIGH.
  4. True unprojected data shown as grey points with error bars (loaded from npz).
  5. Pull panel uses log-space sigma_band and domain-restricted interpolation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import quad
import os

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS   = 20
N_SAMPLE = 100_000
N_LAYERS = 12
HIDDEN   = 256

R_LOW  = 0.1   # kpc — grey low-statistics band below this
R_HIGH = 10.0  # kpc — grey low-statistics band above this

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
    print(f"Local mode: processing {len(DATA_FILES)} files")


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


def percentile_bands(arr_list):
    """Log10-space statistics, NaN-safe. Returns values in log10 space."""
    arr = np.array(arr_list, dtype=float)
    arr = np.where(arr > 0, arr, np.nan)
    log_arr = np.where(np.isfinite(arr), np.log10(arr), np.nan)
    mu    = np.nanmean(log_arr,             axis=0)
    p16   = np.nanpercentile(log_arr, 16,   axis=0)
    p84   = np.nanpercentile(log_arr, 84,   axis=0)
    p2p5  = np.nanpercentile(log_arr,  2.5, axis=0)
    p97p5 = np.nanpercentile(log_arr, 97.5, axis=0)
    return mu, p16, p84, p2p5, p97p5


# ── Coupling layer & NFlow ────────────────────────────────────────────────────
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


class NFlow(keras.Model):
    def __init__(self, n_layers=N_LAYERS, hidden=HIDDEN, **kwargs):
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


# ── Ground truth loader (for n(r) / sigma^2 data points) ─────────────────────
def load_ground_truth(input_filename):
    gt_filename = (input_filename
                   .replace('3D_data', '6D_data')
                   .replace('observables_', 'Mock_isotropic_'))
    try:
        df_6d = pd.read_csv(gt_filename, sep=r'\s+', comment='#',
                            names=['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
    except FileNotFoundError:
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
    n_err         = np.sqrt(np.maximum(counts_6d, 1)) / vol_6d

    sigma2_true = []
    sigma2_err  = []
    for k in range(len(gt_bins) - 1):
        m  = (r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])
        Nk = m.sum()
        if Nk > 1:
            s2 = vr_6d[m].var()
            sigma2_true.append(s2)
            sigma2_err.append(s2 * np.sqrt(2.0 / (Nk - 1)))
        else:
            sigma2_true.append(np.nan)
            sigma2_err.append(np.nan)

    return gt_r_mid, n_true, n_err, np.array(sigma2_true), np.array(sigma2_err)


# ── DM profile plot ───────────────────────────────────────────────────────────
def make_profile_plot(base_name, npz_file, input_filename):
    """
    Plot rho(r) and M(r) with log10 values on a linear y-axis (semilogx),
    matching the professor's figure style. Grey low-statistics shading included.
    """
    data = np.load(npz_file, allow_pickle=True)
    r_grid    = data['r_grid']
    rho_r_all = list(data['rho_r_all'])
    M_r_all   = list(data['M_r_all'])

    mu_rho, p16_rho, p84_rho, p2p5_rho, p97p5_rho = percentile_bands(rho_r_all)
    mu_M,   p16_M,   p84_M,   p2p5_M,   p97p5_M   = percentile_bands(M_r_all)

    dm_params = get_dm_params(base_name)
    if dm_params is not None:
        rho_true_arr, M_true_arr = compute_true_dm_profiles(r_grid, dm_params)
        print(f"  [profile plot] Analytic DM truth computed for '{base_name}'")
    else:
        rho_true_arr = M_true_arr = None
        print(f"  [profile plot] No DM_PARAMS match — true curves omitted")

    YELLOW = '#FFD700'
    GREEN  = '#228B22'
    GREY   = '#AAAAAA'
    LW     = 2.0

    xmin = r_grid[0] * 0.8
    xmax = r_grid[-1] * 1.2

    fig = plt.figure(figsize=(12, 7))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40)

    for col, (title, ylabel, mu, p16, p84, p2p5, p97p5, true_arr) in enumerate([
        ('DM mass density',
         r'$\log\,[\rho(r)\,/\,\mathrm{M}_\odot\,\mathrm{kpc}^{-3}]$',
         mu_rho, p16_rho, p84_rho, p2p5_rho, p97p5_rho, rho_true_arr),

        ('Enclosed mass',
         r'$\log\,[M(r)\,/\,\mathrm{M}_\odot]$',
         mu_M,   p16_M,   p84_M,   p2p5_M,   p97p5_M,   M_true_arr),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3, 1], hspace=0.05)

        ax  = fig.add_subplot(inner[0])
        axp = fig.add_subplot(inner[1], sharex=ax)

        # Grey low-statistics shading
        for panel in (ax, axp):
            panel.axvspan(xmin,   R_LOW,  color=GREY, alpha=0.45, zorder=0)
            panel.axvspan(R_HIGH, xmax,   color=GREY, alpha=0.45, zorder=0)

        # Bands — log10 values on linear axis
        valid = np.isfinite(p2p5) & np.isfinite(p97p5)
        if valid.any():
            ax.fill_between(r_grid,
                            np.where(np.isfinite(p2p5),  p2p5,  np.nan),
                            np.where(np.isfinite(p97p5), p97p5, np.nan),
                            color=YELLOW, alpha=1.0, zorder=1, label=r'2$\sigma$')
            ax.fill_between(r_grid,
                            np.where(np.isfinite(p16), p16, np.nan),
                            np.where(np.isfinite(p84), p84, np.nan),
                            color=GREEN, alpha=1.0, zorder=2, label=r'1$\sigma$')

        valid_mu = np.isfinite(mu)
        if valid_mu.any():
            ax.semilogx(r_grid[valid_mu], mu[valid_mu],
                        color='green', lw=LW, ls='--', zorder=3,
                        label=r'Estimated, $\beta=0$')

        if true_arr is not None:
            mask = np.isfinite(true_arr) & (true_arr > 0)
            ax.semilogx(r_grid[mask], np.log10(true_arr[mask]),
                        color='red', lw=LW, ls='-', zorder=4,
                        label='True: analytic')

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title,   fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, which='both', alpha=0.2)
        ax.set_xlim(xmin, xmax)
        plt.setp(ax.get_xticklabels(), visible=False)

        # Pull panel
        sigma_log = np.where(
            np.isfinite(p84) & np.isfinite(p16),
            np.maximum((p84 - p16) / 2.0, 1e-10),
            np.nan
        )
        axp.fill_between(r_grid, -2, 2, color=YELLOW, alpha=1.0, zorder=1)
        axp.fill_between(r_grid, -1, 1, color=GREEN,  alpha=1.0, zorder=2)
        axp.axhline(0, color='black', ls='--', lw=1.0, zorder=3)

        if true_arr is not None:
            valid_true = np.isfinite(true_arr) & (true_arr > 0)
            in_range   = (r_grid >= r_grid[valid_true].min()) & \
                         (r_grid <= r_grid[valid_true].max())
            true_interp = np.full_like(r_grid, np.nan)
            if valid_true.sum() >= 2:
                true_interp[in_range] = np.interp(
                    r_grid[in_range],
                    r_grid[valid_true],
                    np.log10(true_arr[valid_true])
                )
            pull = (mu - true_interp) / sigma_log
            axp.semilogx(r_grid, pull, color='red', lw=LW, ls='-', zorder=4)

        axp.set_ylim(-3.5, 3.5)
        axp.set_ylabel('Pull', fontsize=10)
        axp.set_xlabel('r [kpc]', fontsize=11)
        axp.grid(True, which='both', alpha=0.2)
        axp.set_xlim(xmin, xmax)

    fig.suptitle(
        f'{base_name}  |  {N_RUNS} NFlow runs  '
        r'(green = 1$\sigma$,  yellow = 2$\sigma$)',
        fontsize=12, y=1.02)

    out_png = f'{base_name}_cv{N_RUNS}_dm_profiles.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_png}")


# ── Scatter plot ──────────────────────────────────────────────────────────────
def make_scatter_plot(base_name, obs, all_generated, mean_obs, std_obs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
    fig.subplots_adjust(wspace=0.05)

    vmin = np.percentile(obs[:, 2], 2)
    vmax = np.percentile(obs[:, 2], 98)

    ax = axes[0]
    ax.set_facecolor('black')
    sc = ax.scatter(obs[:, 0], obs[:, 1], c=obs[:, 2],
                    cmap='coolwarm', s=0.3, alpha=0.6,
                    vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(f'Real 3D Data (Projected)\n{len(obs):,} stars',
                 color='white', fontsize=12)
    ax.set_xlabel('X [kpc]', color='white', fontsize=11)
    ax.set_ylabel('Y [kpc]', color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    ax2 = axes[1]
    ax2.set_facecolor('black')
    all_gen  = np.concatenate(all_generated, axis=0)
    idx      = np.random.choice(len(all_gen), size=min(100_000, len(all_gen)), replace=False)
    gen_plot = all_gen[idx]
    ax2.scatter(gen_plot[:, 0], gen_plot[:, 1], c=gen_plot[:, 2],
                cmap='coolwarm', s=0.3, alpha=0.4,
                vmin=vmin, vmax=vmax, rasterized=True)
    ax2.set_title(f'Generated Data (Normalizing Flow)\n{N_RUNS} runs × {N_SAMPLE:,} stars',
                  color='white', fontsize=12)
    ax2.set_xlabel('X [kpc]', color='white', fontsize=11)
    ax2.set_ylabel('Y [kpc]', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Line-of-Sight Velocity (Vz) [km/s]', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    lim = max(np.abs(obs[:, :2]).max(), np.abs(gen_plot[:, :2]).max()) * 1.05
    for a in axes:
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')

    plt.suptitle(base_name.replace('_', ' '), color='white', fontsize=13, y=1.01)
    out_png = f'{base_name}_cv{N_RUNS}_scatter.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved -> {out_png}")


# ── Grid plot ─────────────────────────────────────────────────────────────────
def make_grid_plot(base_name, obs, all_generated, mean_obs, std_obs):
    cols, rows = 5, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16), facecolor='black')
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    vmin = np.percentile(obs[:, 2], 2)
    vmax = np.percentile(obs[:, 2], 98)
    lim  = np.abs(obs[:, :2]).max() * 1.1

    for run_idx, (ax, gen) in enumerate(zip(axes.flat, all_generated)):
        ax.set_facecolor('black')
        idx = np.random.choice(len(gen), size=min(20_000, len(gen)), replace=False)
        g   = gen[idx]
        ax.scatter(g[:, 0], g[:, 1], c=g[:, 2],
                   cmap='coolwarm', s=0.2, alpha=0.5,
                   vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(f'Run {run_idx+1}', color='white', fontsize=9)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    for ax in axes.flat[len(all_generated):]:
        ax.set_visible(False)

    fig.suptitle(
        f'{base_name.replace("_", " ")}  —  All {N_RUNS} NFlow runs\n'
        'Colour = Line-of-Sight Velocity Vz [km/s]',
        color='white', fontsize=13, y=1.01)

    out_png = f'{base_name}_cv{N_RUNS}_grid.png'
    plt.savefig(out_png, dpi=120, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
for input_filename in DATA_FILES:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")

    npz_file = f'{base_name}_cv{N_RUNS}.npz'
    if not os.path.exists(npz_file):
        print(f"  [Skip] {npz_file} not found — job may still be running")
        continue

    df       = pd.read_csv(input_filename, sep=r'\s+', comment='#',
                           names=['X', 'Y', 'V_Z'])
    obs      = df[['X', 'Y', 'V_Z']].values.astype(np.float32)
    mean_obs = obs.mean(0); std_obs = obs.std(0)
    obs_norm = (obs - mean_obs) / std_obs

    all_generated = []

    for run in range(1, N_RUNS + 1):
        weights_path = f'{base_name}_run{run:02d}.weights'
        if not os.path.exists(weights_path + '.index'):
            print(f"  [Skip run {run}] weights not found: {weights_path}")
            continue

        print(f"  Loading run {run:02d}...")
        model = NFlow(n_layers=N_LAYERS, hidden=HIDDEN)
        _     = model.log_prob(tf.constant(obs_norm[:10]))
        ckpt  = tf.train.Checkpoint(model=model)
        ckpt.restore(weights_path).expect_partial()

        samples_norm = model.sample(N_SAMPLE)
        samples      = samples_norm * std_obs + mean_obs
        all_generated.append(samples)

    if not all_generated:
        print(f"  [Skip] No weights found for {base_name}")
        continue

    print(f"  Generating plots ({len(all_generated)} runs loaded)...")
    make_scatter_plot(base_name, obs, all_generated, mean_obs, std_obs)
    make_grid_plot(base_name, obs, all_generated, mean_obs, std_obs)
    make_profile_plot(base_name, npz_file, input_filename)

print(f"\n{'='*60}")
print("Done! Three plots per data file:")
print("  *_cv20_scatter.png     — side-by-side real vs all 20 runs combined")
print("  *_cv20_grid.png        — 4x5 grid of all 20 individual runs")
print("  *_cv20_dm_profiles.png — rho(r) and M(r) with analytic true curves")
print(f"{'='*60}")