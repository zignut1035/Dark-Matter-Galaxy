"""
find_best_run_clean.py  —  Find the best NFlow run and produce a clean
                            comparison plot: estimated profiles vs true data.
                            No grey bands, no uncertainty shading — just the
                            best run against the analytic truth.

Usage:
    python3 find_best_run_clean.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob, os
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS   = 20
N_SAMPLE = 100_000
N_LAYERS = 12
HIDDEN   = 256
G_CONST  = 4.30091e-6

DATA_FILES = sorted(glob.glob('3D_data/observables_*.dat'))
if not DATA_FILES:
    raise FileNotFoundError("No data files found in 3D_data/")

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
    if r <= 0: return 0.0
    val, _ = quad(lambda rp: 4*np.pi*rp**2*rho_true(rp, **params), 0, r,
                  limit=200, epsabs=1e-6, epsrel=1e-6)
    return val

def get_dm_params(base_name):
    for key in sorted(DM_PARAMS.keys(), key=len, reverse=True):
        if key in base_name:
            return DM_PARAMS[key]
    return None

# ── Model ─────────────────────────────────────────────────────────────────────
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
    def __init__(self, n_layers=12, hidden=256, **kwargs):
        super().__init__(**kwargs)
        self.coupling_layers = [CouplingLayer(hidden) for _ in range(n_layers)]
        self.perms     = [tf.constant([1,2,0] if i%2==0 else [2,0,1]) for i in range(n_layers)]
        self.inv_perms = [tf.argsort(p) for p in self.perms]

    def log_prob(self, x):
        log_det = tf.zeros(tf.shape(x)[0])
        for layer, perm in zip(self.coupling_layers, self.perms):
            x = tf.gather(x, perm, axis=1)
            x, ld = layer(x, reverse=False)
            log_det += ld
        log_base = -0.5 * tf.reduce_sum(
            x**2 + tf.cast(tf.math.log(2*np.pi), tf.float32), axis=1)
        return log_base + log_det

    def sample(self, n):
        z = tf.random.normal((n, 3))
        for layer, inv_perm in zip(reversed(self.coupling_layers), reversed(self.inv_perms)):
            z = layer(z, reverse=True)
            z = tf.gather(z, inv_perm, axis=1)
        return z.numpy()

    def call(self, x):
        return self.log_prob(x)

# ── Ground truth loader ───────────────────────────────────────────────────────
def load_ground_truth(input_filename):
    gt_filename = (input_filename
                   .replace('3D_data', '6D_data')
                   .replace('observables_', 'Mock_isotropic_'))
    try:
        df_6d = pd.read_csv(gt_filename, sep=r'\s+', comment='#',
                            names=['X','Y','Z','VX','VY','VZ'])
    except FileNotFoundError:
        return None

    r_6d  = np.sqrt(df_6d['X']**2 + df_6d['Y']**2 + df_6d['Z']**2)
    vr_6d = ((df_6d['X']*df_6d['VX'] + df_6d['Y']*df_6d['VY']
              + df_6d['Z']*df_6d['VZ']) / r_6d)

    gt_bins  = np.logspace(np.log10(r_6d.min()*1.01), np.log10(r_6d.max()*0.99), 30)
    gt_r_mid = np.sqrt(gt_bins[:-1] * gt_bins[1:])
    counts   = np.histogram(r_6d, bins=gt_bins)[0]
    vol      = (4/3)*np.pi*(gt_bins[1:]**3 - gt_bins[:-1]**3)
    n_true   = counts / vol

    sigma2_true = []
    for k in range(len(gt_bins)-1):
        m = (r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])
        sigma2_true.append(vr_6d[m].var() if m.sum() > 1 else np.nan)

    return gt_r_mid, n_true, np.array(sigma2_true)

# ── Score run ─────────────────────────────────────────────────────────────────
def score_run(r_grid, n_r, sigma_r2, gt):
    if gt is None: return np.inf
    gt_r, n_true, sigma2_true = gt
    valid_n = np.isfinite(n_true) & (n_true > 0)
    valid_s = np.isfinite(sigma2_true) & (sigma2_true > 0)
    if valid_n.sum() < 3 or valid_s.sum() < 3: return np.inf
    n_interp = np.interp(r_grid, gt_r[valid_n], n_true[valid_n], left=np.nan, right=np.nan)
    s_interp = np.interp(r_grid, gt_r[valid_s], sigma2_true[valid_s], left=np.nan, right=np.nan)
    mask_n = np.isfinite(n_interp) & (n_r > 0) & (n_interp > 0)
    mask_s = np.isfinite(s_interp) & (sigma_r2 > 0) & (s_interp > 0)
    mse_n = np.mean((np.log10(n_r[mask_n]) - np.log10(n_interp[mask_n]))**2) if mask_n.sum() > 0 else np.inf
    mse_s = np.mean((np.log10(sigma_r2[mask_s]) - np.log10(s_interp[mask_s]))**2) if mask_s.sum() > 0 else np.inf
    return (mse_n + mse_s) / 2.0

# ── MAIN CLEAN PLOT ───────────────────────────────────────────────────────────
def plot_clean_comparison(base_name, r_grid,
                           best_n_r, best_sigma_b0, best_sigma_bm5,
                           best_M_r, best_rho_r,
                           gt, dm_params, best_run_idx, best_mse):
    """
    4-panel clean comparison plot:
      n(r) | σ²_r(r) | ρ(r) | M(r)
    Each panel: best-run estimate vs true analytic curve (and 6D data points).
    No grey low-stats shading. No uncertainty bands. Clean and readable.
    """

    # True DM profiles on the same r_grid
    if dm_params is not None:
        rho_arr = np.array([rho_true(r, **dm_params) for r in r_grid])
        M_arr   = np.array([M_true(r,  **dm_params) for r in r_grid])
    else:
        rho_arr = M_arr = None

    fig = plt.figure(figsize=(20, 5))
    fig.patch.set_facecolor('white')
    outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)

    panels = [
        # (title, ylabel, estimated_b0, estimated_bm5, true_r, true_vals, data_r, data_vals)
        (
            'Stellar number density',
            r'$n(r)$ [kpc$^{-3}$]',
            best_n_r, best_n_r,
            gt[0] if gt else None, gt[1] if gt else None,
            gt[0] if gt else None, gt[1] if gt else None,
        ),
        (
            'Radial velocity dispersion',
            r'$\sigma_r^2$ [(km s$^{-1}$)$^2$]',
            best_sigma_b0, best_sigma_bm5,
            gt[0] if gt else None, gt[2] if gt else None,
            gt[0] if gt else None, gt[2] if gt else None,
        ),
        (
            'DM mass density',
            r'$\rho(r)$ [M$_\odot$ kpc$^{-3}$]',
            best_rho_r, best_rho_r,
            r_grid, rho_arr,
            None, None,
        ),
        (
            'Enclosed mass',
            r'$M(r)$ [M$_\odot$]',
            best_M_r, best_M_r,
            r_grid, M_arr,
            None, None,
        ),
    ]

    for col, (title, ylabel, est_b0, est_bm5, true_r, true_vals, data_r, data_vals) in enumerate(panels):
        ax = fig.add_subplot(outer[col])
        ax.set_facecolor('white')

        # ── Estimated: β=0 ───────────────────────────────────────────────────
        valid_b0 = np.isfinite(est_b0) & (est_b0 > 0)
        if valid_b0.any():
            ax.loglog(r_grid[valid_b0], est_b0[valid_b0],
                      color='#1a7abf', lw=2.5, ls='-', zorder=3,
                      label=r'Estimated ($\beta=0$)')

        # ── Estimated: β=−0.5 (only for n and σ) ────────────────────────────
        if col == 1:  # only velocity dispersion panel shows both
            valid_bm5 = np.isfinite(est_bm5) & (est_bm5 > 0)
            if valid_bm5.any():
                ax.loglog(r_grid[valid_bm5], est_bm5[valid_bm5],
                          color='#1a7abf', lw=2.0, ls='--', zorder=3,
                          label=r'Estimated ($\beta=-0.5$)')

        # ── True analytic curve ───────────────────────────────────────────────
        if true_r is not None and true_vals is not None:
            valid_t = np.isfinite(true_vals) & (np.array(true_vals) > 0)
            if valid_t.any():
                ax.loglog(np.array(true_r)[valid_t], np.array(true_vals)[valid_t],
                          color='#d62728', lw=2.5, ls='-', zorder=4,
                          label='True (analytic)')

        # ── 6D data points (n and σ panels only) ────────────────────────────
        if col <= 1 and data_r is not None and data_vals is not None:
            valid_d = np.isfinite(data_vals) & (np.array(data_vals) > 0)
            if valid_d.any():
                ax.scatter(np.array(data_r)[valid_d], np.array(data_vals)[valid_d],
                           color='#555555', s=18, zorder=5, label='True (6D data)',
                           marker='o', linewidths=0)

        ax.set_xlabel('r [kpc]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, which='both', alpha=0.25, color='grey', linewidth=0.5)
        ax.tick_params(labelsize=10)

        # Clean spine style
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#444444')

    profile_type = 'Core' if 'core' in base_name else 'Cusp'
    dataset_num  = ''.join(filter(str.isdigit, base_name))
    fig.suptitle(
        f'JFlow Results: {profile_type} DM Profile (Dataset {dataset_num})  '
        f'—  Best run #{best_run_idx+1} of {N_RUNS}  |  MSE = {best_mse:.4f}',
        fontsize=13, fontweight='bold', y=1.03
    )

    out_png = f'{base_name}_clean_comparison.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved -> {out_png}")


def plot_clean_scatter(base_name, obs, samples, best_run_idx, best_mse):
    """Side-by-side scatter: real projected data vs best NFlow run."""
    vmin = np.percentile(obs[:, 2], 2)
    vmax = np.percentile(obs[:, 2], 98)
    lim  = np.percentile(np.sqrt(obs[:,0]**2 + obs[:,1]**2), 99) * 1.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
    fig.subplots_adjust(wspace=0.06)

    # ── REAL DATA (Left Panel) ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('black')
    
    # ADDED 'extent' TO FIX THE GIANT HEXAGON BUG
    sc = ax.hexbin(obs[:,0], obs[:,1], C=obs[:,2],
                   gridsize=100, extent=[-lim, lim, -lim, lim], 
                   cmap='coolwarm', reduce_C_function=np.mean,
                   vmin=vmin, vmax=vmax, mincnt=1, rasterized=True)
    
    ax.set_title(f'Real Data (Projected)\n{len(obs):,} stars',
                 color='white', fontsize=12)

    # ── GENERATED DATA (Right Panel) ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('black')
    
    # ADDED 'extent' TO FIX THE GIANT HEXAGON BUG
    ax2.hexbin(samples[:,0], samples[:,1], C=samples[:,2],
               gridsize=100, extent=[-lim, lim, -lim, lim], 
               cmap='coolwarm', reduce_C_function=np.mean,
               vmin=vmin, vmax=vmax, mincnt=1, rasterized=True)
    
    ax2.set_title(
        f'Generated by Normalizing Flow\nBest run #{best_run_idx+1}  —  {len(samples):,} stars',
        color='white', fontsize=12)

    # ── Formatting ────────────────────────────────────────────────────────────
    for a in axes:
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_aspect('equal')
        a.set_xlabel('X [kpc]', color='white', fontsize=11)
        a.set_ylabel('Y [kpc]', color='white', fontsize=11)
        a.tick_params(colors='white')
        for spine in a.spines.values():
            spine.set_edgecolor('white')

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Mean Line-of-Sight Velocity Vz [km/s]', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    profile_type = 'Core' if 'core' in base_name else 'Cusp'
    dataset_num  = ''.join(filter(str.isdigit, base_name))
    plt.suptitle(
        f'JFlow: {profile_type} DM Profile (Dataset {dataset_num})  —  Spatial Distribution',
        color='white', fontsize=13, y=1.02)

    out_png = f'{base_name}_clean_scatter.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
summary = []

for input_filename in DATA_FILES:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")

    npz_file = f'{base_name}_cv{N_RUNS}.npz'
    if not os.path.exists(npz_file):
        print(f"  [Skip] {npz_file} not found"); continue

    npz           = np.load(npz_file, allow_pickle=True)
    r_grid        = npz['r_grid']
    all_n_r       = npz['n_r_all']
    all_sigma_b0  = npz['sigma_b0_all']
    all_sigma_bm5 = npz['sigma_bm5_all']
    all_M_r       = list(npz['M_r_all'])
    all_rho_r     = list(npz['rho_r_all'])
    mean_obs      = npz['mean_obs']
    std_obs       = npz['std_obs']

    gt        = load_ground_truth(input_filename)
    dm_params = get_dm_params(base_name)

    # Score every run
    all_scores = np.array([score_run(r_grid, all_n_r[i], all_sigma_b0[i], gt)
                           for i in range(N_RUNS)])
    best_run_idx = int(np.argmin(all_scores))
    best_mse     = all_scores[best_run_idx]

    print(f"  >> BEST run: #{best_run_idx+1}  (MSE = {best_mse:.4f})")
    print(f"  >> Mean MSE: {all_scores.mean():.4f} ± {all_scores.std():.4f}")

    summary.append({
        'file': base_name, 'best_run': best_run_idx+1,
        'best_mse': best_mse, 'mean_mse': all_scores.mean(), 'std_mse': all_scores.std()
    })

    # Save ranking CSV
    pd.DataFrame({
        'run': np.arange(1, N_RUNS+1),
        'mse': all_scores,
        'rank': np.argsort(np.argsort(all_scores)) + 1
    }).sort_values('rank').to_csv(f'{base_name}_run_ranking.csv', index=False)

    # Clean 4-panel profile comparison
    plot_clean_comparison(
        base_name, r_grid,
        all_n_r[best_run_idx],
        all_sigma_b0[best_run_idx],
        all_sigma_bm5[best_run_idx],
        np.array(all_M_r[best_run_idx], dtype=float),
        np.array(all_rho_r[best_run_idx], dtype=float),
        gt, dm_params, best_run_idx, best_mse
    )

    # Load weights and make scatter plot
    df       = pd.read_csv(input_filename, sep=r'\s+', comment='#', names=['X','Y','V_Z'])
    obs      = df[['X','Y','V_Z']].values.astype(np.float32)
    obs_norm = (obs - mean_obs) / std_obs

    weights_path = f'{base_name}_run{best_run_idx+1:02d}.weights'
    if not os.path.exists(weights_path + '.index'):
        print(f"  [Skip scatter] Weights not found: {weights_path}"); continue

    model = NFlow(n_layers=N_LAYERS, hidden=HIDDEN)
    _ = model.log_prob(tf.constant(obs_norm[:10]))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(weights_path).expect_partial()

    samples_norm = model.sample(N_SAMPLE)
    samples      = samples_norm * std_obs + mean_obs

    R_real  = np.sqrt(obs[:,0]**2 + obs[:,1]**2)
    R_gen   = np.sqrt(samples[:,0]**2 + samples[:,1]**2)
    samples = samples[R_gen < np.percentile(R_real, 99) * 1.5]

    plot_clean_scatter(base_name, obs, samples, best_run_idx, best_mse)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'File':<40} {'Best Run':>8} {'Best MSE':>10} {'Mean MSE':>10}")
print("-"*60)
for s in summary:
    print(f"{s['file']:<40} {s['best_run']:>8} {s['best_mse']:>10.4f} {s['mean_mse']:>10.4f}")
print(f"{'='*60}")
print("\nOutput files per dataset:")
print("  *_clean_comparison.png  — 4-panel profile plot (no grey bands)")
print("  *_clean_scatter.png     — spatial distribution comparison")
print("  *_run_ranking.csv       — MSE ranking of all 20 runs")