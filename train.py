import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
import glob

# ── All 6 input files ─────────────────────────────────────────────────────────
DATA_FILES = sorted(glob.glob('3D_data/observables_*.dat'))

if not DATA_FILES:
    raise FileNotFoundError("No data files found in 3D_data/. Check your working directory.")

print(f"Found {len(DATA_FILES)} data files:")
for f in DATA_FILES:
    print(f"  {f}")


# ── 1. Coupling layer ─────────────────────────────────────────────────────────
class CouplingLayer(keras.layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.net_s = keras.Sequential([
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(2)
        ])
        self.net_t = keras.Sequential([
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(hidden, activation='tanh'),
            keras.layers.Dense(2)
        ])

    def call(self, x, reverse=False):
        x0, x1 = x[:, :1], x[:, 1:]
        s = self.net_s(x0)
        s = tf.clip_by_value(s, -5.0, 5.0)
        t = self.net_t(x0)

        if not reverse:
            y1      = x1 * tf.exp(s) + t
            log_det = tf.reduce_sum(s, axis=1)
            return tf.concat([x0, y1], axis=1), log_det
        else:
            y1 = (x1 - t) * tf.exp(-s)
            return tf.concat([x0, y1], axis=1)


# ── 2. Normalizing Flow ───────────────────────────────────────────────────────
class NFlow(keras.Model):
    def __init__(self, n_layers=8, hidden=128, **kwargs):
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


# ── 3. Training function (defined per-model inside the loop) ─────────────────
# train_step is defined inside the main loop to avoid @tf.function graph
# caching issues when multiple models are trained sequentially.


# ── 4. Profile computation helpers ───────────────────────────────────────────
def compute_profiles(samples, mean, std):
    """Denormalise samples, compute Σ(R), σ_los(R), n(r), σ_r²(r), M(r)."""
    s    = samples * std + mean
    X_s  = s[:, 0];  Y_s = s[:, 1];  VZ_s = s[:, 2]
    R_s  = np.sqrt(X_s**2 + Y_s**2)

    # Clip extreme NF tails (central 98 %)
    lo, hi = np.percentile(R_s, 1), np.percentile(R_s, 99)
    mask = (R_s > lo) & (R_s < hi)
    R_s  = R_s[mask];  VZ_s = VZ_s[mask]

    # Radial bins (log-spaced)
    R_bins = np.logspace(np.log10(R_s.min()*1.01),
                         np.log10(R_s.max()*0.99), 25)
    R_mid  = 0.5 * (R_bins[:-1] + R_bins[1:])

    # Σ(R) – projected surface density
    counts, _ = np.histogram(R_s, bins=R_bins)
    area       = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)
    Sigma      = counts / area

    # σ_los²(R)
    sigma_los2 = np.array([
        VZ_s[(R_s >= R_bins[k]) & (R_s < R_bins[k+1])].var()
        for k in range(len(R_bins) - 1)
    ])

    # ── Abel inversion: Σ(R) → n(r) ──────────────────────────────────────────
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

    # ── Jeans inversion: σ_los + n(r) → σ_r²(r) ─────────────────────────────
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
        integrand = lambda R: (dSS_dR(R) * (1 - beta*(r/R)**2)
                               / np.sqrt(R**2 - r**2))
        val, _ = quad(integrand, r * 1.001, R_max)
        nr = np.exp(n_spl(np.log(r)))
        return -val / (np.pi * nr) if nr > 0 else 0.0

    sigma_r2_b0  = np.array([jeans_sigma_r2(r, beta=0.0)  for r in r_grid])
    sigma_r2_bm5 = np.array([jeans_sigma_r2(r, beta=-0.5) for r in r_grid])

    # ── Dark matter enclosed mass M(r) ────────────────────────────────────────
    G = 4.30091e-6   # (km/s)² · kpc / M☉
    dln_n_dln_r       = n_spl.derivative()(np.log(r_grid))
    log_sig2_spl_b0   = UnivariateSpline(np.log(r_grid),
                                         np.log(sigma_r2_b0 + 1e-30), s=1)
    dln_sig2_dln_r_b0 = log_sig2_spl_b0.derivative()(np.log(r_grid))
    M_r_b0 = -(r_grid * sigma_r2_b0 / G) * (dln_n_dln_r + dln_sig2_dln_r_b0)

    return r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_r_b0


# ── 5. Ground-truth loader ────────────────────────────────────────────────────
def load_ground_truth(input_filename):
    gt_filename = (input_filename
                   .replace('3D_data', '6D_data')
                   .replace('observables_', 'Mock_isotropic_'))
    try:
        df_6d = pd.read_csv(gt_filename, sep=r'\s+', comment='#',
                            names=['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
    except FileNotFoundError:
        print(f"  [Warning] Ground truth not found: {gt_filename}")
        return None

    r_6d = np.sqrt(df_6d['X']**2 + df_6d['Y']**2 + df_6d['Z']**2)
    vr_6d = ((df_6d['X']*df_6d['VX'] + df_6d['Y']*df_6d['VY']
              + df_6d['Z']*df_6d['VZ']) / r_6d)

    gt_bins  = np.logspace(np.log10(r_6d.min()*1.01),
                           np.log10(r_6d.max()*0.99), 30)
    gt_r_mid = np.sqrt(gt_bins[:-1] * gt_bins[1:])

    counts_6d, _ = np.histogram(r_6d, bins=gt_bins)
    vol_6d        = (4/3) * np.pi * (gt_bins[1:]**3 - gt_bins[:-1]**3)
    n_true        = counts_6d / vol_6d

    sigma2_true = []
    for k in range(len(gt_bins) - 1):
        m = (r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])
        sigma2_true.append(vr_6d[m].var() if m.sum() > 1 else np.nan)
    sigma2_true = np.array(sigma2_true)

    return gt_r_mid, n_true, sigma2_true


# ── 6. Plotting function ──────────────────────────────────────────────────────
def make_plot(base_name, r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_r_b0, gt):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if gt is not None:
        gt_r_mid, n_true, sigma2_true = gt
        axes[0].loglog(gt_r_mid, n_true, 'k-', lw=2, alpha=0.7,
                       label='True 6D Physics')
        axes[1].semilogx(gt_r_mid, np.log(sigma2_true), 'k-', lw=2, alpha=0.7,
                         label='True 6D Physics')

    # Density
    axes[0].loglog(r_grid, n_r, 'g--', lw=2, label='Estimated, NFlow')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel('n(r) [kpc⁻³]')
    axes[0].set_title('Stellar number density')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Velocity dispersion
    axes[1].semilogx(r_grid, np.log(sigma_r2_b0),  'g--', lw=2,
                     label=r'Estimated $\beta=0$')
    axes[1].semilogx(r_grid, np.log(sigma_r2_bm5), 'b:',  lw=2,
                     label=r'Estimated $\beta=-0.5$')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'log $\sigma_r^2$ [(km/s)²]')
    axes[1].set_title('Radial velocity dispersion')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Enclosed mass
    axes[2].loglog(r_grid, M_r_b0, 'g--', lw=2,
                   label=r'Estimated $M(r)$ ($\beta=0$)')
    axes[2].set_xlabel('r [kpc]')
    axes[2].set_ylabel(r'Mass M(r) [$M_\odot$]')
    axes[2].set_title('Dark Matter Enclosed Mass')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f'DM mass profile from projected observables only\n({base_name})',
        fontsize=11)
    plt.tight_layout()
    out_png = f'{base_name}_profiles.png'
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Saved → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop over all 6 data files
# ══════════════════════════════════════════════════════════════════════════════
for input_filename in DATA_FILES:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {input_filename}  ({base_name})")
    print(f"{'='*60}")

    # ── Load & normalise ──────────────────────────────────────────────────────
    df = pd.read_csv(input_filename, sep=r'\s+', comment='#',
                     names=['X', 'Y', 'V_Z'])
    obs      = df[['X', 'Y', 'V_Z']].values.astype(np.float32)
    mean_obs = obs.mean(0)
    std_obs  = obs.std(0)
    obs_norm = (obs - mean_obs) / std_obs

    print(f"  Stars loaded: {len(obs)}")
    print(f"  X range:   {obs[:,0].min():.4f} to {obs[:,0].max():.4f} kpc")
    print(f"  Y range:   {obs[:,1].min():.4f} to {obs[:,1].max():.4f} kpc")
    print(f"  V_Z range: {obs[:,2].min():.4f} to {obs[:,2].max():.4f} km/s")

    dataset = (tf.data.Dataset
               .from_tensor_slices(obs_norm)
               .shuffle(len(obs_norm))
               .batch(256)
               .prefetch(1))

    # ── Build & train one model per file ─────────────────────────────────────
    model     = NFlow(n_layers=12, hidden=256)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(model.log_prob(x))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print("  Training NF...")
    best_nll     = np.inf
    patience     = 0
    max_patience = 5

    for epoch in range(100):
        for x in dataset:
            train_step(x)

        if epoch % 20 == 0:
            nll = -tf.reduce_mean(model.log_prob(
                      tf.constant(obs_norm))).numpy()
            current_lr = float(optimizer.learning_rate)
            print(f"    Epoch {epoch:3d}  NLL = {nll:.4f}  LR = {current_lr:.2e}")

            if nll < best_nll - 1e-3:
                best_nll = nll
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    new_lr = max(current_lr * 0.5, 1e-5)
                    optimizer.learning_rate.assign(new_lr)
                    print(f"    → No improvement, reducing LR to {new_lr:.2e}")
                    patience = 0

    print("  Training done.")

    # Force full build with real data, then save via Checkpoint
    _ = model.log_prob(tf.constant(obs_norm[:10]))
    weights_path = f'{base_name}.weights'
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.write(weights_path)
    print(f"  Weights saved → {weights_path}")

    # ── Sample & compute profiles ─────────────────────────────────────────────
    print("  Sampling 100,000 points...")
    samples_norm = model.sample(100_000)
    r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_r_b0 = compute_profiles(
        samples_norm, mean_obs, std_obs)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n  {'r [kpc]':>10} | {'n(r) [kpc-3]':>14} | "
          f"{'sigma_r2 b=0':>14} | {'Mass b=0 [M_sun]':>18}")
    print("  " + "-"*63)
    for i in range(len(r_grid)):
        print(f"  {r_grid[i]:>10.4f} | {n_r[i]:>14.4e} | "
              f"{sigma_r2_b0[i]:>14.4f} | {M_r_b0[i]:>18.4e}")

    # ── Load ground truth & plot ──────────────────────────────────────────────
    gt = load_ground_truth(input_filename)
    make_plot(base_name, r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, M_r_b0, gt)

print(f"\n{'='*60}")
print("All done! One PNG saved per data file.")
print(f"{'='*60}")