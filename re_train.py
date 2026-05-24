import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
import glob

# ── All input files ───────────────────────────────────────────────────────────
DATA_FILES = sorted(glob.glob('3D_data/observables_*.dat'))[5:]
if not DATA_FILES:
    raise FileNotFoundError("No data files found in 3D_data/")

print(f"Found {len(DATA_FILES)} data files:")
for f in DATA_FILES:
    print(f"  {f}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Coupling Layer  (swish, zero-init, 3 hidden layers)
# ══════════════════════════════════════════════════════════════════════════════
class CouplingLayer(keras.layers.Layer):
    def __init__(self, hidden=256, **kwargs):
        super().__init__(**kwargs)

        def _make_net(out_dim):
            return keras.Sequential([
                keras.layers.Dense(hidden, activation='swish'),
                keras.layers.Dense(hidden, activation='swish'),
                keras.layers.Dense(hidden, activation='swish'),
                keras.layers.Dense(out_dim,
                                   kernel_initializer='zeros',
                                   bias_initializer='zeros'),
            ])

        self.net_s = _make_net(2)
        self.net_t = _make_net(2)

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


# ══════════════════════════════════════════════════════════════════════════════
# 2. Normalizing Flow
# ══════════════════════════════════════════════════════════════════════════════
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
# 3. Azimuthal augmentation
# ══════════════════════════════════════════════════════════════════════════════
@tf.function
def augment_rotate(x):
    angle = tf.random.uniform((), 0.0, 2.0 * np.pi)
    c, s  = tf.cos(angle), tf.sin(angle)
    x_rot = x[:, 0:1] * c - x[:, 1:2] * s
    y_rot = x[:, 0:1] * s + x[:, 1:2] * c
    return tf.concat([x_rot, y_rot, x[:, 2:3]], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Cosine annealing LR
# ══════════════════════════════════════════════════════════════════════════════
def cosine_lr(epoch, n_epochs, lr_max=3e-4, lr_min=5e-6):
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * epoch / n_epochs))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Profile computation
# ══════════════════════════════════════════════════════════════════════════════
MIN_STARS = 50

def compute_profiles(samples, mean, std, n_bins=50):
    s = samples * std + mean
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
        raise ValueError(f"Only {good.sum()} good bins — increase N_SAMPLE.")

    Rm     = R_mid[good]
    Sig    = Sigma[good]
    sl2    = sigma_los2[good]
    n_good = good.sum()

    s_profile = float(n_good) * 4.0
    s_light   = float(n_good) * 0.8

    # ── Abel inversion ────────────────────────────────────────────────────────
    log_Sig_spl = UnivariateSpline(np.log(Rm), np.log(Sig), s=s_profile, k=3)

    def dSigma_dR(R):
        lnR = np.log(R)
        return np.exp(log_Sig_spl(lnR)) / R * float(log_Sig_spl.derivative()(lnR))

    R_max_abel = Rm[-1]

    def abel_density(r):
        if r >= R_max_abel:
            return 1e-30
        val, _ = quad(lambda R: dSigma_dR(R) / np.sqrt(R**2 - r**2),
                      r * 1.001, R_max_abel, limit=150)
        return max(-val / np.pi, 1e-30)

    skip    = max(3, int(0.15 * n_good))
    r_inner = Rm[skip]
    r_outer = Rm[-skip - 1]
    if r_inner >= r_outer:
        r_inner = Rm[2]
        r_outer = Rm[-3]

    r_grid = np.logspace(np.log10(r_inner), np.log10(r_outer), 35)
    n_r    = np.clip(np.array([abel_density(r) for r in r_grid]), 1e-30, None)
    n_spl  = UnivariateSpline(np.log(r_grid), np.log(n_r), s=s_light, k=3)

    # ── Jeans inversion ───────────────────────────────────────────────────────
    SS    = Sig * sl2
    good2 = SS > 0
    log_SS_spl = UnivariateSpline(
        np.log(Rm[good2]), np.log(SS[good2]), s=s_profile, k=3)

    def dSS_dR(R):
        lnR = np.log(R)
        return np.exp(log_SS_spl(lnR)) / R * float(log_SS_spl.derivative()(lnR))

    def jeans_sigma_r2(r, beta=0.0):
        if r >= R_max_abel:
            return 1e-30
        try:
            val, _ = quad(
                lambda R: dSS_dR(R) * (1.0 - beta * (r / R)**2)
                          / np.sqrt(R**2 - r**2),
                r * 1.001, R_max_abel, limit=150)
        except Exception:
            return 1e-30
        nr = float(np.exp(n_spl(np.log(r))))
        return max(-val / (np.pi * nr), 1e-30) if nr > 1e-30 else 1e-30

    sigma_r2_b0  = np.array([jeans_sigma_r2(r, 0.0)  for r in r_grid])
    sigma_r2_bm5 = np.array([jeans_sigma_r2(r, -0.5) for r in r_grid])

    return r_grid, n_r, sigma_r2_b0, sigma_r2_bm5


# ══════════════════════════════════════════════════════════════════════════════
# 6. Ground-truth loader
# ══════════════════════════════════════════════════════════════════════════════
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

    r_6d  = np.sqrt(df_6d['X']**2 + df_6d['Y']**2 + df_6d['Z']**2)
    vr_6d = ((df_6d['X']*df_6d['VX'] + df_6d['Y']*df_6d['VY']
              + df_6d['Z']*df_6d['VZ']) / r_6d)

    gt_bins  = np.logspace(np.log10(r_6d.min()*1.01),
                           np.log10(r_6d.max()*0.99), 40)
    gt_r_mid = np.sqrt(gt_bins[:-1] * gt_bins[1:])

    counts_6d, _ = np.histogram(r_6d, bins=gt_bins)
    vol_6d        = (4/3) * np.pi * (gt_bins[1:]**3 - gt_bins[:-1]**3)
    n_true        = counts_6d / vol_6d

    sigma2_true = []
    for k in range(len(gt_bins) - 1):
        m = (r_6d >= gt_bins[k]) & (r_6d < gt_bins[k+1])
        sigma2_true.append(vr_6d[m].var() if m.sum() > 1 else np.nan)

    return gt_r_mid, n_true, np.array(sigma2_true)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plotting — 2 panels only (density + velocity dispersion)
# ══════════════════════════════════════════════════════════════════════════════
def make_plot(base_name, r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, gt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if gt is not None:
        gt_r_mid, n_true, sigma2_true = gt
        axes[0].loglog(gt_r_mid, n_true, 'k-', lw=2, alpha=0.7,
                       label='True 6D Physics')
        valid = sigma2_true > 0
        axes[1].semilogx(gt_r_mid[valid], np.log(sigma2_true[valid]),
                         'k-', lw=2, alpha=0.7, label='True 6D Physics')

    axes[0].loglog(r_grid, n_r, 'g--', lw=2, label='Estimated, NFlow')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel('n(r) [kpc⁻³]')
    axes[0].set_title('Stellar number density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    v0 = sigma_r2_b0  > 1e-20
    v5 = sigma_r2_bm5 > 1e-20
    if v0.any():
        axes[1].semilogx(r_grid[v0], np.log(sigma_r2_b0[v0]),
                         'g--', lw=2, label=r'Estimated $\beta=0$')
    if v5.any():
        axes[1].semilogx(r_grid[v5], np.log(sigma_r2_bm5[v5]),
                         'b:', lw=2, label=r'Estimated $\beta=-0.5$')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'log $\sigma_r^2$ [(km/s)²]')
    axes[1].set_title('Radial velocity dispersion')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f'NFlow reconstruction from projected observables\n({base_name})',
        fontsize=11)
    plt.tight_layout()
    out_png = f'{base_name}_v2_profiles.png'
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Saved → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
N_EPOCHS  = 150
BATCH     = 256
N_SAMPLE  = 300_000

for input_filename in DATA_FILES:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {input_filename}  ({base_name})")
    print(f"{'='*60}")

    df       = pd.read_csv(input_filename, sep=r'\s+', comment='#',
                           names=['X', 'Y', 'V_Z'])
    obs      = df[['X', 'Y', 'V_Z']].values.astype(np.float32)
    mean_obs = obs.mean(0)
    std_obs  = obs.std(0)
    obs_norm = (obs - mean_obs) / std_obs

    print(f"  Stars loaded : {len(obs)}")

    dataset = (tf.data.Dataset
               .from_tensor_slices(obs_norm)
               .shuffle(len(obs_norm), reshuffle_each_iteration=True)
               .batch(BATCH)
               .map(augment_rotate, num_parallel_calls=tf.data.AUTOTUNE)
               .prefetch(tf.data.AUTOTUNE))

    model     = NFlow(n_layers=12, hidden=256)
    optimizer = keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(model.log_prob(x))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print(f"  Training NF for {N_EPOCHS} epochs...")
    best_nll     = np.inf
    best_weights = None
    obs_tf       = tf.constant(obs_norm)

    for epoch in range(N_EPOCHS):
        optimizer.learning_rate.assign(cosine_lr(epoch, N_EPOCHS))
        for x_batch in dataset:
            train_step(x_batch)

        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            nll = float(-tf.reduce_mean(model.log_prob(obs_tf)).numpy())
            tag = ''
            if nll < best_nll:
                best_nll     = nll
                best_weights = model.get_weights()
                tag = ' ← best'
            print(f"    Epoch {epoch:3d}  NLL = {nll:.4f}  "
                  f"LR = {cosine_lr(epoch, N_EPOCHS):.2e}{tag}")

    model.set_weights(best_weights)
    print(f"  Best NLL = {best_nll:.4f}  (weights restored)")

    _ = model.log_prob(tf.constant(obs_norm[:10]))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.write(f'{base_name}_v2.weights')
    print(f"  Weights saved → {base_name}_v2.weights")

    print(f"  Sampling {N_SAMPLE:,} points...")
    samples_norm = model.sample(N_SAMPLE)

    try:
        r_grid, n_r, sigma_r2_b0, sigma_r2_bm5 = compute_profiles(
            samples_norm, mean_obs, std_obs)
    except Exception as e:
        print(f"  [Warning] Profile computation failed: {e}")
        continue

    print(f"\n  {'r [kpc]':>10} | {'n(r) [kpc-3]':>14} | "
          f"{'sigma_r2 b=0':>14} | {'sigma_r2 b=-0.5':>16}")
    print("  " + "-"*61)
    for i in range(len(r_grid)):
        print(f"  {r_grid[i]:>10.4f} | {n_r[i]:>14.4e} | "
              f"{sigma_r2_b0[i]:>14.4f} | {sigma_r2_bm5[i]:>16.4f}")

    gt = load_ground_truth(input_filename)
    make_plot(base_name, r_grid, n_r, sigma_r2_b0, sigma_r2_bm5, gt)

print(f"\n{'='*60}")
print("All done!  One PNG (2-panel) saved per data file.")
print(f"{'='*60}")