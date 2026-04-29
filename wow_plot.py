import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import glob

# ── 1. Model architecture (must match train.py) ───────────────────────────────
class CouplingLayer(keras.layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.net_s = keras.Sequential([keras.layers.Dense(hidden, activation='tanh'),
                                       keras.layers.Dense(hidden, activation='tanh'),
                                       keras.layers.Dense(2)])
        self.net_t = keras.Sequential([keras.layers.Dense(hidden, activation='tanh'),
                                       keras.layers.Dense(hidden, activation='tanh'),
                                       keras.layers.Dense(2)])

    def call(self, x, reverse=False):
        x0, x1 = x[:, :1], x[:, 1:]
        s = tf.clip_by_value(self.net_s(x0), -5.0, 5.0)
        t = self.net_t(x0)
        if not reverse:
            return tf.concat([x0, x1 * tf.exp(s) + t], axis=1), tf.reduce_sum(s, axis=1)
        else:
            return tf.concat([x0, (x1 - t) * tf.exp(-s)], axis=1)


class NFlow(keras.Model):
    def __init__(self, n_layers=8, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.coupling_layers = [CouplingLayer(hidden) for _ in range(n_layers)]
        self.perms = [tf.constant([1, 2, 0]) for _ in range(n_layers)]
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


# ── 2. Find all data files ────────────────────────────────────────────────────
DATA_FILES = sorted(glob.glob('3D_data/observables_*.dat'))

if not DATA_FILES:
    raise FileNotFoundError("No data files found in 3D_data/. Check your working directory.")

print(f"Found {len(DATA_FILES)} data files.")

# ── 3. Loop over all files ────────────────────────────────────────────────────
plt.style.use('dark_background')

for input_filename in DATA_FILES:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    weights_path = f'{base_name}.weights.index'

    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")

    # Check weights exist before trying to load
    if not os.path.exists(weights_path):
        print(f"  [Skipping] Weights file not found: {weights_path}")
        print(f"  Run train.py first to generate the weights.")
        continue

    # Load original telescope data
    df = pd.read_csv(input_filename, sep=r'\s+', comment='#',
                     names=['X', 'Y', 'V_Z'])
    obs        = df.values.astype(np.float32)
    mean, std  = obs.mean(0), obs.std(0)

    # Load model
    model = NFlow(n_layers=8, hidden=128)
    _ = model.log_prob(tf.constant(obs[:10]))   # build weights
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(f'{base_name}.weights').expect_partial()
    print(f"  Weights loaded from {weights_path}")

    # Sample
    print(f"  Sampling 500,000 stars...")
    samples_norm = model.sample(500_000)
    samples      = samples_norm * std + mean

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    lim          = 1.5
    vmin, vmax   = -10, 10

    # Left: raw telescope data
    sc1 = axes[0].scatter(df['X'], df['Y'], c=df['V_Z'], cmap='coolwarm',
                          s=4, alpha=0.8, vmin=vmin, vmax=vmax)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_title('What the Telescope Sees\n(Noisy, sparse data)',
                      fontsize=16, pad=15)
    axes[0].set_xlabel('X [kpc]', fontsize=14)
    axes[0].set_ylabel('Y [kpc]', fontsize=14)
    axes[0].grid(False)

    # Right: AI reconstruction
    sc2 = axes[1].hexbin(samples[:, 0], samples[:, 1], C=samples[:, 2],
                         gridsize=150, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_title('What the AI Reconstructed\n(Continuous physical model)',
                      fontsize=16, pad=15)
    axes[1].set_xlabel('X [kpc]', fontsize=14)
    axes[1].grid(False)

    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), pad=0.02, aspect=40)
    cbar.set_label(
        'Line-of-Sight Velocity ($V_Z$) [km/s]\n'
        '← Moving towards us (Blue)     Moving away (Red) →',
        fontsize=12)

    plt.suptitle(
        f'ML Reconstruction of Galaxy Kinematics\n({base_name})',
        fontsize=20, y=1.02, fontweight='bold')
    plt.subplots_adjust(top=0.85)

    out_png = f'{base_name}_presentation_visual.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_png}")

print(f"\n{'='*60}")
print("All done!")