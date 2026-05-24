import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import glob
import os

# ── 1. Model Definitions (Must match train.py exactly) ────────────────────────
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


# ── 2. Data Loading & Plotting Loop ───────────────────────────────────────────
data_files = sorted(glob.glob('3D_data/observables_*.dat'))
print(f"Found {len(data_files)} target datasets.")

for filepath in data_files:
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\nProcessing: {base_name}")
    
    # Define paths
    gt_filename = filepath.replace('3D_data', '6D_data').replace('observables_', 'Mock_isotropic_')
    weights_path = f'{base_name}.weights'
    npz_path = f'{base_name}_profiles.npz'
    
    # 1. Load True 6D Data
    try:
        df_6d = pd.read_csv(gt_filename, sep=r'\s+', comment='#', 
                            names=['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
        X_true = df_6d['X'].values
        Y_true = df_6d['Y'].values
        VZ_true = df_6d['VZ'].values
    except FileNotFoundError:
        print(f"  [Skip] True 6D data not found: {gt_filename}")
        continue

    # 2. Verify AI weights and normalizations exist
    if not os.path.exists(npz_path) or not glob.glob(weights_path + '*'):
        print(f"  [Skip] Missing .weights or .npz for {base_name}. Run train.py first.")
        continue

    data = np.load(npz_path)
    mean_obs = data['mean_obs']
    std_obs = data['std_obs']

    # 3. Load Model & Generate Stars
    print("  Loading trained AI and generating mock stars...")
    model = NFlow(n_layers=12, hidden=256)
    _ = model(tf.zeros((1, 3))) # Dummy pass to initialize weights
    
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.read(weights_path).expect_partial()
    
    # Generate the exact same number of stars as the real dataset
    n_stars = len(X_true)
    samples_norm = model.sample(n_stars)
    samples = samples_norm * std_obs + mean_obs # Unnormalize
    
    X_gen = samples[:, 0]
    Y_gen = samples[:, 1]
    VZ_gen = samples[:, 2]

    # 4. Plot Side-by-Side Comparison
    print("  Creating comparison plot...")
    v_abs = np.percentile(np.abs(VZ_true), 98) # Symmetric color scaling limit
    
    # 1. MAKE THE IMAGE WIDER AND TALLER (figsize increased to 18x8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0a0a0a')
    
    # Left Panel: True 6D Data
    sort_idx1 = np.argsort(np.abs(VZ_true))
    # 2. MAKE THE STARS BIGGER (s=0.5 instead of 0.2)
    sc1 = ax1.scatter(X_true[sort_idx1], Y_true[sort_idx1], c=VZ_true[sort_idx1], 
                      cmap='RdBu_r', s=0.5, alpha=0.9, vmin=-v_abs, vmax=v_abs, edgecolors='none')
    ax1.set_title(f'Real 6D Data (Projected)\n{n_stars:,} stars', color='white', fontsize=14)
    ax1.set_facecolor('#0a0a0a')
    
    # Right Panel: AI Generated Data
    sort_idx2 = np.argsort(np.abs(VZ_gen))
    sc2 = ax2.scatter(X_gen[sort_idx2], Y_gen[sort_idx2], c=VZ_gen[sort_idx2], 
                      cmap='RdBu_r', s=0.5, alpha=0.9, vmin=-v_abs, vmax=v_abs, edgecolors='none')
    ax2.set_title(f'Generated 3D Data (Normalizing Flow)\n{n_stars:,} stars', color='white', fontsize=14)
    ax2.set_facecolor('#0a0a0a')
    
    # ── FORCE FORMATTING AND LIMITS ──
    # 3. ZOOM IN CLOSER (Changed from 2.0 to 1.2 kpc)
    zoom_radius = 1.2  

    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(-zoom_radius, zoom_radius)
        ax.set_ylim(-zoom_radius, zoom_radius)
        ax.autoscale(False) 
        
        ax.set_xlabel('X [kpc]', color='white', fontsize=12)
        ax.set_ylabel('Y [kpc]', color='white', fontsize=12)
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(True, color='white', alpha=0.15)
        
    # Global Colorbar
    cbar = plt.colorbar(sc2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label('Line-of-Sight Velocity ($V_Z$) [km/s]', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    out_png = f'{base_name}_true_vs_generated.png'
    # High-res output (dpi=300)
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_png}")

print("\nDone! Visualizations complete.")