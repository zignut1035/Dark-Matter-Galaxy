import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.interpolate import UnivariateSpline

# ── True DM profile parameters (from professor) ───────────────────────────────
# rho(r) = rho_DM * (r/rs)^(-gamma) * (1 + (r/rs)^alpha)^((gamma-beta)/alpha)

DM_PARAMS = {
    'observables_core_data1': dict(rho_DM=0.064e9, rs=1.0,  alpha=1, beta=3, gamma=0.0),
    'observables_core_data2': dict(rho_DM=0.6e9,   rs=0.45, alpha=1, beta=3, gamma=0.0),
    'observables_core_data3': dict(rho_DM=1.5e9,   rs=0.27, alpha=1, beta=3, gamma=0.0),
    'observables_cusp_data1': dict(rho_DM=0.064e9, rs=1.0,  alpha=1, beta=3, gamma=1.0),
    'observables_cusp_data2': dict(rho_DM=0.002e9, rs=5.0,  alpha=1, beta=3, gamma=1.0),
    'observables_cusp_data3': dict(rho_DM=0.0004e9,rs=13.0, alpha=1, beta=3, gamma=1.0),
}

def true_rho(r, rho_DM, rs, alpha, beta, gamma):
    """Generalised DM density profile."""
    x = r / rs
    return rho_DM * x**(-gamma) * (1 + x**alpha)**((gamma - beta) / alpha)


def estimated_rho_from_M(r_grid, M_r):
    """
    Derive density from enclosed mass:  rho(r) = (1/4pi r^2) * dM/dr
    Uses spline differentiation on log-log space for stability.
    """
    # Filter positive M values before log
    mask = M_r > 0
    if mask.sum() < 4:
        return np.full_like(r_grid, np.nan)

    log_r  = np.log(r_grid[mask])
    log_M  = np.log(M_r[mask])
    spl    = UnivariateSpline(log_r, log_M, s=1, k=3)

    # dM/dr = M(r) * d(lnM)/d(lnr) / r
    rho_est = np.full_like(r_grid, np.nan)
    for i, r in enumerate(r_grid):
        if not mask[i]:
            continue
        dln_M_dln_r = float(spl.derivative()(np.log(r)))
        M_r_val     = np.exp(spl(np.log(r)))
        dM_dr       = M_r_val * dln_M_dln_r / r
        rho_est[i]  = dM_dr / (4 * np.pi * r**2)

    return rho_est


# ── Load saved profiles and plot ──────────────────────────────────────────────
profile_files = sorted(glob.glob('*_profiles.npz'))

if not profile_files:
    raise FileNotFoundError(
        "No *_profiles.npz files found.\n"
        "Run train_nflow.py first to generate them.")

print(f"Found {len(profile_files)} profile files:")
for f in profile_files: print(f"  {f}")

fig_all, axes_all = plt.subplots(
    2, 3, figsize=(18, 10), sharex=False, sharey=False)
axes_flat = axes_all.flatten()

for idx, npz_file in enumerate(profile_files):
    base_name = npz_file.replace('_profiles.npz', '')
    print(f"\nPlotting DM profile: {base_name}")

    data   = np.load(npz_file)
    r_grid = data['r_grid']
    M_r    = data['M_r_b0']

    # Match to true parameters
    if base_name in DM_PARAMS:
        params = DM_PARAMS[base_name]
        print(f"  Matched params: {params}")
    else:
        print(f"  [Warning] No DM params found for {base_name} — skipping true profile")
        params = None

    # Estimated rho from M(r)
    rho_est = estimated_rho_from_M(r_grid, M_r)

    # True rho
    r_fine = np.logspace(np.log10(r_grid.min()), np.log10(r_grid.max()), 200)
    if params:
        rho_true = true_rho(r_fine, **params)

    # ── Individual plot ───────────────────────────────────────────────────────
    fig_single, ax = plt.subplots(figsize=(7, 5))

    if params:
        ax.loglog(r_fine, rho_true, 'k-', lw=2, label='True DM profile')

    valid = np.isfinite(rho_est) & (rho_est > 0)
    if valid.sum() > 2:
        ax.loglog(r_grid[valid], rho_est[valid], 'r--', lw=2,
                  label=r'Estimated $\rho_{DM}(r)$ from $M(r)$')

    ax.set_xlabel('r [kpc]', fontsize=12)
    ax.set_ylabel(r'$\rho_{DM}(r)$ [$M_\odot$ kpc$^{-3}$]', fontsize=12)
    ax.set_title(f'Dark Matter Density Profile\n({base_name})', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_single = f'{base_name}_DM_profile.png'
    fig_single.tight_layout()
    fig_single.savefig(out_single, dpi=150)
    plt.close(fig_single)
    print(f"  Saved → {out_single}")

    # ── Combined subplot ──────────────────────────────────────────────────────
    if idx < len(axes_flat):
        ax2 = axes_flat[idx]
        if params:
            ax2.loglog(r_fine, rho_true, 'k-', lw=2, label='True')
        if valid.sum() > 2:
            ax2.loglog(r_grid[valid], rho_est[valid], 'r--', lw=2,
                       label='Estimated')
        ax2.set_xlabel('r [kpc]')
        ax2.set_ylabel(r'$\rho_{DM}$ [$M_\odot$ kpc$^{-3}$]')
        ax2.set_title(base_name.replace('observables_', ''))
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(profile_files), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig_all.suptitle('Dark Matter Density Profiles — All Models', fontsize=13)
fig_all.tight_layout()
fig_all.savefig('DM_profiles_all.png', dpi=150)
plt.close(fig_all)
print(f"\nCombined plot saved → DM_profiles_all.png")

print(f"\n{'='*60}")
print("Done! Individual PNGs + combined DM_profiles_all.png saved.")
print(f"{'='*60}")