import numpy as np

# 1. Load the full 6D phase-space data
# (Replace 'data1.dat' with your actual file name)
data = np.loadtxt('Mock_isotropic_cusp_data3.dat')

# Column Index Reference:
# 0 = x, 1 = y, 2 = z, 3 = vx, 4 = vy, 5 = vz

# 2. "Blind" the data 
# We slice the array to keep only columns 0, 1, and 5
observable_data = data[:, [0, 1, 5]]

# 3. Verify the shapes
print(f"Original 6D data shape: {data.shape}")
print(f"Blinded 3D observable shape: {observable_data.shape}")
# Example output:
# Original 6D data shape: (10000, 6)
# Blinded 3D observable shape: (10000, 3)

# 4. Save the blinded data to a new file for JFlow training
# Saving in scientific notation (%.18e) to match your original format
np.savetxt('observables_cusp_data3.dat', observable_data, 
           header='x [kpc]    y [kpc]    vz [km/s]', 
           fmt='%.18e')

print("Successfully saved blinded data to 'observables_data1.dat'")