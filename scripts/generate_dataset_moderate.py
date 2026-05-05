import numpy as np
import pandas as pd
from damage import assemble_damaged_system
from fem_beam import apply_boundary_conditions, solve_modes

E_base = 210e9
rho = 7800
A = 0.01
I = 1e-6
L_total = 1.0
n_elements = 10

samples = []

for _ in range(1000):
    damage_elem = np.random.randint(0, n_elements)
    alpha = np.random.uniform(0.1, 0.3)

    # Zones
    if damage_elem < 3:
        zone = 0
    elif damage_elem < 7:
        zone = 1
    else:
        zone = 2

    # Mild variation
    E = E_base * np.random.uniform(0.95, 1.05)

    K, M = assemble_damaged_system(
        n_elements, E, I, rho, A, L_total, damage_elem, alpha
    )

    K, M = apply_boundary_conditions(K, M)

    freqs, modes = solve_modes(K, M)

    # Moderate noise
    noise = np.random.normal(0, 0.07, size=freqs.shape)
    freqs = freqs + noise * freqs

    # Fixed sensor positions
    sensor_indices = np.linspace(0, len(modes[:, 0]) - 1, 5, dtype=int)
    mode_shape = modes[sensor_indices, 0]

    # Normalize
    mode_shape = mode_shape / np.linalg.norm(mode_shape)

    sample = list(freqs) + list(mode_shape) + [zone]
    samples.append(sample)

columns = ['f1','f2','f3','m1','m2','m3','m4','m5','zone']
df = pd.DataFrame(samples, columns=columns)

df.to_csv("data/dataset_moderate.csv", index=False)

print("Moderate dataset generated!")