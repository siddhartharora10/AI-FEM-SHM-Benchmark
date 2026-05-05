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

for _ in range(500):
    damage_elem = np.random.randint(0, n_elements)
    alpha = np.random.uniform(0.1, 0.3)

    # Overlapping zones (harder classification)
    if damage_elem <= 4:
        zone = 0
    elif damage_elem <= 7:
        zone = 1
    else:
        zone = 2

    # Material variability
    E_variation = np.random.uniform(0.85, 1.15)
    E = E_base * E_variation

    K, M = assemble_damaged_system(
        n_elements, E, I, rho, A, L_total, damage_elem, alpha
    )

    K, M = apply_boundary_conditions(K, M)

    freqs, modes = solve_modes(K, M)

    # Stronger noise (15%)
    noise = np.random.normal(0, 0.15, size=freqs.shape)
    freqs_noisy = freqs + noise * freqs

    # Sparse sensors (randomized positions)
    total_nodes = len(modes[:, 0])
    sensor_indices = np.sort(
        np.random.choice(total_nodes, size=5, replace=False)
    )
    mode_shape = modes[sensor_indices, 0]

    # Normalize mode shape (critical)
    mode_shape = mode_shape / np.linalg.norm(mode_shape)

    sample = list(freqs_noisy) + list(mode_shape) + [zone]
    samples.append(sample)

columns = ['f1','f2','f3','m1','m2','m3','m4','m5','zone']

df = pd.DataFrame(samples, columns=columns)
df.to_csv("data/dataset_realistic.csv", index=False)

print("FINAL realistic dataset generated!")