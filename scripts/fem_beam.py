import numpy as np
from scipy.linalg import eigh

def beam_element_matrices(E, I, rho, A, L):
    k = (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])

    m = (rho * A * L / 420) * np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2]
    ])

    return k, m


def assemble_system(n_elements, E, I, rho, A, L_total):
    dof = 2 * (n_elements + 1)
    K = np.zeros((dof, dof))
    M = np.zeros((dof, dof))

    L = L_total / n_elements

    for i in range(n_elements):
        k, m = beam_element_matrices(E, I, rho, A, L)
        idx = 2*i
        K[idx:idx+4, idx:idx+4] += k
        M[idx:idx+4, idx:idx+4] += m

    return K, M


def apply_boundary_conditions(K, M):
    # Fix first node (cantilever)
    K = K[2:, 2:]
    M = M[2:, 2:]
    return K, M


def solve_modes(K, M, n_modes=3):
    eigvals, eigvecs = eigh(K, M)
    freqs = np.sqrt(np.abs(eigvals)) / (2*np.pi)
    return freqs[:n_modes], eigvecs[:, :n_modes]