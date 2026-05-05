import numpy as np
from fem_beam import beam_element_matrices

def assemble_damaged_system(n_elements, E, I, rho, A, L_total, damage_elem, alpha):
    """
    Assemble FEM system with single localized damage.

    Parameters:
    - n_elements: number of beam elements
    - E: Young's modulus
    - I: moment of inertia
    - rho: density
    - A: cross-sectional area
    - L_total: total beam length
    - damage_elem: index of damaged element
    - alpha: damage severity (0.1–0.3 typical)

    Returns:
    - K: global stiffness matrix
    - M: global mass matrix
    """

    dof = 2 * (n_elements + 1)
    K = np.zeros((dof, dof))
    M = np.zeros((dof, dof))

    L = L_total / n_elements

    for i in range(n_elements):

        # Apply damage only to selected element
        if i == damage_elem:
            E_local = (1 - alpha) * E
        else:
            E_local = E

        k, m = beam_element_matrices(E_local, I, rho, A, L)

        idx = 2 * i
        K[idx:idx+4, idx:idx+4] += k
        M[idx:idx+4, idx:idx+4] += m

    return K, M


# 🔥 OPTIONAL (for advanced experiments)
def assemble_multi_damage_system(n_elements, E, I, rho, A, L_total, damage_elems, alpha_list):
    """
    Assemble FEM system with multiple damaged elements.

    Parameters:
    - damage_elems: list of element indices
    - alpha_list: corresponding severity values
    """

    dof = 2 * (n_elements + 1)
    K = np.zeros((dof, dof))
    M = np.zeros((dof, dof))

    L = L_total / n_elements

    for i in range(n_elements):

        if i in damage_elems:
            idx_alpha = damage_elems.index(i)
            alpha = alpha_list[idx_alpha]
            E_local = (1 - alpha) * E
        else:
            E_local = E

        k, m = beam_element_matrices(E_local, I, rho, A, L)

        idx = 2 * i
        K[idx:idx+4, idx:idx+4] += k
        M[idx:idx+4, idx:idx+4] += m

    return K, M