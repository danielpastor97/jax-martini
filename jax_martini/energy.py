import pdb
import numpy as onp

import jax

jax.config.update("jax_enable_x64", True)
from jax_md import space, smap
import jax.numpy as jnp

from jax_martini import utils


def get_mapped_bond_fn(bonds, bond_type_idxs, bond_type_ks, bond_type_lengths):

    def bond_val(R, bond, bond_type, displacement_fn):

        i = bond[0]
        j = bond[1]

        spring_k = bond_type_ks[bond_type]
        r0 = bond_type_lengths[bond_type]

        r = space.distance(displacement_fn(R[i], R[j]))
        return 0.5 * spring_k * (r - r0) ** 2

    def mapped_bonds(R, box_dimensions):
        displacement_fn, _ = space.periodic(box_dimensions)
        bond_vals = jax.vmap(bond_val, (None, 0, 0, None))(
            R, bonds, bond_type_idxs, displacement_fn
        )
        return bond_vals.sum()

    return mapped_bonds


# computing the angle between three particles
def compute_angle(displacement, R, triplet):
    """Compute angle between three particles using arctan2.

    Args:
        displacement: Displacement function from space
        R: Array of positions
        triplet: Tuple of 3 indices (i,j,k) defining the angle

    Returns:
        Angle in radians between vectors ij and jk
    """
    # extracting the indices
    i, j, k = triplet
    # calculating the displacements
    dr_ji = displacement(R[j], R[i])
    dr_jk = displacement(R[j], R[k])

    # calculating the cross and dot products
    cross_prod = jnp.cross(dr_ji, dr_jk)
    dot_prod = jnp.dot(dr_ji, dr_jk)

    # using arctan2 for better numerical stability
    # arctan2(|a × b|, a · b) gives angle between vectors
    return jnp.arctan2(jnp.sqrt(jnp.sum(cross_prod**2)), dot_prod)


# https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#harmonicangle
# Martini2 uses angle type 2 (G96 Angle) so MSE is defined w.r.t. cos(theta)
def get_mapped_angle_fn(
    angles, angle_type_idxs, angle_type_ks, angle_type_theta0s, use_m2=False
):

    def angle_val(R, angle, angle_type, displacement_fn):
        i = angle[0]
        j = angle[1]
        k = angle[2]

        spring_k = angle_type_ks[angle_type]
        theta0 = angle_type_theta0s[angle_type]
        theta = compute_angle(displacement_fn, R, (i, j, k))
        if use_m2:
            return 0.5 * spring_k * (jnp.cos(theta) - jnp.cos(theta0)) ** 2, theta
        else:
            return 0.5 * spring_k * (theta - theta0) ** 2, theta

    def mapped_angles(R, box_dimensions):
        displacement_fn, _ = space.periodic(box_dimensions)
        angle_vals, thetas = jax.vmap(angle_val, (None, 0, 0, None))(
            R, angles, angle_type_idxs, displacement_fn
        )
        return angle_vals.sum()

    return mapped_angles


def lennard_jones(r, eps, sigma, **kwargs):
    cutoff = 1.1

    # calculating the standard LJ potential
    v = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)

    # calculating the value of the potential at cutoff
    v_c = 4 * eps * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)

    # applying the shifting function: V_s(r) = V(r) - V(r_c) for r < r_c, 0 otherwise
    energy = jnp.where(
        r < cutoff, v - v_c, 0.0  # shifting the potential by subtracting V(r_c)
    )

    return energy


def pair_lj_fn(R, pair, sigmas, epsilons, types, displacement_fn):
    i = pair[0]
    j = pair[1]

    i_type = types[i]
    j_type = types[j]

    sigma = sigmas[i_type, j_type]
    eps = epsilons[i_type, j_type]

    r = space.distance(displacement_fn(R[i], R[j]))
    return lennard_jones(r, eps, sigma)
