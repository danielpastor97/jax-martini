import MDAnalysis as mda
import lipyphilic as lpp

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def calculate_membrane_thickness_and_area_per_lipid(
    universe: mda.Universe,
    lipid_sel: str = "name GL1 GL2",
    thickness_sel: str = "name PO4",
):
    """Calculating membrane thickness and area per lipid."""

    # Assign lipids to leaflets
    leaflets = lpp.AssignLeaflets(universe=universe, lipid_sel=lipid_sel)
    leaflets.run()

    # Calculate area per lipid
    apl = lpp.analysis.AreaPerLipid(
        universe=universe, lipid_sel=lipid_sel, leaflets=leaflets.leaflets
    )
    apl.run()

    # Calculate membrane thickness
    thicknesses = lpp.analysis.MembThickness(
        universe=universe, lipid_sel=thickness_sel, leaflets=leaflets.leaflets
    )
    thicknesses.run()

    avg_area_per_lipid = jnp.mean(apl.areas, axis=0)
    return thicknesses.memb_thickness, avg_area_per_lipid


def compute_obs_membrane(u):
    # Compute thickness and area per lipid
    thicknesses, area_per_lipid = calculate_membrane_thickness_and_area_per_lipid(u)
    assert thicknesses.shape[-1] == area_per_lipid.shape[-1]
    assert len(thicknesses.shape) == 1
    assert len(area_per_lipid.shape) == 1

    return thicknesses, area_per_lipid
