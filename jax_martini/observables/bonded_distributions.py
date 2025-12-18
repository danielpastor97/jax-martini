import MDAnalysis as mda
import numpy as onp
from typing import List, Tuple, Dict
from MDAnalysis.lib.distances import calc_bonds, calc_angles


# Calculating bond statistics
def calculate_bond_statistics(
    universe: mda.Universe,
    bonds_dict: Dict[str, List[Tuple[str, str]]],
    start: int = 0,
    stop: int = None,
    step: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Efficiently calculate mean and standard deviation of bond distances.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe containing trajectory
    bonds_dict : Dict[str, List[Tuple[str, str]]]
        Dictionary with residue names as keys and lists of tuples containing pairs of atom names defining bonds
    start : int, optional
        First frame to analyze (default: 0)
    stop : int, optional
        Last frame to analyze (default: None, means last frame)
    step : int, optional
        Step between frames to analyze (default: 1)

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with bond names as keys and their statistics as values
        Format: {'bondname': {'mean': float, 'std': float}}
    """
    # Pre-computing selections for all bonds
    selections = {}
    for resname, bonds in bonds_dict.items():
        selections[resname] = {}
        for bond in bonds:
            atom1_sel = universe.select_atoms(f"name {bond[0]} and resname {resname}")
            atom2_sel = universe.select_atoms(f"name {bond[1]} and resname {resname}")
            selections[resname][f"{bond[0]}-{bond[1]}"] = (atom1_sel, atom2_sel)

    # Iterating through trajectory once
    all_bond_dists = {}
    for ts in universe.trajectory[start:stop:step]:
        for resname, bond_names in selections.items():
            all_bond_dists[resname] = (
                {} if resname not in all_bond_dists else all_bond_dists[resname]
            )
            for bond_name, (sel1, sel2) in bond_names.items():
                all_bond_dists[resname][bond_name] = (
                    []
                    if bond_name not in all_bond_dists[resname]
                    else all_bond_dists[resname][bond_name]
                )

                # Calculating distances for all instances of this bond type
                distances = calc_bonds(
                    sel1.positions, sel2.positions, box=ts.dimensions
                )
                perframe_dists = []
                for i in range(distances.shape[0]):
                    perframe_dists.append(distances[i])
                all_bond_dists[resname][bond_name].append(perframe_dists)

    for resname in all_bond_dists.keys():
        for bond_name in all_bond_dists[resname].keys():
            all_bond_dists[resname][bond_name] = onp.array(
                all_bond_dists[resname][bond_name]
            )

    return all_bond_dists


# Calculating angle statistics
def calculate_angle_statistics(
    universe: mda.Universe,
    angles_dict: Dict[str, List[Tuple[str, str, str]]],
    start: int = 0,
    stop: int = None,
    step: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Efficiently calculate mean and standard deviation of angle distances.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe containing trajectory
    angles_dict : Dict[str, List[Tuple[str, str, str]]]
        Dictionary with residue names as keys and lists of tuples containing pairs of atom names defining angles
    start : int, optional
        First frame to analyze (default: 0)
    stop : int, optional
        Last frame to analyze (default: None, means last frame)
    step : int, optional
        Step between frames to analyze (default: 1)

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with angle names as keys and their statistics as values
        Format: {'angle': {'mean': float, 'std': float}}
    """
    # Pre-computing selections for all angles
    selections = {}
    for resname, angles in angles_dict.items():
        selections[resname] = {}
        for angle in angles:
            atom1_sel = universe.select_atoms(f"name {angle[0]} and resname {resname}")
            atom2_sel = universe.select_atoms(f"name {angle[1]} and resname {resname}")
            atom3_sel = universe.select_atoms(f"name {angle[2]} and resname {resname}")
            selections[resname][f"{angle[0]}-{angle[1]}-{angle[2]}"] = (
                atom1_sel,
                atom2_sel,
                atom3_sel,
            )

    # Iterating through trajectory once
    all_angle_dists = {}
    for ts in universe.trajectory[start:stop:step]:
        for resname, angle_names in selections.items():
            all_angle_dists[resname] = (
                {} if resname not in all_angle_dists else all_angle_dists[resname]
            )
            for angle_name, (sel1, sel2, sel3) in angle_names.items():
                all_angle_dists[resname][angle_name] = (
                    []
                    if angle_name not in all_angle_dists[resname]
                    else all_angle_dists[resname][angle_name]
                )

                # Calculating angles for all instances of this angle type
                angles = calc_angles(
                    sel1.positions, sel2.positions, sel3.positions, box=ts.dimensions
                )
                perframe_dists = []
                for i in range(angles.shape[0]):
                    perframe_dists.append(angles[i])
                all_angle_dists[resname][angle_name].append(perframe_dists)

    for resname in all_angle_dists.keys():
        for angle_name in all_angle_dists[resname].keys():
            all_angle_dists[resname][angle_name] = onp.array(
                all_angle_dists[resname][angle_name]
            )

    return all_angle_dists
