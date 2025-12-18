import numpy as onp
from pathlib import Path
import itertools
import jax
import jax.numpy as jnp
from copy import deepcopy


# Hardcoded for reproducibility
bead_types = [
    "P6",
    "P5",
    "P4",
    "P3",
    "P2",
    "P1",
    "N6",
    "N5",
    "N4",
    "N3",
    "N2",
    "N1",
    "C6",
    "C5",
    "C4",
    "C3",
    "C2",
    "C1",
    "X4",
    "X3",
    "X2",
    "X1",
    "P6d",
    "P5d",
    "P4d",
    "P3d",
    "P2d",
    "P1d",
    "N6d",
    "N5d",
    "N4d",
    "N3d",
    "N2d",
    "N1d",
    "P6a",
    "P5a",
    "P4a",
    "P3a",
    "P2a",
    "P1a",
    "N6a",
    "N5a",
    "N4a",
    "N3a",
    "N2a",
    "N1a",
    "C6v",
    "C5v",
    "C4v",
    "C3v",
    "C2v",
    "C1v",
    "X4v",
    "X3v",
    "X2v",
    "X1v",
    "C6e",
    "C5e",
    "C4e",
    "C3e",
    "C2e",
    "C1e",
    "X3e",
    "X4e",
    "X2e",
    "X1e",
    "D",
    "Q5",
    "Q4",
    "Q3",
    "Q2",
    "Q1",
    "Q5p",
    "Q4p",
    "Q3p",
    "Q2p",
    "Q1p",
    "Q5n",
    "Q4n",
    "Q3n",
    "Q2n",
    "Q1n",
    "P6q",
    "P5q",
    "P4q",
    "P3q",
    "P2q",
    "P1q",
    "N6q",
    "N5q",
    "N4q",
    "N3q",
    "N2q",
    "N1q",
    "C6q",
    "C5q",
    "C4q",
    "C3q",
    "C2q",
    "C1q",
    "X4q",
    "X3q",
    "X2q",
    "X1q",
    "P6dq",
    "P5dq",
    "P4dq",
    "P3dq",
    "P2dq",
    "P1dq",
    "N6dq",
    "N5dq",
    "N4dq",
    "N3dq",
    "N2dq",
    "N1dq",
    "P6aq",
    "P5aq",
    "P4aq",
    "P3aq",
    "P2aq",
    "P1aq",
    "N6aq",
    "N5aq",
    "N4aq",
    "N3aq",
    "N2aq",
    "N1aq",
    "C6vq",
    "C5vq",
    "C4vq",
    "C3vq",
    "C2vq",
    "C1vq",
    "X4vq",
    "X3vq",
    "X2vq",
    "X1vq",
    "C6eq",
    "C5eq",
    "C4eq",
    "C3eq",
    "C2eq",
    "C1eq",
    "X4eq",
    "X3eq",
    "X2eq",
    "X1eq",
    "P6h",
    "P5h",
    "P4h",
    "P3h",
    "P2h",
    "P1h",
    "N6h",
    "N5h",
    "N4h",
    "N3h",
    "N2h",
    "N1h",
    "C6h",
    "C5h",
    "C4h",
    "C3h",
    "C2h",
    "C1h",
    "X4h",
    "X3h",
    "X2h",
    "X1h",
    "P6dh",
    "P5dh",
    "P4dh",
    "P3dh",
    "P2dh",
    "P1dh",
    "N6dh",
    "N5dh",
    "N4dh",
    "N3dh",
    "N2dh",
    "N1dh",
    "P6ah",
    "P5ah",
    "P4ah",
    "P3ah",
    "P2ah",
    "P1ah",
    "N6ah",
    "N5ah",
    "N4ah",
    "N3ah",
    "N2ah",
    "N1ah",
    "C6vh",
    "C5vh",
    "C4vh",
    "C3vh",
    "C2vh",
    "C1vh",
    "X4vh",
    "X3vh",
    "X2vh",
    "X1vh",
    "C6eh",
    "C5eh",
    "C4eh",
    "C3eh",
    "C2eh",
    "C1eh",
    "X4eh",
    "X3eh",
    "X2eh",
    "X1eh",
    "P6r",
    "P5r",
    "P4r",
    "P3r",
    "P2r",
    "P1r",
    "N6r",
    "N5r",
    "N4r",
    "N3r",
    "N2r",
    "N1r",
    "C6r",
    "C5r",
    "C4r",
    "C3r",
    "C2r",
    "C1r",
    "X4r",
    "X3r",
    "X2r",
    "X1r",
    "P6dr",
    "P5dr",
    "P4dr",
    "P3dr",
    "P2dr",
    "P1dr",
    "N6dr",
    "N5dr",
    "N4dr",
    "N3dr",
    "N2dr",
    "N1dr",
    "P6ar",
    "P5ar",
    "P4ar",
    "P3ar",
    "P2ar",
    "P1ar",
    "N6ar",
    "N5ar",
    "N4ar",
    "N3ar",
    "N2ar",
    "N1ar",
    "C6vr",
    "C5vr",
    "C4vr",
    "C3vr",
    "C2vr",
    "C1vr",
    "X4vr",
    "X3vr",
    "X2vr",
    "X1vr",
    "C6er",
    "C5er",
    "C4er",
    "C3er",
    "C2er",
    "C1er",
    "X4er",
    "X3er",
    "X2er",
    "X1er",
    "SP6",
    "SP5",
    "SP4",
    "SP3",
    "SP2",
    "SP1",
    "SN6",
    "SN5",
    "SN4",
    "SN3",
    "SN2",
    "SN1",
    "SC6",
    "SC5",
    "SC4",
    "SC3",
    "SC2",
    "SC1",
    "SX4",
    "SX3",
    "SX2",
    "SX1",
    "SP6d",
    "SP5d",
    "SP4d",
    "SP3d",
    "SP2d",
    "SP1d",
    "SN6d",
    "SN5d",
    "SN4d",
    "SN3d",
    "SN2d",
    "SN1d",
    "SP6a",
    "SP5a",
    "SP4a",
    "SP3a",
    "SP2a",
    "SP1a",
    "SN6a",
    "SN5a",
    "SN4a",
    "SN3a",
    "SN2a",
    "SN1a",
    "SC6v",
    "SC5v",
    "SC4v",
    "SC3v",
    "SC2v",
    "SC1v",
    "SX4v",
    "SX3v",
    "SX2v",
    "SX1v",
    "SC6e",
    "SC5e",
    "SC4e",
    "SC3e",
    "SC2e",
    "SC1e",
    "SX4e",
    "SX3e",
    "SX2e",
    "SX1e",
    "SD",
    "SQ5",
    "SQ4",
    "SQ3",
    "SQ2",
    "SQ1",
    "SQ5p",
    "SQ4p",
    "SQ3p",
    "SQ2p",
    "SQ1p",
    "SQ5n",
    "SQ4n",
    "SQ3n",
    "SQ2n",
    "SQ1n",
    "SP6q",
    "SP5q",
    "SP4q",
    "SP3q",
    "SP2q",
    "SP1q",
    "SN6q",
    "SN5q",
    "SN4q",
    "SN3q",
    "SN2q",
    "SN1q",
    "SC6q",
    "SC5q",
    "SC4q",
    "SC3q",
    "SC2q",
    "SC1q",
    "SX4q",
    "SX3q",
    "SX2q",
    "SX1q",
    "SP6dq",
    "SP5dq",
    "SP4dq",
    "SP3dq",
    "SP2dq",
    "SP1dq",
    "SN6dq",
    "SN5dq",
    "SN4dq",
    "SN3dq",
    "SN2dq",
    "SN1dq",
    "SP6aq",
    "SP5aq",
    "SP4aq",
    "SP3aq",
    "SP2aq",
    "SP1aq",
    "SN6aq",
    "SN5aq",
    "SN4aq",
    "SN3aq",
    "SN2aq",
    "SN1aq",
    "SC6vq",
    "SC5vq",
    "SC4vq",
    "SC3vq",
    "SC2vq",
    "SC1vq",
    "SX4vq",
    "SX3vq",
    "SX2vq",
    "SX1vq",
    "SC6eq",
    "SC5eq",
    "SC4eq",
    "SC3eq",
    "SC2eq",
    "SC1eq",
    "SX4eq",
    "SX3eq",
    "SX2eq",
    "SX1eq",
    "SP6h",
    "SP5h",
    "SP4h",
    "SP3h",
    "SP2h",
    "SP1h",
    "SN6h",
    "SN5h",
    "SN4h",
    "SN3h",
    "SN2h",
    "SN1h",
    "SC6h",
    "SC5h",
    "SC4h",
    "SC3h",
    "SC2h",
    "SC1h",
    "SX4h",
    "SX3h",
    "SX2h",
    "SX1h",
    "SP6dh",
    "SP5dh",
    "SP4dh",
    "SP3dh",
    "SP2dh",
    "SP1dh",
    "SN6dh",
    "SN5dh",
    "SN4dh",
    "SN3dh",
    "SN2dh",
    "SN1dh",
    "SP6ah",
    "SP5ah",
    "SP4ah",
    "SP3ah",
    "SP2ah",
    "SP1ah",
    "SN6ah",
    "SN5ah",
    "SN4ah",
    "SN3ah",
    "SN2ah",
    "SN1ah",
    "SC6vh",
    "SC5vh",
    "SC4vh",
    "SC3vh",
    "SC2vh",
    "SC1vh",
    "SX4vh",
    "SX3vh",
    "SX2vh",
    "SX1vh",
    "SC6eh",
    "SC5eh",
    "SC4eh",
    "SC3eh",
    "SC2eh",
    "SC1eh",
    "SX4eh",
    "SX3eh",
    "SX2eh",
    "SX1eh",
    "SP6r",
    "SP5r",
    "SP4r",
    "SP3r",
    "SP2r",
    "SP1r",
    "SN6r",
    "SN5r",
    "SN4r",
    "SN3r",
    "SN2r",
    "SN1r",
    "SC6r",
    "SC5r",
    "SC4r",
    "SC3r",
    "SC2r",
    "SC1r",
    "SX4r",
    "SX3r",
    "SX2r",
    "SX1r",
    "SP6dr",
    "SP5dr",
    "SP4dr",
    "SP3dr",
    "SP2dr",
    "SP1dr",
    "SN6dr",
    "SN5dr",
    "SN4dr",
    "SN3dr",
    "SN2dr",
    "SN1dr",
    "SP6ar",
    "SP5ar",
    "SP4ar",
    "SP3ar",
    "SP2ar",
    "SP1ar",
    "SN6ar",
    "SN5ar",
    "SN4ar",
    "SN3ar",
    "SN2ar",
    "SN1ar",
    "SC6vr",
    "SC5vr",
    "SC4vr",
    "SC3vr",
    "SC2vr",
    "SC1vr",
    "SX4vr",
    "SX3vr",
    "SX2vr",
    "SX1vr",
    "SC6er",
    "SC5er",
    "SC4er",
    "SC3er",
    "SC2er",
    "SC1er",
    "SX4er",
    "SX3er",
    "SX2er",
    "SX1er",
    "TP6",
    "TP5",
    "TP4",
    "TP3",
    "TP2",
    "TP1",
    "TN6",
    "TN5",
    "TN4",
    "TN3",
    "TN2",
    "TN1",
    "TC6",
    "TC5",
    "TC4",
    "TC3",
    "TC2",
    "TC1",
    "TX4",
    "TX3",
    "TX2",
    "TX1",
    "TP6d",
    "TP5d",
    "TP4d",
    "TP3d",
    "TP2d",
    "TP1d",
    "TN6d",
    "TN5d",
    "TN4d",
    "TN3d",
    "TN2d",
    "TN1d",
    "TP6a",
    "TP5a",
    "TP4a",
    "TP3a",
    "TP2a",
    "TP1a",
    "TN6a",
    "TN5a",
    "TN4a",
    "TN3a",
    "TN2a",
    "TN1a",
    "TC6v",
    "TC5v",
    "TC4v",
    "TC3v",
    "TC2v",
    "TC1v",
    "TX4v",
    "TX3v",
    "TX2v",
    "TX1v",
    "TC6e",
    "TC5e",
    "TC4e",
    "TC3e",
    "TC2e",
    "TC1e",
    "TX4e",
    "TX3e",
    "TX2e",
    "TX1e",
    "TD",
    "TQ5",
    "TQ4",
    "TQ3",
    "TQ2",
    "TQ1",
    "TQ5p",
    "TQ4p",
    "TQ3p",
    "TQ2p",
    "TQ1p",
    "TQ5n",
    "TQ4n",
    "TQ3n",
    "TQ2n",
    "TQ1n",
    "TP6q",
    "TP5q",
    "TP4q",
    "TP3q",
    "TP2q",
    "TP1q",
    "TN6q",
    "TN5q",
    "TN4q",
    "TN3q",
    "TN2q",
    "TN1q",
    "TC6q",
    "TC5q",
    "TC4q",
    "TC3q",
    "TC2q",
    "TC1q",
    "TX4q",
    "TX3q",
    "TX2q",
    "TX1q",
    "TP6dq",
    "TP5dq",
    "TP4dq",
    "TP3dq",
    "TP2dq",
    "TP1dq",
    "TN6dq",
    "TN5dq",
    "TN4dq",
    "TN3dq",
    "TN2dq",
    "TN1dq",
    "TP6aq",
    "TP5aq",
    "TP4aq",
    "TP3aq",
    "TP2aq",
    "TP1aq",
    "TN6aq",
    "TN5aq",
    "TN4aq",
    "TN3aq",
    "TN2aq",
    "TN1aq",
    "TC6vq",
    "TC5vq",
    "TC4vq",
    "TC3vq",
    "TC2vq",
    "TC1vq",
    "TX4vq",
    "TX3vq",
    "TX2vq",
    "TX1vq",
    "TC6eq",
    "TC5eq",
    "TC4eq",
    "TC3eq",
    "TC2eq",
    "TC1eq",
    "TX4eq",
    "TX3eq",
    "TX2eq",
    "TX1eq",
    "TP6h",
    "TP5h",
    "TP4h",
    "TP3h",
    "TP2h",
    "TP1h",
    "TN6h",
    "TN5h",
    "TN4h",
    "TN3h",
    "TN2h",
    "TN1h",
    "TC6h",
    "TC5h",
    "TC4h",
    "TC3h",
    "TC2h",
    "TC1h",
    "TX4h",
    "TX3h",
    "TX2h",
    "TX1h",
    "TP6dh",
    "TP5dh",
    "TP4dh",
    "TP3dh",
    "TP2dh",
    "TP1dh",
    "TN6dh",
    "TN5dh",
    "TN4dh",
    "TN3dh",
    "TN2dh",
    "TN1dh",
    "TP6ah",
    "TP5ah",
    "TP4ah",
    "TP3ah",
    "TP2ah",
    "TP1ah",
    "TN6ah",
    "TN5ah",
    "TN4ah",
    "TN3ah",
    "TN2ah",
    "TN1ah",
    "TC6vh",
    "TC5vh",
    "TC4vh",
    "TC3vh",
    "TC2vh",
    "TC1vh",
    "TX4vh",
    "TX3vh",
    "TX2vh",
    "TX1vh",
    "TC6eh",
    "TC5eh",
    "TC4eh",
    "TC3eh",
    "TC2eh",
    "TC1eh",
    "TX4eh",
    "TX3eh",
    "TX2eh",
    "TX1eh",
    "TP6r",
    "TP5r",
    "TP4r",
    "TP3r",
    "TP2r",
    "TP1r",
    "TN6r",
    "TN5r",
    "TN4r",
    "TN3r",
    "TN2r",
    "TN1r",
    "TC6r",
    "TC5r",
    "TC4r",
    "TC3r",
    "TC2r",
    "TC1r",
    "TX4r",
    "TX3r",
    "TX2r",
    "TX1r",
    "TP6dr",
    "TP5dr",
    "TP4dr",
    "TP3dr",
    "TP2dr",
    "TP1dr",
    "TN6dr",
    "TN5dr",
    "TN4dr",
    "TN3dr",
    "TN2dr",
    "TN1dr",
    "TP6ar",
    "TP5ar",
    "TP4ar",
    "TP3ar",
    "TP2ar",
    "TP1ar",
    "TN6ar",
    "TN5ar",
    "TN4ar",
    "TN3ar",
    "TN2ar",
    "TN1ar",
    "TC6vr",
    "TC5vr",
    "TC4vr",
    "TC3vr",
    "TC2vr",
    "TC1vr",
    "TX4vr",
    "TX3vr",
    "TX2vr",
    "TX1vr",
    "TC6er",
    "TC5er",
    "TC4er",
    "TC3er",
    "TC2er",
    "TC1er",
    "TX4er",
    "TX3er",
    "TX2er",
    "TX1er",
    "W",
    "SW",
    "TW",
    "U",
]
n_bead_types = len(bead_types)
bead_type_to_idx = {bead_type: idx for idx, bead_type in enumerate(bead_types)}
idx_to_bead_type = {idx: bead_type for idx, bead_type in enumerate(bead_types)}

bead_types_m2 = [
    "Qda",
    "Qd",
    "Qa",
    "Q0",
    "P5",
    "P4",
    "P3",
    "P2",
    "P1",
    "Nda",
    "Nd",
    "Na",
    "N0",
    "C5",
    "C4",
    "C3",
    "C2",
    "C1",
    "SQda",
    "SQd",
    "SQa",
    "SQ0",
    "SP5",
    "SP4",
    "SP3",
    "SP2",
    "SP1",
    "SNda",
    "SNd",
    "SNa",
    "SN0",
    "SC5",
    "SC4",
    "SC3",
    "SC2",
    "SC1",
    "SP1c",
]
n_bead_types_m2 = len(bead_types_m2)
bead_type_to_idx_m2 = {bead_type: idx for idx, bead_type in enumerate(bead_types_m2)}
idx_to_bead_type_m2 = {idx: bead_type for idx, bead_type in enumerate(bead_types_m2)}


def read_lj_params(fpath="jax_lipids/data/params/m3/martini_v3.0.0.itp"):
    fpath = Path(fpath)
    assert fpath.exists()

    with open(fpath, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)

    start_nonbonded = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[ nonbond_params ]":
            start_nonbonded = l_idx
            break

    assert start_nonbonded is not None
    epsilons = onp.zeros((n_bead_types, n_bead_types))  # FIXME: default zero?
    sigmas = onp.zeros((n_bead_types, n_bead_types))  # FIXME: default zero?
    for l_idx in range(start_nonbonded + 1, n_lines):
        line = lines[l_idx]
        tokens = line.strip().split()
        assert len(tokens) == 5

        type1 = bead_type_to_idx[tokens[0]]
        type2 = bead_type_to_idx[tokens[1]]
        assert tokens[2] == "1"
        sigma = float(tokens[3])
        eps = float(tokens[4])

        epsilons[type1, type2] = eps
        sigmas[type1, type2] = sigma

        # FIXME: symmetry correct?
        epsilons[type2, type1] = eps
        sigmas[type2, type1] = sigma

    return epsilons, sigmas


def read_topology(fpath, use_m2=False):

    fpath = Path(fpath)
    assert fpath.exists()

    with open(fpath, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)

    # Get starting lines for different subsectionss
    start_moltype = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[moleculetype]":
            start_moltype = l_idx
            break
    assert start_moltype is not None

    start_atoms = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[atoms]":
            start_atoms = l_idx
            break
    assert start_atoms is not None

    start_bonds = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[bonds]":
            start_bonds = l_idx
            break
    assert start_bonds is not None

    start_angles = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[angles]":
            start_angles = l_idx
            break
    assert start_angles is not None

    ## Check correct ordering
    assert start_moltype < start_atoms
    assert start_atoms < start_bonds
    assert start_bonds < start_angles

    # Read molecules
    molecules = list()
    for l_idx in range(start_moltype + 1, start_atoms):
        line = lines[l_idx].strip()
        if not line:
            continue
        if line[0] == ";":
            continue

        tokens = line.split()
        assert len(tokens) == 2
        molname, nrexcl = tokens
        molecules.append(molname)
    assert len(molecules) == 1  # FIXME: assume 1 molecule for now

    # Read atoms
    atom_types = list()
    atom_charges = list()
    if use_m2:
        bead_type_mapper = bead_type_to_idx_m2
    else:
        bead_type_mapper = bead_type_to_idx
    for l_idx in range(start_atoms + 1, start_bonds):
        line = lines[l_idx].strip()
        if not line:
            continue
        if line[0] == ";":
            continue

        tokens = line.split()
        assert len(tokens) == 8 or len(tokens) == 7  # FIXME: is mass optional?

        atom_type_idx = bead_type_mapper[tokens[1]]
        atom_types.append(atom_type_idx)
        atom_charge = float(tokens[6])
        atom_charges.append(atom_charge)
    atom_types = onp.array(atom_types)
    atom_charges = onp.array(atom_charges)

    # Read bonds
    bonds = list()
    bond_type_idxs = list()
    if use_m2:
        bond_type_mapper = bond_type_to_idx_m2
    else:
        bond_type_mapper = bond_type_to_idx
    for l_idx in range(start_bonds + 1, start_angles):
        line = lines[l_idx].strip()
        if not line:
            continue
        if line[0] == ";":
            continue

        tokens = line.split()
        if use_m2:
            assert len(tokens) == 4
        else:
            assert len(tokens) == 3

        idx1 = int(tokens[0])
        assert idx1 >= 1
        idx2 = int(tokens[1])
        assert idx2 >= 1

        if use_m2:
            assert tokens[2] == "1"  # harmonic type
            bond_type = tokens[3]
        else:
            bond_type = tokens[2]
        bond_type_idxs.append(bond_type_mapper[bond_type])

        # Note: we change to 0-indexing
        bonds.append([idx1 - 1, idx2 - 1])
    bonds = onp.array(bonds)
    bond_type_idxs = onp.array(bond_type_idxs)

    # Read angles
    angles = list()
    angle_type_idxs = list()
    if use_m2:
        angle_type_mapper = angle_type_to_idx_m2
    else:
        angle_type_mapper = angle_type_to_idx
    for l_idx in range(start_angles + 1, n_lines):
        line = lines[l_idx].strip()
        if not line:
            continue
        if line[0] == ";":
            continue

        tokens = line.split()
        if use_m2:
            assert tokens[3] == "2"  # harmonic type
            assert len(tokens) == 5
        else:
            assert len(tokens) == 4

        idx1 = int(tokens[0])
        assert idx1 >= 1
        idx2 = int(tokens[1])
        assert idx2 >= 1
        idx3 = int(tokens[2])
        assert idx3 >= 1

        if use_m2:
            angle_type = tokens[4]
        else:
            angle_type = tokens[3]
        angle_type_idxs.append(angle_type_mapper[angle_type])

        # Note: we change to 0-indexing
        angles.append([idx1 - 1, idx2 - 1, idx3 - 1])
    angles = onp.array(angles)
    angle_type_idxs = onp.array(angle_type_idxs)

    return atom_types, atom_charges, bonds, angles, bond_type_idxs, angle_type_idxs


def get_unbonded_neighbors(n, bonded_neighbors):
    """
    Takes a set of bonded neighbors and returns the set
    of unbonded neighbors for a given `n` by enumerating
    all possibilities of pairings for `n` and removing
    the bonded neighbors
    """

    # First, set to all neighbors
    unbonded_neighbors = set(itertools.combinations(range(n), 2))

    # Then, remove all bonded neighbors
    unbonded_neighbors -= set(bonded_neighbors)

    rev_bonded_nbrs = set([(j, i) for (i, j) in bonded_neighbors])
    unbonded_neighbors -= rev_bonded_nbrs

    # Finally, remove identities (which shouldn't be in there in the first place)
    unbonded_neighbors -= set([(i, i) for i in range(n)])

    # Return as a list
    return list(unbonded_neighbors)  # return as list


all_bond_types = [
    # Alkanes/alkenes
    "b_C1_C1_mid",
    "b_C1_C4_mid",
    "b_C4_C1_mid",
    "b_C4_C4_mid",
    "b_C1_C1_end",
    "b_C4_C1_end",
    "b_C1_C4_end",
    "b_C4_C4_end",
    "b_C1_C1_mid_5long",
    "b_C1_C4_mid_5long",
    "b_C4_C1_mid_5long",
    "b_C4_C4_mid_5long",
    "b_SC1_C1_mid",
    "b_SC1_C4_mid",
    "b_SC4_C1_mid",
    "b_SC4_C4_mid",
    "b_SC1_C1_end",
    "b_SC4_C1_end",
    "b_SC1_C4_end",
    "b_SC4_C4_end",
    # Phospholipids
    "b_NC3_PO4_def",
    "b_NH3_PO4_def",
    "b_GL0_PO4_def",
    "b_CNO_PO4_def",
    "b_PO4_GL_def",
    "b_PO4_GL_def_long",
    "b_PO4_ET_def",
    "b_PO4_ET_def_long",
    "b_GL_OH_22_bmp",
    "b_PO4_OH_33_bmp",
    "b_OH_GL_33_bmp",
    "b_PO4_GL_cl",
    # Sphingomyelin and ceramide
    "b_PO4_OH1_sm",
    "b_PO4_AM2_sm",
    "b_OH1_AM2_sm",
    "b_OH1_SC4_sm",
    "b_AM2_SC1_sm",
    "b_AM2_C1_sm_5long",
    "b_COH_OH1_sm",
    "b_COH_AM2_sm",
    # GLYCEROL, MONOGLYCERIDES, DIGLYCERIDES, TRIGLYCERIDES
    "b_GL_GL_glyc",
    "b_GL_GL_glyc_long",
    "b_GL_C1_glyc_5long",
    "b_GL_C4_glyc_5long",
    "b_GL_SC1_glyc",
    "b_GL_SC4_glyc",
    "b_COH_GL_def",
    "b_COH_GL_def_long",
    "b_NC3_GL_def",
    "b_NC3_GL_def_long",
    "b_DOH_GL_def",
    # ETHERLIPIDS, PLASMALOGENS
    "b_ET_ET_ether",
    "b_GL_ET_plasm",
    "b_ET_GL_plasm",
    "b_ET_C1_ether_5long",
    "b_ET_C4_ether_5long",
    "b_ET_SC1_ether",
    "b_ET_SC4_ether",
    "b_ET_C1_plasm_5long",
    "b_ET_C4_plasm_5long",
    "b_ET_SC1_plasm",
    "b_ET_SC4_plasm",
    "b_GL_C1_plasm_5long",
    "b_GL_C4_plasm_5long",
    "b_GL_SC1_plasm",
    "b_GL_SC4_plasm",
    # Fatty Acids
    "b_COO_C1_fa_5long",
    "b_COO_C4_fa_5long",
    "b_COO_SC1_fa",
    "b_COO_SC4_fa",
]
bond_type_to_idx = {bond_type: idx for idx, bond_type in enumerate(all_bond_types)}


all_bond_types_m2 = [
    "mb_np",
    "mb_gp",
    "mb_pg1",
    "mb_pg2",
    "mb_pa",
    "mb_gg",
    "mb_aa",
    "mb_cc",
]
bond_type_to_idx_m2 = {
    bond_type: idx for idx, bond_type in enumerate(all_bond_types_m2)
}


all_angle_types = [
    # Alkanes/alkenes
    "a_C1_C1_C1_def",
    "a_C1_C4_C1_def",
    "a_C1_C1_C4_def",
    "a_C4_C1_C1_def",
    "a_C1_C4_C4_def",
    "a_C4_C4_C1_def",
    "a_C4_C1_C4_def",
    "a_C4_C4_C4_def",
    # Phospholipids
    "a_NC3_PO4_GL_def",
    "a_NC3_PO4_ET_def",
    "a_NH3_PO4_GL_def",
    "a_NH3_PO4_ET_def",
    "a_GL0_PO4_GL_def",
    "a_GL0_PO4_ET_def",
    "a_CNO_PO4_GL_def",
    "a_CNO_PO4_ET_def",
    "a_PS1_PS2_PO4_def",
    "a_PS2_PO4_GL_def",
    "a_PS2_PO4_ET_def",
    "a_PO4_GL_C_def",
    "a_PO4_ET_C_def",
    "a_PO4_GL_OH_22_bmp",
    "a_OH_GL_C_22_bmp",
    "a_PO4_OH_GL_33_bmp",
    "a_OH_GL_C_33_bmp",
    "a_PO4_GL_PO4_cl",
    "a_GL_PO4_GL_cl",
    # Sphingomyelin and ceramide
    "a_NC3_PO4_OH1_sm",
    "a_PO4_OH1_AM2_sm",
    "a_PO4_OH1_C_sm",
    "a_OH1_AM2_C_sm",
    "a_AM2_OH1_C_sm",
    "a_AM2_C1_C1_sm",
    "a_AM2_C1_C4_sm",
    "a_AM2_C4_C1_sm",
    "a_AM2_C4_C4_sm",
    "a_OH1_C1_C1_sm",
    "a_COH_OH1_C_cera",
    # GLYCEROL, MONOGLYCERIDES, DIGLYCERIDES, TRIGLYCERIDES
    "a_GL_GL_C_glyc",
    "a_GL_C1_C1_glyc",
    "a_GL_C1_C4_glyc",
    "a_GL_C4_C1_glyc",
    "a_GL_C4_C4_glyc",
    "a_COH_GL_C_def",
    "a_DOH_GL_C_def",
    # ETHERLIPIDS, PLASMALOGENS
    "a_ET_ET_C_ether",
    "a_ET_C1_C1_ether",
    "a_ET_C1_C4_ether",
    "a_ET_C4_C1_ether",
    "a_ET_C4_C4_ether",
    "a_GL_ET_C_plasm",
    "a_ET_GL_C_plasm",
    "a_ET_C1_C1_plasm",
    "a_ET_C1_C4_plasm",
    "a_ET_C4_C1_plasm",
    "a_ET_C4_C4_plasm",
    "a_GL_C1_C1_plasm",
    "a_GL_C1_C4_plasm",
    "a_GL_C4_C1_plasm",
    "a_GL_C4_C4_plasm",
    # Fatty Acids
    "a_COO_C1_C1_fa",
    "a_COO_C1_C4_fa",
    "a_COO_C4_C1_fa",
    "a_COO_C4_C4_fa",
    # Misplaced phospholipid
    "a_NC3_GL_C_def",
]
angle_type_to_idx = {angle_type: idx for idx, angle_type in enumerate(all_angle_types)}


all_angle_types_m2 = [
    "ma_pgg",
    "ma_paa",
    "ma_pgc",
    "ma_pac",
    "ma_gcc",
    "ma_acc",
    "ma_adc",
    "ma_ccc",
    "ma_cdc",
    "ma_ddd",
]
angle_type_to_idx_m2 = {
    angle_type: idx for idx, angle_type in enumerate(all_angle_types_m2)
}


def read_bond_angle_params(
    fpath="jax_lipids/data/params/m3/martini_v3.0.0_ffbonded_v2_openbeta.itp",
    use_m2=False,
):
    fpath = Path(fpath)
    assert fpath.exists()

    with open(fpath, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)

    is_bond = True

    angle_dict = dict()  # maps name to (angle, k)
    bond_dict = dict()  # maps name to (length, k)
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line == "[ bondtypes ]":
            is_bond = True
        elif stripped_line == "[ angletypes ]":
            is_bond = False
        elif stripped_line[:7] == "#define":
            if ";" in stripped_line:
                stripped_line = stripped_line.split(";", 1)[0]
                assert stripped_line[:7] == "#define"
            tokens = stripped_line.split()
            name = tokens[1]

            if is_bond and name != "a_NC3_GL_C_def":
                if use_m2:
                    assert "mb" in name
                    length = float(tokens[2])
                    k = float(tokens[3])
                else:
                    if name != "a_PO4_OH_GL_33_bmp":
                        assert tokens[2] == "1"  # harmonic type
                    length = float(tokens[3])
                    k = float(tokens[4])
                    assert name[:2] == "b_"
                bond_dict[name] = (length, k)
            else:
                if use_m2:
                    assert "ma" in name
                    angle = onp.deg2rad(float(tokens[2]))
                    k = float(tokens[3])
                else:
                    angle = onp.deg2rad(float(tokens[3]))
                    k = float(tokens[4])
                    assert name[:2] == "a_"
                angle_dict[name] = (angle, k)

    bond_ks = list()
    bond_lengths = list()
    if use_m2:
        curr_bond_types = all_bond_types_m2
    else:
        curr_bond_types = all_bond_types
    for bond_name in curr_bond_types:
        assert bond_name in bond_dict

        length = bond_dict[bond_name][0]
        k = bond_dict[bond_name][1]

        bond_ks.append(k)
        bond_lengths.append(length)
    bond_ks = onp.array(bond_ks)
    bond_lengths = onp.array(bond_lengths)

    angle_ks = list()
    angle_angles = list()
    if use_m2:
        curr_angle_types = all_angle_types_m2
    else:
        curr_angle_types = all_angle_types
    for angle_name in curr_angle_types:
        assert angle_name in angle_dict

        angle = angle_dict[angle_name][0]
        k = angle_dict[angle_name][1]

        angle_ks.append(k)
        angle_angles.append(angle)
    angle_ks = onp.array(angle_ks)
    angle_angles = onp.array(angle_angles)

    return bond_ks, bond_lengths, angle_ks, angle_angles


def update_bond_angle_params(
    bond_ks,
    bond_lengths,
    angle_ks,
    angle_angles,
    fpath="jax_lipids/data/params/martini_v3.0.0_ffbonded_v2_openbeta.itp",
    use_m2=False,
):
    """
    Writes the updated bond and angle parameters back to the itp file.

    Args:
        bond_ks: Array of bond force constants
        bond_lengths: Array of equilibrium bond lengths
        angle_ks: Array of angle force constants
        angle_angles: Array of equilibrium angles (in radians)
        fpath: Path to the parameter file to update
    """
    # Convert angles to degrees for writing
    angle_angles_deg = onp.rad2deg(angle_angles)

    # Create parameter dictionaries
    if use_m2:
        curr_bond_types = all_bond_types_m2
    else:
        curr_bond_types = all_bond_types
    bond_params = {
        name: (length, k)
        for name, length, k in zip(curr_bond_types, bond_lengths, bond_ks)
    }

    if use_m2:
        curr_angle_types = all_angle_types_m2
    else:
        curr_angle_types = all_angle_types
    angle_params = {
        name: (angle, k)
        for name, angle, k in zip(curr_angle_types, angle_angles_deg, angle_ks)
    }

    # Track if any parameters have changed
    any_changes = False

    # Read existing file
    with open(fpath, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line[0] == "[" or stripped_line[0] == ";":
            new_lines.append(line)
            continue

        if stripped_line[:7] == "#define":
            tokens = stripped_line.split()
            if len(tokens) >= 5:  # Make sure we have enough tokens
                name = tokens[1]

                if name in bond_params:
                    # Update bond parameters
                    length, k = bond_params[name]
                    if use_m2:
                        length_idx = 2
                        k_idx = 3
                    else:
                        length_idx = 3
                        k_idx = 4
                    current_length = float(tokens[length_idx])
                    current_k = float(tokens[k_idx])

                    # Only update if values have changed
                    if abs(length - current_length) > 1e-6 or abs(k - current_k) > 1e-6:
                        # new_line = line.replace(tokens[3], f"{length:.3f}").replace(tokens[4], f"{k:.1f}")
                        new_line = line.replace(
                            tokens[length_idx], f"{length:.6f}"
                        ).replace(tokens[k_idx], f"{k:.6f}")
                        new_lines.append(new_line)
                        any_changes = True
                    else:
                        new_lines.append(line)

                elif name in angle_params:
                    # Update angle parameters
                    angle, k = angle_params[name]
                    if use_m2:
                        angle_idx = 2
                        k_idx = 3
                    else:
                        angle_idx = 3
                        k_idx = 4
                    current_angle = float(tokens[angle_idx])
                    current_k = float(tokens[k_idx])

                    # Only update if values have changed
                    if abs(angle - current_angle) > 1e-6 or abs(k - current_k) > 1e-6:
                        new_line = line.replace(
                            tokens[angle_idx], f"{angle:.1f}"
                        ).replace(tokens[k_idx], f"{k:.1f}")
                        # new_line = line.replace(tokens[3], f"{angle:.6f}").replace(tokens[4], f"{k:.6f}")
                        new_lines.append(new_line)
                        any_changes = True
                    else:
                        new_lines.append(line)

                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Only write to file if parameters have changed
    if any_changes:
        with open(fpath, "w") as f:
            f.writelines(new_lines)


def update_lj_params(
    lj_epsilons, lj_sigmas, fpath="jax_lipids/data/params/martini_v3.0.0.itp"
):
    """
    Updates the Lennard-Jones parameters in the given parameter file.
    Only writes to file if parameters have actually changed.

    Args:
        lj_epsilons: Array of Lennard-Jones epsilon values (n_bead_types x n_bead_types)
        lj_sigmas: Array of Lennard-Jones sigma values (n_bead_types x n_bead_types)
        fpath: Path to the parameter file to update
    """
    # Read file lines
    with open(fpath, "r") as f:
        lines = f.readlines()

    # Find the nonbonded section
    start_nonbonded = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[ nonbond_params ]":
            start_nonbonded = l_idx
            break

    if start_nonbonded is None:
        raise ValueError("Could not find [ nonbond_params ] section in file")

    # Track if any parameters have changed
    any_changes = False

    # Create a map of existing parameter lines and values for quick lookup
    param_map = {}
    l_idx = start_nonbonded + 1
    while l_idx < len(lines):
        line = lines[l_idx].strip()
        if not line or line.startswith("["):
            break
        if line and not line.startswith(";"):  # Skip empty lines and comments
            tokens = line.split()
            if len(tokens) == 5:
                type1, type2 = tokens[0], tokens[1]
                current_sigma = float(tokens[3])
                current_epsilon = float(tokens[4])
                param_map[(type1, type2)] = (l_idx, current_sigma, current_epsilon)
        l_idx += 1

    # Update only changed parameters in-place
    for i in range(n_bead_types):
        for j in range(i, n_bead_types):  # Only upper triangle

            type1 = idx_to_bead_type[i]
            type2 = idx_to_bead_type[j]
            sigma = lj_sigmas[i, j]
            epsilon = lj_epsilons[i, j]

            # Find existing line
            key_forward = (type1, type2)
            key_rev = (type2, type1)
            for key in [key_forward, key_rev]:
                if key in param_map:
                    line_idx, current_sigma, current_epsilon = param_map[key]

                    # Only update if values have changed (within numerical precision)
                    if (
                        abs(sigma - current_sigma) > 1e-10
                        or abs(epsilon - current_epsilon) > 1e-10
                    ):
                        padding1 = " " * (6 - len(type1))
                        padding2 = " " * (6 - len(type2))
                        lines[line_idx] = (
                            f"{padding1}{type1}{padding2}{type2}  1 {sigma:.6e}    {epsilon:.6e}\n"
                        )
                        any_changes = True

    # Only write to file if parameters have changed
    if any_changes:
        with open(fpath, "w") as f:
            f.writelines(lines)


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


if __name__ == "__main__":
    # read_lj_params()

    top_fpath = "jax_lipids/data/sys_defs/POPC.itp"
    read_topology(top_fpath)

    read_bond_angle_params()


def read_lj_params_dry(fpath="data/params/m2/dry_martini_v2.1.itp"):
    fpath = Path(fpath)
    assert fpath.exists()

    with open(fpath, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)

    # Do a first full pass of the file to get all constants
    constants_mapper = dict()
    name_order = list()
    for l_idx, line in enumerate(lines):
        if line[:7] == "#define":
            tokens = line.split()
            assert tokens[0] == "#define"
            name = tokens[1]
            assert name[:3] == "dm_"
            sigma = float(tokens[2])
            epsilon = float(tokens[3])
            constants_mapper[name] = (sigma, epsilon)

            name_order.append(name)

    start_nonbonded = None
    for l_idx, line in enumerate(lines):
        if line.strip() == "[ nonbond_params ]":
            start_nonbonded = l_idx
            break

    assert start_nonbonded is not None
    epsilons = onp.zeros((n_bead_types_m2, n_bead_types_m2))  # FIXME: default zero?
    sigmas = onp.zeros((n_bead_types_m2, n_bead_types_m2))  # FIXME: default zero?

    is_set = onp.zeros((n_bead_types_m2, n_bead_types_m2), dtype=onp.int32)
    constant_idxs = onp.zeros((n_bead_types_m2, n_bead_types_m2), dtype=onp.int32)
    for l_idx in range(start_nonbonded + 1, n_lines):
        line = lines[l_idx]
        if not line.strip() or line[0] == ";":
            continue
        tokens = line.strip().split()
        assert len(tokens) == 4

        assert tokens[2] == "1"
        name = tokens[3]
        sigma, eps = constants_mapper[name]

        type1 = bead_type_to_idx_m2[tokens[0]]
        type2 = bead_type_to_idx_m2[tokens[1]]

        epsilons[type1, type2] = eps
        sigmas[type1, type2] = sigma

        # FIXME: symmetry correct?
        epsilons[type2, type1] = eps
        sigmas[type2, type1] = sigma

        is_set[type1, type2] = 1
        is_set[type2, type1] = 1
        constant_idx = name_order.index(name)
        constant_idxs[type1, type2] = constant_idx
        constant_idxs[type2, type1] = constant_idx

    sigma_order = onp.array([constants_mapper[name][0] for name in name_order])
    epsilon_order = onp.array([constants_mapper[name][1] for name in name_order])

    # return epsilons, sigmas
    return (
        epsilons,
        sigmas,
        name_order,
        sigma_order,
        epsilon_order,
        is_set,
        constant_idxs,
    )


def update_lj_params_dry(
    lj_epsilons,
    lj_sigmas,
    lj_name_order,
    fpath="jax_lipids/data/params/m2/dry_martini_v2.1.itp",
):
    """
    Updates the Lennard-Jones parameters in the given parameter file.
    Only writes to file if parameters have actually changed.

    Args:
        lj_epsilons: Array of Lennard-Jones epsilon values (n_bead_types x n_bead_types)
        lj_sigmas: Array of Lennard-Jones sigma values (n_bead_types x n_bead_types)
        fpath: Path to the parameter file to update
    """
    # Read file lines
    with open(fpath, "r") as f:
        lines = f.readlines()

    # Find the nonbonded section
    order_idx = 0
    for l_idx, line in enumerate(lines):
        if line[:7] == "#define":
            tokens = line.split()
            assert tokens[0] == "#define"
            name = tokens[1]
            assert name[:3] == "dm_"
            assert name == lj_name_order[order_idx]
            epsilon = lj_epsilons[order_idx]
            sigma = lj_sigmas[order_idx]
            order_idx += 1

            padding1 = " " * (14 - len(name))

            lines[l_idx] = f"#define {name}{padding1} {sigma:.6e} {epsilon:.6e}\n"

    with open(fpath, "w") as f:
        f.writelines(lines)


def fill_template(
    template_fpath,
    new_fpath,
    n_prod_steps=None,
    n_eq_steps=None,
    sample_every=None,
    sym_temp=None,
    # For melting temperatures
    gel_temp=None,
    liquid_temp=None,
):
    with open(template_fpath, "r") as f:
        template = f.read()

    filled = deepcopy(template)

    # Replace keywords with values
    if n_prod_steps is not None:
        filled = template.replace("N_PROD_STEPS", str(n_prod_steps))
    if n_eq_steps is not None:
        filled = filled.replace("N_EQ_STEPS", str(n_eq_steps))
    if sample_every is not None:
        filled = filled.replace("SAMPLE_EVERY", str(sample_every))
    if sym_temp is not None:
        filled = filled.replace("TEMPERATURE", str(sym_temp))
    if gel_temp is not None:
        filled = filled.replace("GEL_TEMP", str(gel_temp))
    if liquid_temp is not None:
        filled = filled.replace("LIQUID_TEMP", str(liquid_temp))

    with open(new_fpath, "w") as f:
        f.write(filled)
