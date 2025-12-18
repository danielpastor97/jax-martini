from pathlib import Path
import argparse
import numpy as onp
import random
import time
import shutil
import shlex
from copy import deepcopy
import subprocess
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from tqdm import tqdm
import os
import MDAnalysis as mda
import json
import pickle
import ray
import socket
from collections import Counter
import gc

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
from jax import value_and_grad, vmap
import optax

from jax_martini import utils, energy, checkpoint
from jax_martini.observables import bonded_distributions, wasserstein, structural, tm


def get_kt(t_kelvin):
    """Converts a temperature in Kelvin to kT in simulation units."""
    return t_kelvin * 0.0083144621


def percent_deviation(measured_value, true_value):
    return jnp.abs(measured_value - true_value) / true_value * 100


all_subdir_names = {
    "DLPC": "DUPC",
    "DUPC": "DUPC",
    "DMPC": "DMPC",
    "DOPC": "DOPC",
    "DPPC": "DPPC",
    "DSPC": "DSPC",
    "PDPC": "PDPC",
    "POPC": "POPC",
    "SDPC": "SDPC",
}


def run(args):

    ray_num_cpus = args["ray_num_cpus"]
    ray_num_gpus = args["ray_num_gpus"]

    @ray.remote(num_gpus=ray_num_gpus, num_cpus=ray_num_cpus)
    def run_subprocess_ray(cmd_list):
        time.sleep(1)

        node_devices = jax.devices()

        hostname = socket.gethostbyname(socket.gethostname())

        start = time.time()
        p = subprocess.Popen(cmd_list)
        p.wait()
        end = time.time()

        rc = p.returncode
        return rc, end - start, hostname, [str(d) for d in node_devices]

    swarmcg_w0 = args["swarmcg_w0"]
    swarmcg_w1 = args["swarmcg_w1"]
    swarmcg_w2 = args["swarmcg_w2"]
    swarmcg_w3 = args["swarmcg_w3"]
    swarmcg_eps = args["swarmcg_eps"]
    swarmcg_eps_tm = args["swarmcg_eps_tm"]

    checkpoint_every = args["checkpoint_every"]
    if checkpoint_every is None:
        scan = jax.lax.scan
    else:
        scan = functools.partial(
            checkpoint.checkpoint_scan, checkpoint_every=checkpoint_every
        )

    optimizer_type = args["optimizer_type"]

    random.seed(args["seed"])

    continue_opt = args["continue_opt"]
    serial = args["serial"]
    use_ray = args["use_ray"]
    if use_ray:
        assert not serial
    elif serial:
        assert not use_ray

    if use_ray:
        if "ip_head" in os.environ:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init()

    run_name = args["run_name"]
    gromacs_path = args["gromacs_path"]

    save_every = args["save_every"]

    targets_dir = args["targets_dir"]
    targets_dir = Path(targets_dir)
    assert targets_dir.exists() and targets_dir.is_dir()

    target_fpaths = [
        fpath for fpath in targets_dir.glob("**/*") if fpath.suffix == ".yaml"
    ]
    n_apl_targets = 0
    n_dhh_targets = 0
    n_tm_targets = 0
    target_info = dict()
    lipid_types = list()
    membrane_sim_temps = dict()
    all_bond_types = list()
    all_angle_types = list()
    target_tms = {}
    target_tm_sim_temps = {}
    for target_fpath in target_fpaths:
        assert Path(target_fpath).exists()
        with open(target_fpath, "r") as f:
            all_target_data = json.load(f)

        for temp, temp_data in all_target_data["Temperature"].items():
            if (
                "Thicknesses" in temp_data
                and "DHH (peak-peak)" in temp_data["Thicknesses"]
            ):
                n_dhh_targets += 1

            if "APL" in temp_data:
                n_apl_targets += 1

        sys_name = all_target_data["system"]

        lipid_types.append(sys_name)

        if "Tm" in all_target_data:
            target_tms[sys_name] = all_target_data["Tm"]

            assert "Tm_sim_temps" in all_target_data
            tm_sim_temps = all_target_data["Tm_sim_temps"]

            assert len(set((tm_sim_temps))) == len(tm_sim_temps)  # no duplicates
            assert list(tm_sim_temps)  # non empty

            target_tm_sim_temps[sys_name] = list(tm_sim_temps)

            n_tm_targets += 1

        for target_temp, target_data in all_target_data["Temperature"].items():

            assert target_temp.isdigit()  # For now, assume is integer

            sys_t_kelvin = int(target_temp)

            if "Bonds" in target_data:
                for lipid_type, l_bonds in target_data["Bonds"].items():
                    for l_bond_name, l_bond_target_data in l_bonds.items():
                        all_bond_types.append(l_bond_name)
                        if "distribution" in l_bond_target_data:
                            l_bond_distribution = onp.load(
                                l_bond_target_data["distribution"]
                            )
                            if len(l_bond_distribution.shape) != 1:
                                raise RuntimeError(f"Invalid distribution shape")
                            l_bond_distribution = jnp.array(l_bond_distribution)
                            target_data["Bonds"][lipid_type][l_bond_name][
                                "distribution"
                            ] = l_bond_distribution
                        else:
                            raise RuntimeError(f"Missing distribution path.")

            if "Angles" in target_data:
                for lipid_type, l_angles in target_data["Angles"].items():
                    for l_angle_name, l_angle_target_data in l_angles.items():
                        all_angle_types.append(l_angle_name)
                        if "distribution" in l_angle_target_data:
                            l_angle_distribution = onp.load(
                                l_angle_target_data["distribution"]
                            )
                            if len(l_angle_distribution.shape) != 1:
                                raise RuntimeError(f"Invalid distribution shape")
                            l_angle_distribution = jnp.array(l_angle_distribution)
                            target_data["Angles"][lipid_type][l_angle_name][
                                "distribution"
                            ] = l_angle_distribution
                        else:
                            raise RuntimeError(f"Missing distribution path.")

            if sys_name in target_info:
                target_info[sys_name][sys_t_kelvin] = target_data
            else:
                target_info[sys_name] = {sys_t_kelvin: target_data}

            if sys_name in membrane_sim_temps:
                membrane_sim_temps[sys_name].append(sys_t_kelvin)
            else:
                membrane_sim_temps[sys_name] = [sys_t_kelvin]

        if "Tm" in all_target_data:
            for tm_sim_temp in tm_sim_temps:
                assert (sys_name not in target_info) or (
                    tm_sim_temp not in target_info[sys_name]
                )  # Avoids conflicts about which simulation to run
                if sys_name not in target_info:
                    target_info[sys_name] = {tm_sim_temp: {}}
                else:
                    target_info[sys_name][tm_sim_temp] = {}
                assert (sys_name not in membrane_sim_temps) or (
                    tm_sim_temp not in membrane_sim_temps[sys_name]
                )
                if sys_name not in membrane_sim_temps:
                    membrane_sim_temps[sys_name] = [tm_sim_temp]
                else:
                    membrane_sim_temps[sys_name].append(tm_sim_temp)

            membrane_sim_temps[sys_name] = sorted(membrane_sim_temps[sys_name])

    all_bond_types = list(set(all_bond_types))
    n_bond_types = len(all_bond_types)
    all_angle_types = list(set(all_angle_types))
    n_angle_types = len(all_angle_types)
    membrane_sims = list(membrane_sim_temps.keys())
    lipid_types = list(set(lipid_types))

    n_eq_steps_per_sim = args["n_eq_steps_per_sim"]
    n_steps_per_sim = args["n_steps_per_sim"]
    sample_every = args["sample_every"]
    assert n_steps_per_sim % sample_every == 0
    n_ref_states = int(n_steps_per_sim // sample_every)

    tm_n_eq_steps_per_sim = args["tm_n_eq_steps_per_sim"]
    tm_n_biphasic_eq_steps_per_sim = args["tm_n_biphasic_eq_steps_per_sim"]
    tm_n_steps_per_sim = args["tm_n_steps_per_sim"]
    tm_sample_every = args["tm_sample_every"]
    assert tm_n_steps_per_sim % tm_sample_every == 0
    tm_n_ref_states = int(tm_n_steps_per_sim // tm_sample_every)

    assert tm_n_ref_states == n_ref_states

    n_iters = args["n_iters"]
    lr = args["lr"]
    min_neff_factor = args["min_neff_factor"]
    max_approx_iters = args["max_approx_iters"]

    min_n_eff = int(n_ref_states * min_neff_factor)

    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path(args["output_dir"])
    data_dir = Path("data/")
    tm_data_dir = Path("data/membrane_init/m3_biphasic/")

    run_dir = output_dir / run_name
    ref_traj_dir = run_dir / "ref_traj"
    sigmoid_dir = run_dir / "sigmoid"
    log_dir = run_dir / "log"
    obj_dir = run_dir / "obj"
    last_dir = obj_dir / "last"
    if continue_opt:
        assert run_dir.exists()
    else:
        run_dir.mkdir(parents=False, exist_ok=False)
        ref_traj_dir.mkdir(parents=False, exist_ok=False)
        log_dir.mkdir(parents=False, exist_ok=False)
        obj_dir.mkdir(parents=False, exist_ok=False)
        last_dir.mkdir(parents=False, exist_ok=False)
        sigmoid_dir.mkdir(parents=False, exist_ok=False)

    loss_path = log_dir / "loss.txt"
    iter_params_path = log_dir / "iter_params.txt"
    resample_times_path = log_dir / "resample_times.txt"
    grads_path = log_dir / "grads.txt"
    thicknesses_path = log_dir / "thicknesses.txt"
    bonds_path = log_dir / "bonds.txt"
    angles_path = log_dir / "angles.txt"
    areas_path = log_dir / "areas.txt"
    all_losses_path = log_dir / "all_losses.txt"
    global_time_path = log_dir / "global_time.txt"
    dhh_loss_path = log_dir / "dhh_loss.txt"
    apl_loss_path = log_dir / "apl_loss.txt"
    otb_loss_path = log_dir / "otb_loss.txt"
    tm_loss_path = log_dir / "tm_loss.txt"
    tms_path = log_dir / "tms.txt"
    tm_sim_temps_path = log_dir / "tm_sim_temps.txt"
    tm_sim_apls_path = log_dir / "tm_sim_apls.txt"
    n_effs_path = log_dir / "n_effs.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the topology of each membrane type
    ## Load the topology of each type of lipid
    all_single_lipid_topologies = {}
    for l_type in lipid_types:

        itp_fpath = data_dir / "lipid_defs/m3" / f"{l_type}.itp"
        atom_types, _, bond_pairs, angle_triplets, bond_type_idxs, angle_type_idxs = (
            utils.read_topology(itp_fpath)
        )

        rel_bond_pair_to_bond_type = dict()
        ordered_bond_type_idxs = list()
        for rel_bond_pair, bond_type_idx in zip(bond_pairs, bond_type_idxs):
            i, j = rel_bond_pair
            rel_bond_pair_to_bond_type[(i, j)] = bond_type_idx

            ordered_bond_type_idxs.append(bond_type_idx)

        rel_angle_triplet_to_angle_type = dict()
        ordered_angle_type_idxs = list()
        for rel_angle_triplet, angle_type_idx in zip(angle_triplets, angle_type_idxs):
            i, j, k = rel_angle_triplet
            rel_angle_triplet_to_angle_type[(i, j, k)] = angle_type_idx

            ordered_angle_type_idxs.append(angle_type_idx)

        all_single_lipid_topologies[l_type] = {
            "atom_types": atom_types,
            "rel_bond_pair_to_bond_type": rel_bond_pair_to_bond_type,
            "rel_angle_triplet_to_angle_type": rel_angle_triplet_to_angle_type,
            "ordered_bond_type_idxs": ordered_bond_type_idxs,
            "ordered_angle_type_idxs": ordered_angle_type_idxs,
        }

    def get_angle_triplet_info(repr_u):
        m_sim_angle_triplets = repr_u.atoms.angles.__dict__["_bix"]  # 0-indexed
        m_sims_angle_type_idxs = list()
        for angle_triplet in m_sim_angle_triplets:
            i, j, k = angle_triplet

            # Get the index of the lipid molecule for i and j
            i_mol_idx = repr_u.atoms.resindices[i]
            j_mol_idx = repr_u.atoms.resindices[j]
            k_mol_idx = repr_u.atoms.resindices[k]
            ## Check that they're the same
            assert i_mol_idx == j_mol_idx
            assert i_mol_idx == k_mol_idx

            # Get the index of that lipid
            mol_idx = i_mol_idx

            # Get the type of that lipid
            mol_type = repr_u.atoms.residues[mol_idx].resname

            # Get the absolute indices (0-indexed) corresponding that lipid
            mol_indices = repr_u.atoms.residues[mol_idx].atoms.indices

            # Get the relative indices of i and j in the lipid
            ## Note: assumes proper ordering
            rel_angle_indices = angle_triplet - mol_indices.min()
            rel_angle_indices = tuple(rel_angle_indices)

            # Get the bond type
            angle_type = all_single_lipid_topologies[mol_type][
                "rel_angle_triplet_to_angle_type"
            ][rel_angle_indices]
            m_sims_angle_type_idxs.append(angle_type)

        m_sim_angle_triplets = jnp.array(m_sim_angle_triplets, dtype=jnp.int32)
        m_sims_angle_type_idxs = jnp.array(m_sims_angle_type_idxs, dtype=jnp.int32)

        return m_sim_angle_triplets, m_sims_angle_type_idxs

    def get_bond_pair_info(repr_u):
        m_sim_bond_pairs = repr_u.atoms.bonds.__dict__["_bix"]  # 0-indexed
        m_sims_bond_type_idxs = list()
        for bond_pair in m_sim_bond_pairs:
            i, j = bond_pair

            # Get the index of the lipid molecule for i and j
            i_mol_idx = repr_u.atoms.resindices[i]
            j_mol_idx = repr_u.atoms.resindices[j]
            ## Check that they're the same
            assert i_mol_idx == j_mol_idx

            # Get the index of that lipid
            mol_idx = i_mol_idx

            # Get the type of that lipid
            mol_type = repr_u.atoms.residues[mol_idx].resname

            # Get the absolute indices (0-indexed) corresponding that lipid
            mol_indices = repr_u.atoms.residues[mol_idx].atoms.indices

            # Get the relative indices of i and j in the lipid
            ## Note: assumes proper ordering
            rel_bond_indices = bond_pair - mol_indices.min()
            rel_bond_indices = tuple(rel_bond_indices)

            # Get the bond type
            bond_type = all_single_lipid_topologies[mol_type][
                "rel_bond_pair_to_bond_type"
            ][rel_bond_indices]
            m_sims_bond_type_idxs.append(bond_type)
        m_sims_bond_type_idxs = jnp.array(m_sims_bond_type_idxs, dtype=jnp.int32)
        m_sim_bond_pairs = jnp.array(m_sim_bond_pairs, dtype=jnp.int32)

        return m_sim_bond_pairs, m_sims_bond_type_idxs

    def populate_bond_angle_info(
        m_sim,
        repr_t_kelvin,
        all_m_sim_copy_number,  # will modify in place
    ):

        subdir_name = all_subdir_names[m_sim]
        repr_tpr_path = (
            data_dir
            / "membrane_init"
            / "m3"
            / subdir_name
            / f"{repr_t_kelvin}K"
            / "representative.tpr"
        )
        repr_u = mda.Universe(repr_tpr_path)

        # Get bonds dict
        bonds_dict_mapper = {}
        for resname in onp.unique(repr_u.residues.resnames):
            bonds_dict_mapper[resname] = []
            res_sel = repr_u.select_atoms(f"resname {resname}")
            for bonds in res_sel.atoms.bonds:
                bond = bonds.atoms.names.tolist()
                if (
                    bond not in bonds_dict_mapper[resname]
                    and bond[::-1] not in bonds_dict_mapper[resname]
                ):
                    bonds_dict_mapper[resname].append(bond)
        bonds_dict_mapper = {
            k: v for k, v in bonds_dict_mapper.items() if v
        }  

        # Get copy numbers
        all_m_sim_copy_number[m_sim] = dict()
        lipid_name_counts = dict(Counter([res.resname for res in repr_u.residues]))
        del lipid_name_counts["W"]
        # del lipid_name_counts["ION"]
        for l_type, l_type_count in lipid_name_counts.items():
            all_m_sim_copy_number[m_sim][l_type] = l_type_count

        # Get angles dict
        angles_dict_mapper = {}
        for resname in onp.unique(repr_u.residues.resnames):
            angles_dict_mapper[resname] = []
            res_sel = repr_u.select_atoms(f"resname {resname}")
            for angles in res_sel.atoms.angles:
                angle = angles.atoms.names.tolist()
                if (
                    angle not in angles_dict_mapper[resname]
                    and angle[::-1] not in angles_dict_mapper[resname]
                ):
                    angles_dict_mapper[resname].append(angle)
        angles_dict_mapper = {k: v for k, v in angles_dict_mapper.items() if v}

        # Get bond/angle pairs/triplets
        ## Bonds
        m_sim_bond_pairs, m_sims_bond_type_idxs = get_bond_pair_info(repr_u)

        ## Angles
        m_sim_angle_triplets, m_sims_angle_type_idxs = get_angle_triplet_info(repr_u)

        return (
            bonds_dict_mapper,
            angles_dict_mapper,
            m_sim_bond_pairs,
            m_sims_bond_type_idxs,
            m_sim_angle_triplets,
            m_sims_angle_type_idxs,
        )

    all_m_sim_bond_angle_info = dict()
    all_m_sim_copy_number = dict()
    all_bonds_dict_mapper = dict()
    all_angles_dict_mapper = dict()
    for m_sim in membrane_sims:

        found_not_tm_temp = False
        for repr_t_kelvin in sorted(membrane_sim_temps[m_sim]):
            is_tm_temp = (m_sim in target_tm_sim_temps) and (
                repr_t_kelvin in target_tm_sim_temps[m_sim]
            )
            if not is_tm_temp:
                found_not_tm_temp = True
                break

        if not found_not_tm_temp:
            continue

        (
            bonds_dict_mapper,
            angles_dict_mapper,
            m_sim_bond_pairs,
            m_sims_bond_type_idxs,
            m_sim_angle_triplets,
            m_sims_angle_type_idxs,
        ) = populate_bond_angle_info(
            m_sim,
            repr_t_kelvin,
            all_m_sim_copy_number,  # will modify in place
        )

        all_bonds_dict_mapper[m_sim] = deepcopy(bonds_dict_mapper)
        all_angles_dict_mapper[m_sim] = deepcopy(angles_dict_mapper)
        all_m_sim_bond_angle_info[m_sim] = {
            "bond_pairs": m_sim_bond_pairs,
            "bond_type_idxs": m_sims_bond_type_idxs,
            "angle_triplets": m_sim_angle_triplets,
            "angle_type_idxs": m_sims_angle_type_idxs,
        }

    tm_all_m_sim_bond_angle_info = dict()
    for m_sim, tm_sim_temps in target_tm_sim_temps.items():

        repr_t_kelvin = tm_sim_temps[0]

        subdir_name = all_subdir_names[m_sim]
        repr_tpr_path = tm_data_dir / subdir_name / "representative.tpr"
        repr_u = mda.Universe(repr_tpr_path)

        ## Bonds
        m_sim_bond_pairs, m_sims_bond_type_idxs = get_bond_pair_info(repr_u)

        ## Angles
        m_sim_angle_triplets, m_sims_angle_type_idxs = get_angle_triplet_info(repr_u)

        tm_all_m_sim_bond_angle_info[m_sim] = {
            "bond_pairs": m_sim_bond_pairs,
            "bond_type_idxs": m_sims_bond_type_idxs,
            "angle_triplets": m_sim_angle_triplets,
            "angle_type_idxs": m_sims_angle_type_idxs,
        }

    def get_bond_angle_energy_fn(
        m_sim, bond_ks, bond_lengths, angle_ks, angle_theta0s, is_tm_temp=False
    ):
        if is_tm_temp:
            bonds = tm_all_m_sim_bond_angle_info[m_sim]["bond_pairs"]
            bond_type_idxs = tm_all_m_sim_bond_angle_info[m_sim]["bond_type_idxs"]
        else:
            bonds = all_m_sim_bond_angle_info[m_sim]["bond_pairs"]
            bond_type_idxs = all_m_sim_bond_angle_info[m_sim]["bond_type_idxs"]
        mapped_bond_fn = energy.get_mapped_bond_fn(
            bonds,
            bond_type_idxs,
            bond_ks,
            bond_lengths,
        )

        if is_tm_temp:
            angles = tm_all_m_sim_bond_angle_info[m_sim]["angle_triplets"]
            angle_type_idxs = tm_all_m_sim_bond_angle_info[m_sim]["angle_type_idxs"]
        else:
            angles = all_m_sim_bond_angle_info[m_sim]["angle_triplets"]
            angle_type_idxs = all_m_sim_bond_angle_info[m_sim]["angle_type_idxs"]
        mapped_angle_fn = energy.get_mapped_angle_fn(
            angles,
            angle_type_idxs,
            angle_ks,
            angle_theta0s,
        )

        def bond_angle_energy_fn(R, box_dimensions):
            R_nm = R / 10.0
            box_dimensions_nm = box_dimensions / 10

            bond_val = mapped_bond_fn(R_nm, box_dimensions_nm)
            angle_val = mapped_angle_fn(R_nm, box_dimensions_nm)
            return bond_val + angle_val

        return bond_angle_energy_fn

    # Setup the parameters

    if continue_opt:
        # Note: assumes that init_params should be a dictionary (e.g. not raveled as with conflict-free methods)
        init_params = jnp.load(last_dir / f"params_subset.npy", allow_pickle=True)
        init_params = init_params.item()
    else:
        init_bond_ks, init_bond_lengths, init_angle_ks, init_angle_theta0s = (
            utils.read_bond_angle_params(
                fpath="data/params/m3/martini_v3.0.0_ffbonded_v2_openbeta.itp"
            )
        )
        init_bond_ks = jnp.array(init_bond_ks)
        init_bond_lengths = jnp.array(init_bond_lengths)
        init_angle_ks = jnp.array(init_angle_ks)
        init_angle_theta0s = jnp.array(init_angle_theta0s)

        init_params = {
            "bonds": {
                "log_ks": jnp.log(init_bond_ks),
                "lengths": init_bond_lengths,
            },
            "angles": {"log_ks": jnp.log(init_angle_ks), "theta0s": init_angle_theta0s},
        }

    def get_full_params(base_params):
        params = base_params
        full_params = {
            "bonds": {
                "ks": jnp.exp(params["bonds"]["log_ks"]),
                "lengths": params["bonds"]["lengths"],
            },
            "angles": {
                "ks": jnp.exp(params["angles"]["log_ks"]),
                "theta0s": params["angles"]["theta0s"],
            },
        }
        return full_params

    def setup_membrane_sim(
        params,
        m_sim,
        m_sim_temp,
        seed,
        iter_dir,
        prev_basedir,
        is_tm_temp=False,
        pre_eq_biphasic=False,
    ):

        if not is_tm_temp:
            assert not pre_eq_biphasic

        subdir_name = all_subdir_names[m_sim]

        iter_msim_dir_base = iter_dir / subdir_name
        # iter_msim_dir_base.mkdir(parents=False, exist_ok=False)

        iter_msim_dir = iter_msim_dir_base / f"{m_sim_temp}K"
        iter_msim_dir.mkdir(parents=False, exist_ok=False)

        # FIXME: for now, initial positions are always the same
        if is_tm_temp:
            initial_position_path = tm_data_dir / subdir_name / "membrane.gro"
        elif prev_basedir is None:
            if is_tm_temp:
                initial_position_path = tm_data_dir / subdir_name / "membrane.gro"
            else:
                initial_position_path = (
                    data_dir
                    / "membrane_init"
                    / "m3"
                    / subdir_name
                    / f"{m_sim_temp}K"
                    / "membrane.gro"
                )
        else:
            if is_tm_temp:
                # Note: we don't use the output of the last frame from the last iteration, we use the once-equilibrated biphasic structure
                initial_position_path = (
                    prev_basedir
                    / subdir_name
                    / f"{m_sim_temp}K"
                    / "memb_eq_biphasic.gro"
                )
                assert initial_position_path.exists()

                if not pre_eq_biphasic:
                    # Copy over so it's accessible for the next iteration as well
                    shutil.copy(
                        initial_position_path, iter_msim_dir / "memb_eq_biphasic.gro"
                    )

                """
                # Copy over so it's accessible for the next iteration as well
                shutil.copy(initial_position_path, iter_msim_dir / "memb_eq_biphasic.gro")
                """
            else:
                # Note: whereas here, we use the last frame from the last iteration
                initial_position_path = (
                    prev_basedir / subdir_name / f"{m_sim_temp}K" / "memb.gro"
                )

        no_temp_subdirs = False
        if prev_basedir is None:
            if is_tm_temp:
                prev_basedir = tm_data_dir
                no_temp_subdirs = True
            else:
                prev_basedir = data_dir / "membrane_init" / "m3"

        # Copy in simulation files

        if no_temp_subdirs:
            shutil.copy(
                prev_basedir / subdir_name / "topol.top", iter_msim_dir / "topol.top"
            )
        else:
            shutil.copy(
                prev_basedir / subdir_name / f"{m_sim_temp}K" / "topol.top",
                iter_msim_dir / "topol.top",
            )
        shutil.copy(initial_position_path, iter_msim_dir / "membrane.gro")
        if no_temp_subdirs:
            prod_template_fpath = prev_basedir / subdir_name / "prod.mdp"
        else:
            prod_template_fpath = (
                prev_basedir / subdir_name / f"{m_sim_temp}K" / "prod.mdp"
            )
        utils.fill_template(
            prod_template_fpath,
            iter_msim_dir / "prod.mdp",
            tm_n_steps_per_sim if is_tm_temp else n_steps_per_sim,
            tm_n_eq_steps_per_sim if is_tm_temp else n_eq_steps_per_sim,
            tm_sample_every if is_tm_temp else sample_every,
            m_sim_temp,
        )
        if no_temp_subdirs:
            eq_template_fpath = prev_basedir / subdir_name / "eq.mdp"
        else:
            eq_template_fpath = prev_basedir / subdir_name / f"{m_sim_temp}K" / "eq.mdp"
        utils.fill_template(
            eq_template_fpath,
            iter_msim_dir / "eq.mdp",
            tm_n_steps_per_sim if is_tm_temp else n_steps_per_sim,
            tm_n_eq_steps_per_sim if is_tm_temp else n_eq_steps_per_sim,
            tm_sample_every if is_tm_temp else sample_every,
            m_sim_temp,
        )
        if pre_eq_biphasic:
            gel_temp = target_tms[m_sim] - 30.0
            liquid_temp = target_tms[m_sim] + 30.0
            assert is_tm_temp
            if no_temp_subdirs:
                eq_template_fpath = prev_basedir / subdir_name / "eq_biphasic.mdp"
            else:
                eq_template_fpath = (
                    prev_basedir / subdir_name / f"{m_sim_temp}K" / "eq_biphasic.mdp"
                )

            utils.fill_template(
                eq_template_fpath,
                iter_msim_dir / "eq_biphasic.mdp",
                None,
                tm_n_biphasic_eq_steps_per_sim,
                tm_sample_every,
                m_sim_temp,
                gel_temp=gel_temp,
                liquid_temp=liquid_temp,
            )

        if no_temp_subdirs:
            shutil.copy(
                prev_basedir / subdir_name / "index.ndx", iter_msim_dir / "index.ndx"
            )
        else:
            shutil.copy(
                prev_basedir / subdir_name / f"{m_sim_temp}K" / "index.ndx",
                iter_msim_dir / "index.ndx",
            )

        return iter_msim_dir, initial_position_path

    target_info_ref = ray.put(target_info)

    def process_membrane_sim(params, m_sim, m_sim_temp, iter_dir, local_target_info):

        m_sim_beta = 1 / get_kt(m_sim_temp)

        is_tm_temp = (m_sim in target_tm_sim_temps) and (
            m_sim_temp in target_tm_sim_temps[m_sim]
        )

        subdir_name = all_subdir_names[m_sim]
        iter_msim_dir = iter_dir / subdir_name / f"{m_sim_temp}K"
        assert iter_msim_dir.exists()

        analyze_dir = iter_msim_dir / "analysis"
        analyze_dir.mkdir(parents=False, exist_ok=False)

        time_path = iter_msim_dir / "time.txt"

        # Load the trajectory

        start = time.time()

        traj_path = iter_msim_dir / "memb.trr"
        if not traj_path.exists():
            traj_path = iter_msim_dir / "memb.xtc"
        assert traj_path.exists()

        u = mda.Universe(iter_msim_dir / "memb.tpr", traj_path)
        ref_states = []
        ref_dimensions = []
        for ts in u.trajectory:
            ref_states.append(deepcopy(ts.positions))
            ref_dimensions.append(deepcopy(ts.dimensions.tolist()[:3]))
        ref_states = jnp.array(ref_states)
        ref_dimensions = jnp.array(ref_dimensions)

        end = time.time()
        with open(time_path, "a+") as f:
            f.write(f"Loading trajectory: {end - start}\n")

        # Compute the energies

        start = time.time()

        terms = {
            "Coulomb-(SR)": "energies_gromacs_coulomb.xvg",
            "Potential": "energies_gromacs_potential.xvg",
            "Bond": "energies_gromacs_bond.xvg",
            "Angle": "energies_gromacs_angle.xvg",
            "LJ-(SR)": "energies_gromacs_lj.xvg",
        }

        for term in terms.keys():
            output = iter_msim_dir / terms[term]
            if term == "Coulomb-(SR)":
                term = "4"
            elif term == "LJ-(SR)":
                term = "3"
            cmd = f"{gromacs_path} energy -f {iter_msim_dir / 'memb.edr'} -s {iter_msim_dir / 'memb.tpr'} -o {output}"
            print(cmd)
            subprocess.run(cmd, shell=True, input=term, check=True, text=True)

            # read the energy file
            with open(output, "r") as f:
                if term == "4":
                    ref_coulomb = [
                        float(line.split()[1])
                        for line in f.readlines()
                        if not line.startswith(("#", "@"))
                    ]
                    ref_coulomb = jnp.array(ref_coulomb)
                elif term == "Potential":
                    ref_energies = [
                        float(line.split()[1])
                        for line in f.readlines()
                        if not line.startswith(("#", "@"))
                    ]
                    ref_energies = jnp.array(ref_energies)
                elif term == "Bond":
                    ref_bond_energies = [
                        float(line.split()[1])
                        for line in f.readlines()
                        if not line.startswith(("#", "@"))
                    ]
                    ref_bond_energies = jnp.array(ref_bond_energies)
                elif term == "Angle":
                    ref_angle_energies = [
                        float(line.split()[1])
                        for line in f.readlines()
                        if not line.startswith(("#", "@"))
                    ]
                    ref_angle_energies = jnp.array(ref_angle_energies)
                elif term == "3":
                    ref_lj_energies = [
                        float(line.split()[1])
                        for line in f.readlines()
                        if not line.startswith(("#", "@"))
                    ]
                    ref_lj_energies = jnp.array(ref_lj_energies)

        ref_bond_angle_energies = ref_bond_energies + ref_angle_energies
        ref_not_bond_angle_energies = ref_energies - ref_bond_angle_energies

        end = time.time()
        with open(time_path, "a+") as f:
            f.write(f"Computing energies: {end - start}\n")

        # Process the loaded energies and trajectory

        ## Note: for now, manually ignore initial state
        ref_dimensions = ref_dimensions[1:]
        ref_states = ref_states[1:]
        ref_energies = ref_energies[1:]
        ref_not_bond_angle_energies = ref_not_bond_angle_energies[1:]
        ref_lj_energies = ref_lj_energies[1:]

        ref_bond_angle_energies = ref_energies - ref_not_bond_angle_energies
        ref_states = utils.tree_stack(ref_states)  # Store trajectory as a pytree
        assert ref_states.shape[0] == n_ref_states

        # Compute observables from trajectory

        start = time.time()

        ## Compute thickness and area per lipid with lipyphillic

        assert u.trajectory.totaltime.is_integer()
        assert u.trajectory.dt.is_integer()

        start = time.time()

        all_ref_thicknesses, all_ref_area_per_lipid = structural.compute_obs_membrane(u)
        all_ref_thicknesses = all_ref_thicknesses[1:]
        all_ref_area_per_lipid = all_ref_area_per_lipid[1:]

        end = time.time()
        with open(time_path, "a+") as f:
            f.write(f"Computing Thickness and APL: {end - start}\n")

        ## Compute distributions
        ## Note: assumes unique mapping from bead pair to bond type
        if not is_tm_temp:

            start = time.time()

            bonds_dict_mapper = all_bonds_dict_mapper[m_sim]

            all_bond_dists = bonded_distributions.calculate_bond_statistics(
                u, bonds_dict_mapper, start=0, stop=None, step=1
            )
            all_bond_dists = jax.tree.map(jnp.array, all_bond_dists)
            all_bond_dists = jax.tree.map(lambda arr: arr[1:], all_bond_dists)

            angles_dict_mapper = all_angles_dict_mapper[m_sim]

            all_angle_rads = bonded_distributions.calculate_angle_statistics(
                u, angles_dict_mapper, start=0, stop=None, step=1
            )
            all_angle_rads = jax.tree.map(jnp.array, all_angle_rads)
            all_angle_rads = jax.tree.map(lambda arr: arr[1:], all_angle_rads)

            end = time.time()
            with open(time_path, "a+") as f:
                f.write(f"Computing distributions: {end - start}\n")
        else:
            all_bond_dists = None
            all_angle_rads = None

        # Compute energies (in JAX)
        start = time.time()

        bond_angle_energy_fn = get_bond_angle_energy_fn(
            m_sim,
            params["bonds"]["ks"],
            params["bonds"]["lengths"],
            params["angles"]["ks"],
            params["angles"]["theta0s"],
            is_tm_temp=is_tm_temp,
        )

        energy_scan_fn = lambda state, rs_idx: (
            None,
            bond_angle_energy_fn(ref_states[rs_idx], ref_dimensions[rs_idx]),
        )
        _, calc_bond_angle_energies = scan(
            energy_scan_fn, None, jnp.arange(n_ref_states)
        )

        calc_energies = calc_bond_angle_energies + ref_not_bond_angle_energies

        end = time.time()
        with open(time_path, "a+") as f:
            f.write(f"Computing energies (in JAX): {end - start}\n")

        # Check energy differences
        energy_diffs = onp.abs(calc_bond_angle_energies - ref_bond_angle_energies)

        with open(iter_msim_dir / "summary.txt", "a+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Min. energy diff: {onp.min(energy_diffs)}\n")
            f.write(f"Max. energy diff: {onp.max(energy_diffs)}\n")

        ## Plot the energies
        sns.distplot(calc_bond_angle_energies, label="Calculated", color="red")
        sns.distplot(ref_bond_angle_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_msim_dir / f"energies.png")
        plt.clf()

        ## Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(iter_msim_dir / f"energy_diffs.png")
        plt.clf()

        with open(iter_msim_dir / "summary.txt", "a+") as f:

            curr_mean_thickness = onp.mean(all_ref_thicknesses)
            thickness_type = "DHH (peak-peak)"
            f.write(f"Mean Thickness, {thickness_type} (A): {curr_mean_thickness}\n")

            curr_mean_area = onp.mean(all_ref_area_per_lipid)
            f.write(f"Mean Area (nm^2): {curr_mean_area}\n")

            boltzs = jnp.exp(-m_sim_beta * energy_diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom
            n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
            reweighted_area = jnp.dot(weights, all_ref_area_per_lipid)
            f.write(f"Reweighted Area (nm^2): {reweighted_area}\n")
            f.write(f"Neff: {n_eff}\n")

            if not is_tm_temp:
                f.write(f"\nBonds:\n")
                for lipid_type, lipid_bond_types in all_bond_dists.items():
                    f.write(f"- {lipid_type}:\n")
                    for l_bond_type, l_bond_dists_unflat in lipid_bond_types.items():
                        l_bond_dists = l_bond_dists_unflat.flatten()

                        curr_mean = onp.mean(l_bond_dists)
                        curr_var = onp.var(l_bond_dists)
                        f.write(
                            f"\t- l_bond_type: Mean={onp.round(curr_mean, 3)}, Var={onp.round(curr_var, 3)}\n"
                        )

                f.write(f"\nAngles:\n")
                for lipid_type, lipid_angle_types in all_angle_rads.items():
                    for l_angle_type, l_angles_unflat in lipid_angle_types.items():
                        l_angles = l_angles_unflat.flatten()

                        curr_mean = onp.mean(l_angles)
                        curr_var = onp.var(l_angles)
                        f.write(
                            f"\t- l_angle_type: Mean={onp.round(curr_mean, 3)}, Var={onp.round(curr_var, 3)}\n"
                        )

        end = time.time()
        with open(time_path, "a+") as f:
            f.write(f"Plotting: {end - start}\n")

        msim_info = {
            "ref_states": ref_states,
            "calc_energies": calc_energies,
            "ref_energies": ref_energies,
            "ref_not_bond_angle_energies": ref_not_bond_angle_energies,
            "all_ref_thicknesses": all_ref_thicknesses,
            "all_ref_area_per_lipid": all_ref_area_per_lipid,
            "all_bond_dists": all_bond_dists,
            "all_angle_rads": all_angle_rads,
            "ref_dimensions": ref_dimensions,
        }
        return msim_info

    if use_ray:

        @ray.remote(num_gpus=ray_num_gpus, num_cpus=ray_num_cpus)
        def run_processing_ray(
            params,
            m_sim,
            m_sim_temp,
            iter_dir,
        ):

            local_target_info = ray.get(target_info_ref)

            msim_info = process_membrane_sim(
                params, m_sim, m_sim_temp, iter_dir, local_target_info
            )

            gc.collect()

            return msim_info, m_sim, m_sim_temp

    def get_ref_states(params_subset, i, seed, prev_basedir):

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        params = get_full_params(params_subset)

        with open(iter_dir / "iter_params.txt", "a+") as f:
            f.write(f"{pprint.pformat(params)}")

        # shutil.copytree(prev_basedir / "lipid_defs", iter_dir / "lipid_defs")
        shutil.copytree(data_dir / "lipid_defs/m3", iter_dir / "lipid_defs")

        # Write parameters

        shutil.copytree(data_dir / "params/m3", iter_dir / "params")
        ffbonded_file = iter_dir / "params" / "martini_v3.0.0_ffbonded_v2_openbeta.itp"

        ## Update the ffbonded file
        utils.update_bond_angle_params(
            params["bonds"]["ks"],
            params["bonds"]["lengths"],
            params["angles"]["ks"],
            params["angles"]["theta0s"],
            ffbonded_file,
        )

        ## Update the LJ params, if necessary
        if "lj" in params.keys():
            ff_nonbonded_file = iter_dir / "params" / "martini_v3.0.0.itp"
            utils.update_lj_params(
                params["lj"]["epsilons"], params["lj"]["sigmas"], ff_nonbonded_file
            )

        # Setup simulations
        start_sim = time.time()

        all_iter_msim_dirs = list()
        all_initial_position_paths = list()
        all_pre_eq_biphasic = list()
        for m_sim in membrane_sims:

            subdir_name = all_subdir_names[m_sim]
            iter_msim_dir_base_all_temps = iter_dir / subdir_name
            iter_msim_dir_base_all_temps.mkdir(parents=False, exist_ok=False)

            for m_sim_temp in membrane_sim_temps[m_sim]:

                is_tm_temp = (m_sim in target_tm_sim_temps) and (
                    m_sim_temp in target_tm_sim_temps[m_sim]
                )
                pre_eq_biphasic = is_tm_temp
                all_pre_eq_biphasic.append(pre_eq_biphasic)

                iter_msim_dir, initial_position_path = setup_membrane_sim(
                    params,
                    m_sim,
                    m_sim_temp,
                    seed,
                    iter_dir,
                    prev_basedir,
                    is_tm_temp=is_tm_temp,
                    pre_eq_biphasic=pre_eq_biphasic,
                )
                all_iter_msim_dirs.append(iter_msim_dir)
                all_initial_position_paths.append(initial_position_path)

        # Run simulations

        ## 0. Run biphasic equilibration if necessary

        ## 0.1. Create necessary .tpr files
        procs = list()
        for iter_msim_dir, initial_position_path, pre_eq_biphasic in zip(
            all_iter_msim_dirs, all_initial_position_paths, all_pre_eq_biphasic
        ):
            if not pre_eq_biphasic:
                continue

            pre_eq_init_pos_path = initial_position_path

            cmd = f"{gromacs_path} grompp -f {iter_msim_dir / 'eq_biphasic.mdp'} -c {pre_eq_init_pos_path} -p {iter_msim_dir / 'topol.top'} -n {iter_msim_dir / 'index.ndx'} -o {iter_msim_dir / 'memb_eq_biphasic.tpr'} -maxwarn 1"
            cmd_list = shlex.split(cmd)
            procs.append(subprocess.Popen(cmd_list))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(
                    f"Binary creation for equilibration failed with error code: {rc}"
                )

        ## 0.2. run necessary equilibrations
        if not serial:
            procs = list()
        for iter_msim_dir, initial_position_path, pre_eq_biphasic in zip(
            all_iter_msim_dirs, all_initial_position_paths, all_pre_eq_biphasic
        ):
            if not pre_eq_biphasic:
                continue

            cmd = (
                f"{gromacs_path} mdrun -v -deffnm {iter_msim_dir / 'memb_eq_biphasic'}"
            )

            cmd += " -ntmpi 1"
            cmd_list = shlex.split(cmd)
            if serial:
                proc = subprocess.Popen(cmd_list)
                proc.wait()
                rc = proc.returncode
                if rc != 0:
                    raise RuntimeError(f"Equilibration failed with error code: {rc}")
            else:
                if use_ray and ray_num_gpus > 0:
                    assert ray_num_gpus == 1
                    cmd += " -gpu_id 0"

                if use_ray:
                    cmd_list = shlex.split(cmd)
                    task = run_subprocess_ray.remote(cmd_list)
                    procs.append(task)
                else:
                    procs.append(subprocess.Popen(cmd_list))

        if not serial:

            if use_ray:
                all_ret_info = ray.get(procs)

                all_rcs = [ret_info[0] for ret_info in all_ret_info]
                all_devices = [ret_info[3] for ret_info in all_ret_info]
                print(f"\nLOOK-HERE: {pprint.pformat(all_devices)}\n")
                for rc in all_rcs:
                    if rc != 0:
                        raise RuntimeError(
                            f"Equilibration failed with error code: {rc}"
                        )

                gc.collect()
            else:
                for p in procs:
                    p.wait()

                for p in procs:
                    rc = p.returncode
                    if rc != 0:
                        raise RuntimeError(
                            f"Equilibration failed with error code: {rc}"
                        )

        ## 0.3. copy in correct initial position paths
        new_initial_position_paths = list()
        for iter_msim_dir, initial_position_path, pre_eq_biphasic in zip(
            all_iter_msim_dirs, all_initial_position_paths, all_pre_eq_biphasic
        ):
            if not pre_eq_biphasic:
                new_initial_position_paths.append(initial_position_path)
            else:
                updated_path = iter_msim_dir / "memb_eq_biphasic.gro"
                assert updated_path.exists()
                new_initial_position_paths.append(updated_path)
        all_initial_position_paths = deepcopy(new_initial_position_paths)

        ## 1. Create a .tpr file for equilibration
        procs = list()
        for iter_msim_dir, initial_position_path in zip(
            all_iter_msim_dirs, all_initial_position_paths
        ):
            step1_init_pos_path = initial_position_path

            cmd = f"{gromacs_path} grompp -f {iter_msim_dir / 'eq.mdp'} -c {step1_init_pos_path} -p {iter_msim_dir / 'topol.top'} -n {iter_msim_dir / 'index.ndx'} -o {iter_msim_dir / 'memb_eq.tpr'} -maxwarn 1"

            cmd_list = shlex.split(cmd)
            procs.append(subprocess.Popen(cmd_list))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(
                    f"Binary creation for equilibration failed with error code: {rc}"
                )

        ## 2. Run the equilibration
        if not serial:
            procs = list()
        for iter_msim_dir, initial_position_path in zip(
            all_iter_msim_dirs, all_initial_position_paths
        ):
            cmd = f"{gromacs_path} mdrun -v -deffnm {iter_msim_dir / 'memb_eq'}"

            cmd += " -ntmpi 1"
            cmd_list = shlex.split(cmd)
            if serial:
                proc = subprocess.Popen(cmd_list)
                proc.wait()
                rc = proc.returncode
                if rc != 0:
                    raise RuntimeError(f"Equilibration failed with error code: {rc}")
            else:
                if use_ray and ray_num_gpus > 0:
                    assert ray_num_gpus == 1
                    cmd += " -gpu_id 0"

                if use_ray:
                    cmd_list = shlex.split(cmd)
                    task = run_subprocess_ray.remote(cmd_list)
                    procs.append(task)
                else:
                    procs.append(subprocess.Popen(cmd_list))

        if not serial:

            if use_ray:
                all_ret_info = ray.get(procs)
                all_rcs = [ret_info[0] for ret_info in all_ret_info]
                all_devices = [ret_info[3] for ret_info in all_ret_info]
                print(f"\nLOOK-HERE: {pprint.pformat(all_devices)}\n")
                for rc in all_rcs:
                    if rc != 0:
                        raise RuntimeError(
                            f"Equilibration failed with error code: {rc}"
                        )
            else:
                for p in procs:
                    p.wait()

                for p in procs:
                    rc = p.returncode
                    if rc != 0:
                        raise RuntimeError(
                            f"Equilibration failed with error code: {rc}"
                        )

        ## 3. Create a .tpr file for production
        procs = list()
        for iter_msim_dir, initial_position_path in zip(
            all_iter_msim_dirs, all_initial_position_paths
        ):
            cmd = f"{gromacs_path} grompp -f {iter_msim_dir / 'prod.mdp'} -c {iter_msim_dir / 'memb_eq.gro'} -p {iter_msim_dir / 'topol.top'} -n {iter_msim_dir / 'index.ndx'} -o {iter_msim_dir / 'memb.tpr'}"
            cmd_list = shlex.split(cmd)
            procs.append(subprocess.Popen(cmd_list))
        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(
                    f"Binary creation for production failed with error code: {rc}"
                )

        ## 4. Run the production
        if not serial:
            procs = list()
        for iter_msim_dir, initial_position_path in zip(
            all_iter_msim_dirs, all_initial_position_paths
        ):
            cmd = f"{gromacs_path} mdrun -v -deffnm {iter_msim_dir / 'memb'}"
            cmd += " -ntmpi 1"
            cmd_list = shlex.split(cmd)
            if serial:
                proc = subprocess.Popen(cmd_list)
                proc.wait()
                rc = proc.returncode
                if rc != 0:
                    raise RuntimeError(
                        f"Production simulation failed with error code: {rc}"
                    )
            else:
                if use_ray and ray_num_gpus > 0:
                    assert ray_num_gpus == 1
                    cmd += " -gpu_id 0"

                if use_ray:
                    cmd_list = shlex.split(cmd)
                    task = run_subprocess_ray.remote(cmd_list)
                    procs.append(task)
                else:
                    procs.append(subprocess.Popen(cmd_list))
        if not serial:
            if use_ray:
                all_ret_info = ray.get(procs)
                all_rcs = [ret_info[0] for ret_info in all_ret_info]
                all_devices = [ret_info[3] for ret_info in all_ret_info]
                print(f"\nLOOK-HERE: {pprint.pformat(all_devices)}\n")
                for rc in all_rcs:
                    if rc != 0:
                        raise RuntimeError(
                            f"Production simulation failed with error code: {rc}"
                        )
            else:
                for p in procs:
                    p.wait()

                for p in procs:
                    rc = p.returncode
                    if rc != 0:
                        raise RuntimeError(
                            f"Production simulation failed with error code: {rc}"
                        )

        end_sim = time.time()
        with open(iter_dir / "time.txt", "a+") as f:
            f.write(f"Simulation: {end_sim - start_sim}\n")

        ## Process all the runs
        start_analysis = time.time()

        if use_ray:
            procs = list()
            for m_sim in membrane_sims:
                for m_sim_temp in membrane_sim_temps[m_sim]:
                    task = run_processing_ray.remote(
                        params, m_sim, m_sim_temp, iter_dir
                    )
                    procs.append(task)

            all_ret_info = ray.get(procs)
            all_msim_info_list = [ret_info[0] for ret_info in all_ret_info]
            all_msim_names = [ret_info[1] for ret_info in all_ret_info]
            all_msim_temp_names = [ret_info[2] for ret_info in all_ret_info]

            all_msim_info = dict()
            for msim_info, m_sim, m_sim_temp in zip(
                all_msim_info_list, all_msim_names, all_msim_temp_names
            ):
                if m_sim in all_msim_info:
                    all_msim_info[m_sim][m_sim_temp] = msim_info
                else:
                    all_msim_info[m_sim] = {m_sim_temp: msim_info}
        else:

            all_msim_info = dict()
            for m_sim in membrane_sims:
                for m_sim_temp in membrane_sim_temps[m_sim]:
                    msim_info = process_membrane_sim(
                        params, m_sim, m_sim_temp, iter_dir, target_info
                    )
                    if m_sim in all_msim_info:
                        all_msim_info[m_sim][m_sim_temp] = msim_info
                    else:
                        all_msim_info[m_sim] = {m_sim_temp: msim_info}

        end_analysis = time.time()
        with open(iter_dir / "time.txt", "a+") as f:
            f.write(f"Analysis: {end_analysis - start_analysis}\n")

        return all_msim_info, iter_dir

    def loss_fn(params_subset, all_msim_info):
        params = get_full_params(params_subset)

        total_loss = 0.0
        n_effs = {m_sim: {} for m_sim in membrane_sims}
        all_losses = {m_sim: {} for m_sim in membrane_sims}

        all_tms = {}
        all_tm_sim_temps = {}
        all_tm_sim_apls = {}
        all_sigmoid_params = {}
        tm_expected_areas = {m_sim: {} for m_sim in target_tms.keys()}
        tm_loss = 0.0
        for m_sim, target_tm in target_tms.items():
            tm_sim_temps = target_tm_sim_temps[m_sim]

            tm_sim_temp_areas = list()

            for m_sim_temp in tm_sim_temps:

                m_sim_beta = 1 / get_kt(m_sim_temp)

                m_sim_losses = dict()

                bond_angle_energy_fn = get_bond_angle_energy_fn(
                    m_sim,
                    params["bonds"]["ks"],
                    params["bonds"]["lengths"],
                    params["angles"]["ks"],
                    params["angles"]["theta0s"],
                    is_tm_temp=True,
                )

                msim_info = all_msim_info[m_sim][m_sim_temp]
                ref_states = msim_info["ref_states"]
                ref_energies = msim_info["ref_energies"]

                ref_not_bond_angle_energies = msim_info["ref_not_bond_angle_energies"]
                all_ref_thicknesses = msim_info["all_ref_thicknesses"]
                all_ref_areas = msim_info["all_ref_area_per_lipid"]
                ref_dimensions = msim_info["ref_dimensions"]

                energy_scan_fn = lambda state, rs_idx: (
                    None,
                    bond_angle_energy_fn(ref_states[rs_idx], ref_dimensions[rs_idx]),
                )
                _, calc_bond_angle_energies = scan(
                    energy_scan_fn, None, jnp.arange(n_ref_states)
                )

                calc_energies = calc_bond_angle_energies + ref_not_bond_angle_energies

                diffs = calc_energies - ref_energies  # element-wise subtraction
                boltzs = jnp.exp(-m_sim_beta * diffs)
                denom = jnp.sum(boltzs)
                weights = boltzs / denom

                n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
                n_effs[m_sim][m_sim_temp] = n_eff

                # Area
                expected_area = jnp.dot(weights, all_ref_areas)
                tm_expected_areas[m_sim][m_sim_temp] = expected_area

                tm_sim_temp_areas.append(expected_area)

            # Compute the Tm by fitting a sigmoid
            tm_sim_temp_areas = jnp.array(tm_sim_temp_areas)

            sigmoid_params = tm.get_apl_params(
                tm_sim_temp_areas, jnp.array(tm_sim_temps)
            )
            all_sigmoid_params[m_sim] = sigmoid_params

            all_tm_sim_temps[m_sim] = tm_sim_temps
            all_tm_sim_apls[m_sim] = tm_sim_temp_areas

            m_sim_melt_temp = sigmoid_params[-1]
            all_tms[m_sim] = m_sim_melt_temp

            tm_pct_dev = percent_deviation(m_sim_melt_temp, target_tm)
            swarmcg_tm_loss = (
                swarmcg_w3 + jnp.max(jnp.array([0, tm_pct_dev - swarmcg_eps_tm]))
            ) ** 2 / n_tm_targets
            tm_loss += swarmcg_tm_loss

        expected_thicknesses = {m_sim: {} for m_sim in membrane_sims}
        expected_areas = {m_sim: {} for m_sim in membrane_sims}
        expected_bond_info = {m_sim: {} for m_sim in membrane_sims}
        expected_angle_info = {m_sim: {} for m_sim in membrane_sims}

        all_bond_type_nsims = {b_type: 0 for b_type in all_bond_types}
        all_bond_type_loss_sums = {b_type: 0.0 for b_type in all_bond_types}

        all_angle_type_nsims = {b_type: 0 for b_type in all_angle_types}
        all_angle_type_loss_sums = {a_type: 0.0 for a_type in all_angle_types}

        for m_sim in membrane_sims:

            for m_sim_temp in membrane_sim_temps[m_sim]:

                is_tm_temp = (m_sim in target_tm_sim_temps) and (
                    m_sim_temp in target_tm_sim_temps[m_sim]
                )
                if is_tm_temp:
                    continue

                m_sim_beta = 1 / get_kt(m_sim_temp)

                m_sim_losses = dict()

                bond_angle_energy_fn = get_bond_angle_energy_fn(
                    m_sim,
                    params["bonds"]["ks"],
                    params["bonds"]["lengths"],
                    params["angles"]["ks"],
                    params["angles"]["theta0s"],
                )

                msim_info = all_msim_info[m_sim][m_sim_temp]
                ref_states = msim_info["ref_states"]
                ref_energies = msim_info["ref_energies"]

                ref_not_bond_angle_energies = msim_info["ref_not_bond_angle_energies"]
                all_ref_thicknesses = msim_info["all_ref_thicknesses"]
                all_ref_areas = msim_info["all_ref_area_per_lipid"]
                ref_dimensions = msim_info["ref_dimensions"]

                # Option 1: Fast, requires lots of memory
                # calc_bond_angle_energies = vmap(bond_angle_energy_fn)(ref_states, ref_dimensions)

                # Option 2: slower, requires much less memory
                energy_scan_fn = lambda state, rs_idx: (
                    None,
                    bond_angle_energy_fn(ref_states[rs_idx], ref_dimensions[rs_idx]),
                )
                _, calc_bond_angle_energies = scan(
                    energy_scan_fn, None, jnp.arange(n_ref_states)
                )

                calc_energies = calc_bond_angle_energies + ref_not_bond_angle_energies

                diffs = calc_energies - ref_energies  # element-wise subtraction
                boltzs = jnp.exp(-m_sim_beta * diffs)
                denom = jnp.sum(boltzs)
                weights = boltzs / denom

                # Thickness

                expected_thickness = jnp.dot(weights, all_ref_thicknesses)
                expected_thicknesses[m_sim][m_sim_temp] = expected_thickness

                thickness_type = "DHH (peak-peak)"
                if (
                    "Thicknesses" in target_info[m_sim][m_sim_temp]
                    and thickness_type in target_info[m_sim][m_sim_temp]["Thicknesses"]
                ):
                    target_thickness = target_info[m_sim][m_sim_temp]["Thicknesses"][
                        thickness_type
                    ]
                    rmse_thickness = jnp.sqrt(
                        (expected_thickness - target_thickness) ** 2
                    )
                    total_loss += rmse_thickness

                    all_thickness_pct_devs = vmap(percent_deviation, (0, None))(
                        all_ref_thicknesses, target_thickness
                    )
                    avg_thickness_pct_dev = jnp.dot(weights, all_thickness_pct_devs)
                    swarmcg_thickness_loss = (
                        swarmcg_w1
                        + jnp.max(jnp.array([0, avg_thickness_pct_dev - swarmcg_eps]))
                    ) ** 2 / n_dhh_targets
                    m_sim_losses["thickness_loss"] = swarmcg_thickness_loss

                # Area
                expected_area = jnp.dot(weights, all_ref_areas)
                expected_areas[m_sim][m_sim_temp] = expected_area

                if "APL" in target_info[m_sim][m_sim_temp]:
                    target_area_per_lipid = target_info[m_sim][m_sim_temp]["APL"]
                    rmse_area = jnp.sqrt((expected_area - target_area_per_lipid) ** 2)

                    all_apl_pct_devs = vmap(percent_deviation, (0, None))(
                        all_ref_areas, target_area_per_lipid
                    )
                    avg_apl_pct_dev = jnp.dot(weights, all_apl_pct_devs)
                    # swarmcg_apl_loss = (swarmcg_w1 + jnp.max(jnp.array([0, avg_apl_pct_dev - swarmcg_eps])))**2 / n_targets
                    swarmcg_apl_loss = (
                        swarmcg_w1
                        + jnp.max(jnp.array([0, avg_apl_pct_dev - swarmcg_eps]))
                    ) ** 2 / n_apl_targets
                    m_sim_losses["apl_loss"] = swarmcg_apl_loss

                # Distributions

                ## Bonds
                all_bond_dists = msim_info["all_bond_dists"]
                msim_bond_info = dict()
                total_bond_distr_metric = 0.0
                n_bond_distr = 0
                for lipid_type, lipid_bond_types in all_bond_dists.items():

                    copy_number = all_m_sim_copy_number[m_sim][lipid_type]
                    copied_boltzs = jnp.repeat(boltzs, copy_number)
                    copied_denom = jnp.sum(copied_boltzs)
                    copied_weights = copied_boltzs / copied_denom

                    msim_bond_info[lipid_type] = dict()
                    for l_bond_type, l_bond_dists_unflat in lipid_bond_types.items():
                        l_bond_dists = l_bond_dists_unflat.flatten()

                        if (
                            "Bonds" in target_info[m_sim][m_sim_temp]
                            and lipid_type in target_info[m_sim][m_sim_temp]["Bonds"]
                            and l_bond_type
                            in target_info[m_sim][m_sim_temp]["Bonds"][lipid_type]
                        ):

                            target_distribution = target_info[m_sim][m_sim_temp][
                                "Bonds"
                            ][lipid_type][l_bond_type]["distribution"]
                            curr_distr_metric = wasserstein.wasserstein_1d(
                                u=l_bond_dists,
                                v=target_distribution,
                                u_weights=copied_weights,
                            )

                            msim_bond_info[lipid_type][l_bond_type] = {
                                "metric": curr_distr_metric
                            }

                            all_bond_type_nsims[l_bond_type] += 1
                            all_bond_type_loss_sums[l_bond_type] += (
                                swarmcg_w2 * curr_distr_metric
                            ) ** 2

                            total_bond_distr_metric += curr_distr_metric
                            n_bond_distr += 1

                            msim_bond_info[lipid_type][l_bond_type][
                                "distr_metric"
                            ] = curr_distr_metric

                if n_bond_distr > 0:
                    mean_bond_distr_metric = total_bond_distr_metric / n_bond_distr
                    total_loss += mean_bond_distr_metric

                    m_sim_losses["mean_bond_distr_metric"] = mean_bond_distr_metric

                expected_bond_info[m_sim][m_sim_temp] = msim_bond_info

                ## Angles

                all_angle_rads = msim_info["all_angle_rads"]
                msim_angle_info = dict()
                total_angle_distr_metric = 0.0
                n_angle_distr = 0
                for lipid_type, lipid_angle_types in all_angle_rads.items():

                    copy_number = all_m_sim_copy_number[m_sim][lipid_type]
                    copied_boltzs = jnp.repeat(boltzs, copy_number)
                    copied_denom = jnp.sum(copied_boltzs)
                    copied_weights = copied_boltzs / copied_denom

                    msim_angle_info[lipid_type] = dict()
                    for l_angle_type, l_angle_rads_unflat in lipid_angle_types.items():
                        l_angle_rads = l_angle_rads_unflat.flatten()

                        if (
                            "Angles" in target_info[m_sim][m_sim_temp]
                            and lipid_type in target_info[m_sim][m_sim_temp]["Angles"]
                            and l_angle_type
                            in target_info[m_sim][m_sim_temp]["Angles"][lipid_type]
                        ):

                            target_distribution = target_info[m_sim][m_sim_temp][
                                "Angles"
                            ][lipid_type][l_angle_type]["distribution"]
                            curr_distr_metric = wasserstein.wasserstein_1d(
                                u=onp.rad2deg(l_angle_rads),
                                v=onp.rad2deg(target_distribution),
                                u_weights=copied_weights,
                            )

                            all_angle_type_nsims[l_angle_type] += 1
                            all_angle_type_loss_sums[l_angle_type] += (
                                curr_distr_metric
                            ) ** 2

                            total_angle_distr_metric += curr_distr_metric
                            n_angle_distr += 1

                            msim_angle_info[lipid_type][l_angle_type] = {
                                "distr_metric": curr_distr_metric
                            }

                if n_angle_distr > 0:
                    mean_angle_distr_metric = total_angle_distr_metric / n_angle_distr
                    total_loss += mean_angle_distr_metric

                    m_sim_losses["mean_angle_distr_metric"] = mean_angle_distr_metric

                expected_angle_info[m_sim][m_sim_temp] = msim_angle_info

                n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
                n_effs[m_sim][m_sim_temp] = n_eff

                all_losses[m_sim][m_sim_temp] = m_sim_losses

        dhh_loss = None
        apl_loss = None
        otb_global = None

        # Top-down
        dhh_loss = 0.0
        apl_loss = 0.0
        for m_sim in membrane_sims:
            for m_sim_temp in membrane_sim_temps[m_sim]:
                is_tm_temp = (m_sim in target_tm_sim_temps) and (
                    m_sim_temp in target_tm_sim_temps[m_sim]
                )
                if is_tm_temp:
                    continue

                dhh_loss += all_losses[m_sim][m_sim_temp]["thickness_loss"]
                apl_loss += all_losses[m_sim][m_sim_temp]["apl_loss"]

        dhh_loss = jnp.sqrt(dhh_loss)
        apl_loss = jnp.sqrt(apl_loss)

        # Tm
        tm_loss = jnp.sqrt(tm_loss)

        bonds_sum = 0.0
        for bond_type in all_bond_types:
            bonds_sum += (
                jnp.sqrt(
                    all_bond_type_loss_sums[bond_type] / all_bond_type_nsims[bond_type]
                )
                ** 2
            )

        angles_sum = 0.0
        for angle_type in all_angle_types:
            angles_sum += (
                jnp.sqrt(
                    all_angle_type_loss_sums[angle_type]
                    / all_angle_type_nsims[angle_type]
                )
                ** 2
            )

        otb_global = jnp.sqrt((bonds_sum + angles_sum) / (n_bond_types + n_angle_types))
        otb_global = jnp.add(swarmcg_w0, otb_global)

        ret_loss = jnp.sqrt(
            (dhh_loss**2 + apl_loss**2 + otb_global**2 + tm_loss**2) / 4
        )
        # ret_loss = jnp.sqrt((dhh_loss**2 + apl_loss**2 + otb_global**2) / 3)
        total_loss = ret_loss

        return ret_loss, (
            n_effs,
            expected_thicknesses,
            expected_areas,
            expected_bond_info,
            expected_angle_info,
            all_losses,
            total_loss,
            ret_loss,
            dhh_loss,
            apl_loss,
            otb_global,
            tm_loss,
            all_tms,
            all_sigmoid_params,
            tm_expected_areas,
            all_tm_sim_temps,
            all_tm_sim_apls,
        )

    grad_fn = value_and_grad(loss_fn, has_aux=True)

    if continue_opt:
        with open(loss_path, "r") as f:
            last_iter_completed = len(f.readlines()) - 1
        iter_range = range(last_iter_completed + 1, last_iter_completed + 1 + n_iters)
        first_iter_i = last_iter_completed + 1

        # Delete any unfinished reference state collections
        for rt_dir in ref_traj_dir.glob("iter*"):
            rt_iter = int(rt_dir.stem.split("iter")[1])
            if rt_iter > last_iter_completed:
                print(f"\nDeleting {rt_dir}...\n")
                shutil.rmtree(rt_dir)

    else:
        iter_range = range(n_iters)
        first_iter_i = 0

    seed = 0
    if continue_opt:
        i = last_iter_completed + 1
    else:
        i = 0
    params = deepcopy(init_params)
    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr)
    elif optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=lr)
    elif optimizer_type == "lamb":
        optimizer = optax.lamb(learning_rate=lr)
    elif optimizer_type == "adagrad":
        optimizer = optax.adagrad(learning_rate=lr)
    elif optimizer_type == "lbfgs":
        optimizer = optax.lbfgs(learning_rate=lr)
    elif optimizer_type == "lars":
        optimizer = optax.lars(learning_rate=lr)
    elif optimizer_type == "fromage":
        optimizer = optax.fromage(learning_rate=lr)
    elif optimizer_type == "yogi":
        optimizer = optax.yogi(learning_rate=lr)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")
    if continue_opt:
        with open(last_dir / f"opt_state.pkl", "rb") as f:
            opt_state = pickle.load(f)
    else:
        opt_state = optimizer.init(params)

    prev_basedir = None
    start = time.time()
    all_msim_info, prev_basedir = get_ref_states(params, i, seed, prev_basedir)
    end = time.time()

    with open(resample_times_path, "a") as f:
        f.write(f"{end - start}\n")

    with open(global_time_path, "a") as f:
        f.write(f"Resample: {end - start}\n")

    # Optimize
    all_num_resample_iters = {
        m_sim: {m_sim_temp: 0 for m_sim_temp in membrane_sim_temps[m_sim]}
        for m_sim in membrane_sims
    }
    for i in tqdm(iter_range):

        jax.clear_caches()

        start = time.time()
        (loss, aux), grads = grad_fn(params, all_msim_info)
        end = time.time()

        jax.clear_caches()

        with open(global_time_path, "a") as f:
            f.write(f"Gradient: {end - start}\n")

        n_effs = aux[0]
        for m_sim in membrane_sims:
            for m_sim_temp in membrane_sim_temps[m_sim]:
                all_num_resample_iters[m_sim][m_sim_temp] += 1

        should_rerun_any = False
        for m_sim, m_sim_temps in membrane_sim_temps.items():
            for m_sim_temp in m_sim_temps:
                should_rerun_sim = n_effs[m_sim][m_sim_temp] < min_n_eff
                should_rerun_sim = should_rerun_sim and (i != first_iter_i)

                if should_rerun_sim:
                    should_rerun_any = True

        for m_sim, m_sim_temps_info in all_num_resample_iters.items():
            for m_sim_temp, num_resample_iters in m_sim_temps_info.items():
                if num_resample_iters >= max_approx_iters:
                    should_rerun_any = True

        if should_rerun_any:

            start = time.time()
            resampled_all_msim_info, prev_basedir = get_ref_states(
                params, i, random.randrange(100), prev_basedir
            )
            end = time.time()
            with open(resample_times_path, "a") as f:
                f.write(f"{end - start}\n")

                with open(global_time_path, "a") as f:
                    f.write(f"Resample: {end - start}\n")

            for m_sim, m_sim_temps in membrane_sim_temps.items():
                for m_sim_temp in m_sim_temps:
                    all_num_resample_iters[m_sim][m_sim_temp] = 0
                    all_msim_info[m_sim][m_sim_temp] = deepcopy(
                        resampled_all_msim_info[m_sim][m_sim_temp]
                    )

            start = time.time()
            (loss, aux), grads = grad_fn(params, all_msim_info)
            end = time.time()

            with open(global_time_path, "a") as f:
                f.write(f"Gradient: {end - start}\n")

        (
            n_effs,
            curr_thicknesses,
            curr_areas,
            curr_bond_info,
            curr_angle_info,
            curr_losses,
            total_loss,
            _,
            curr_dhh_loss,
            curr_apl_loss,
            curr_otb_global,
            curr_tm_loss,
            curr_tms,
            curr_sigmoid_params,
            tm_curr_areas,
            tm_curr_sim_temps,
            tm_curr_sim_apls,
        ) = aux

        # Plot sigmoids
        for m_sim, target_tm in target_tms.items():
            sigmoid_params = curr_sigmoid_params[m_sim]
            curr_tm = curr_tms[m_sim]

            tm_sim_temps = target_tm_sim_temps[m_sim]

            apl0_jax, c_p_g_jax, dAPL_jax, k_jax, Tm_jax = sigmoid_params

            areas_to_plot = [
                tm_curr_areas[m_sim][temp] for temp in tm_sim_temps
            ]  # convert dictionary to list
            plt.scatter(tm_sim_temps, areas_to_plot, color="red")

            xs_plot = onp.linspace(onp.min(tm_sim_temps), onp.max(tm_sim_temps), 100)
            ys_plot = tm.calculate_apl(
                xs_plot, apl0_jax, c_p_g_jax, dAPL_jax, k, Tm_jax
            )
            plt.plot(xs_plot, ys_plot, color="black", label="Fit sigmoid")

            plt.axvline(
                x=target_tm,
                linestyle="--",
                color="red",
                label=f"Target = {onp.round(target_tm, 3)}",
            )
            plt.axvline(
                x=Tm_jax,
                linestyle="--",
                color="blue",
                label=f"Current = {onp.round(Tm_jax, 3)}",
            )
            plt.legend()
            plt.savefig(sigmoid_dir / f"iter{i}_{m_sim}.png")
            plt.clf()

        with open(n_effs_path, "a") as f:
            f.write(f"{pprint.pformat(n_effs)}\n")

        with open(dhh_loss_path, "a") as f:
            f.write(f"{curr_dhh_loss}\n")
        with open(apl_loss_path, "a") as f:
            f.write(f"{curr_apl_loss}\n")
        with open(otb_loss_path, "a") as f:
            f.write(f"{curr_otb_global}\n")
        with open(tm_loss_path, "a") as f:
            f.write(f"{curr_tm_loss}\n")

        with open(tms_path, "a") as f:
            f.write(f"{pprint.pformat(curr_tms)}\n")

        with open(tm_sim_temps_path, "a") as f:
            f.write(f"{pprint.pformat(tm_curr_sim_temps)}\n")

        with open(tm_sim_apls_path, "a") as f:
            f.write(f"{pprint.pformat(tm_curr_sim_apls)}\n")

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(all_losses_path, "a") as f:
            f.write(f"{pprint.pformat(curr_losses)}\n")
        with open(iter_params_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")
        with open(thicknesses_path, "a") as f:
            f.write(f"\nIteration {i}:\n")
            for m_sim, m_sim_temps in curr_thicknesses.items():
                for m_sim_temp in m_sim_temps:
                    f.write(
                        f"- {m_sim}, {m_sim_temp}K: {pprint.pformat(curr_thicknesses[m_sim][m_sim_temp])}\n"
                    )
        with open(areas_path, "a") as f:
            f.write(f"\nIteration {i}:\n")
            for m_sim, m_sim_temps in curr_areas.items():
                for m_sim_temp in m_sim_temps:
                    f.write(
                        f"- {m_sim}, {m_sim_temp}K: {pprint.pformat(curr_areas[m_sim][m_sim_temp])}\n"
                    )
        with open(bonds_path, "a") as f:
            f.write(f"\nIteration {i}:\n")
            for m_sim, m_sim_bond_infos in curr_bond_info.items():
                for m_sim_temp, m_sim_bond_info in m_sim_bond_infos.items():
                    f.write(
                        f"- {m_sim}, {m_sim_temp}K: {pprint.pformat(m_sim_bond_info)}\n"
                    )
        with open(angles_path, "a") as f:
            f.write(f"\nIteration {i}:\n")
            for m_sim, m_sim_angle_infos in curr_angle_info.items():
                for m_sim_temp, m_sim_angle_info in m_sim_angle_infos.items():
                    f.write(
                        f"- {m_sim}, {m_sim_temp}K: {pprint.pformat(m_sim_angle_info)}\n"
                    )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        jnp.save(last_dir / f"params_subset.npy", params)

        with open(last_dir / f"opt_state.pkl", "wb") as f:
            pickle.dump(opt_state, f)

        if i and i % save_every == 0:
            jnp.save(obj_dir / f"params_subset_iter{i}.npy", params)

            with open(obj_dir / f"opt_state_iter{i}.pkl", "wb") as f:
                pickle.dump(opt_state, f)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Optimize MARTINI-3 bonded parameters."
    )

    parser.add_argument("--run-name", type=str, help="Run name", required=True)

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--checkpoint-every", type=int, default=25, help="Checkpoint frequency"
    )

    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--use-ray", action="store_true")

    parser.add_argument(
        "--gromacs-path", type=str, required=True, help="GROMACS base directory"
    )

    parser.add_argument(
        "--n-eq-steps-per-sim",
        type=int,
        default=1000,
        help="# of equilibration steps per production simulation",
    )
    parser.add_argument(
        "--n-steps-per-sim",
        type=int,
        default=1000,
        help="# of steps per production simulation",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=100,
        help="Frequency of sampling conformations",
    )

    parser.add_argument(
        "--tm-n-eq-steps-per-sim",
        type=int,
        default=1000,
        help="# of equilibration steps per Tm production simulation",
    )

    parser.add_argument(
        "--tm-n-biphasic-eq-steps-per-sim",
        type=int,
        default=1000,
        help="# of equilibration steps per Tm production simulation for biphasic equilibration, production should be 500000 (10 ns)",
    )

    parser.add_argument(
        "--tm-n-steps-per-sim",
        type=int,
        default=1000,
        help="# of steps per Tm production simulation, production should be 3750000 (75 ns)",
    )
    parser.add_argument(
        "--tm-sample-every",
        type=int,
        default=100,
        help="Frequency of sampling conformations, production should be 2500",
    )

    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="adagrad",
        choices=[
            "adam",
            "rmsprop",
            "lamb",
            "adagrad",
            "lbfgs",
            "lars",
            "fromage",
            "yogi",
        ],
        help="Type of optimizer",
    )

    parser.add_argument(
        "--min-neff-factor",
        type=float,
        default=0.95,
        help="Factor for determining min Neff",
    )
    parser.add_argument(
        "--max-approx-iters",
        type=int,
        default=5,
        help="Maximum number of iterations before resampling",
    )

    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Number of iterations of gradient descent",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    parser.add_argument(
        "--targets-dir",
        type=str,
        default="experiments/targets/swarmcg-pc",
        help="Directory specifying YAML files for optimization targets",
    )

    parser.add_argument(
        "--save-every", type=int, default=5, help="Frequency of saving parameters"
    )

    parser.add_argument("--continue-opt", action="store_true")

    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=1,
        help="Optional number of CPUS per ray worker",
    )
    parser.add_argument(
        "--ray-num-gpus",
        type=int,
        default=0,
        help="Optional number of CPUS per ray worker",
    )

    parser.add_argument(
        "--output-dir", type=str, default="output/", help="Directory for output"
    )

    parser.add_argument(
        "--swarmcg-w0",
        type=float,
        default=0.0,
        help="W0 parameter for scaling bottom up loss",
    )
    parser.add_argument(
        "--swarmcg-w1",
        type=float,
        default=10.0,
        help="W1 parameter for scaling top down loss",
    )
    parser.add_argument(
        "--swarmcg-w2",
        type=float,
        default=50.0,
        help="W2 parameter for scaling bottom up loss",
    )
    parser.add_argument(
        "--swarmcg-w3",
        type=float,
        default=10.0,
        help="W3 parameter for scaling Tm loss",
    )
    parser.add_argument(
        "--swarmcg-eps",
        type=float,
        default=1.5,
        help="Experimental tolerance for swarmcg loss",
    )
    parser.add_argument(
        "--swarmcg-eps-tm",
        type=float,
        default=0.7,
        help="Experimental tolerance for Tm swarmcg loss",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
