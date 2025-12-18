# jax-martini

This repository contains the codebase for running gradient-based optimizations of Martini lipid force field parameters, using a **JAX** implementation of the Martini energy function and **GROMACS** for sampling. The main entry points are the experiment scripts in `experiments/`, which run optimization workflows for **wet Martini 3** and **Dry Martini** setups.

## Repository layout

```text
.
├── data/
│   ├── lipid_defs/      # Lipid topologies
│   ├── membrane_init/   # Starting pre-equilibrated membrane configurations
│   ├── params/          # Initial FF parameter files
│   └── aa_targets/      # Atomistic-derived targets (bond/angle distributions)
│
├── experiments/
│   ├── opt_wet_m3.py    # Optimization script for wet Martini 3
│   ├── opt_dry_m2.py    # Optimization script for Dry Martini
│   └── targets/         # Experiment-specific targets (top-down + bottom-up)
│
├── jax_martini/
│   ├── __init__.py
│   ├── checkpoint.py    # Checkpointing utilities for long runs
│   ├── energy.py        # JAX implementation of Martini energy terms
│   ├── observables/     # Observable calculations (APL, DHH, distributions, etc.)
│   ├── tests/           # Unit/regression tests for core components
│   └── utils.py         # Shared utilities (I/O, logging, helpers)
│
├── output/              # Folder for generated outputs
└── README.md            # This file.
```

## Requirements

### Software
- **Python 3.11** recommended
- **GROMACS** available on your system and callable via the path you pass to `--gromacs-path`.

### Python dependencies
All Python dependencies are listed in `requirements.txt`.
> **Note on JAX installation**: depending on your platform (CPU vs CUDA), you may prefer to install JAX/JAXLIB following the official JAX instructions, and then install the remaining dependencies from `requirements.txt`.

### Optional (recommended)
- **Ray** for multi-CPU job parallelism (`--use-ray`).
- HPC environment (Slurm, etc.) for large experiments.

## Installation

### 1) Clone the repository
``` bash
git clone https://github.com/danielpastor97/jax-martini
cd jax-martini
```

### 2) Install Python requirements
``` bash
pip install -r requirements.txt
```

### 3) Verify GROMACS is available
``` bash
/path/to/bin/gmx --version
```

## Running optimizations
All experiments are launched from the `jax-martini` directory. Use `--help` to see all options:
``` bash
python -m experiments.opt_wet_m3 --help
python -m experiments.opt_dry_m2 --help
```

### Examples (as used in the manuscript)
#### 1) Wet Martini 3 — DPPC (bonded parameters)
``` bash
python -m experiments.opt_wet_m3 \
  --run-name opt-wet-m3-only-dppc \
  --gromacs-path path/to/bin/gmx \
  --targets-dir experiments/targets/m3_pc_no_ion_only_dppc/ \
  --use-ray --ray-num-cpus 5 \
  --n-iters 100 \
  --n-steps-per-sim 10000000 \
  --sample-every 10000 \
  --n-eq-steps-per-sim 1000000 \
  --tm-n-eq-steps-per-sim 1500000 \
  --tm-n-biphasic-eq-steps-per-sim 500000 \
  --tm-n-steps-per-sim 2500000 \
  --tm-sample-every 2500 \
  --swarmcg-w0 11.06 --swarmcg-w1 0.001 --swarmcg-w3 8.29
```

#### 2) Wet Martini 3 — All PC lipids (bonded parameters)
``` bash
python -m experiments.opt_wet_m3 \
  --run-name opt-wet-m3-all-pc \
  --gromacs-path path/to/bin/gmx \
  --targets-dir experiments/targets/m3_pc_no_ion/ \
  --use-ray --ray-num-cpus 5 \
  --n-iters 100 \
  --n-steps-per-sim 10000000 \
  --sample-every 10000 \
  --n-eq-steps-per-sim 1000000 \
  --tm-n-eq-steps-per-sim 1500000 \
  --tm-n-biphasic-eq-steps-per-sim 500000 \
  --tm-n-steps-per-sim 2500000 \
  --tm-sample-every 2500 \
  --swarmcg-w0 5.46 --swarmcg-w1 0.001 --swarmcg-w3 7.48
```

#### 3) Dry Martini 2 — All PC lipids (bonded + nonbonded parameters)
``` bash
python -m experiments.opt_dry_m2 \
  --run-name opt-dry-m2-all-pc \
  --gromacs-path path/to/bin/gmx \
  --targets-dir ./experiments/targets/m2_pc_no_ion \
  --use-ray --ray-num-cpus 5 \
  --n-iters 100 \
  --n-eq-steps-per-sim 1000000 \
  --n-steps-per-sim 10000000 \
  --sample-every 1000 \
  --optimizer-type adagrad \
  --swarmcg-w1 35.8
```

### Inputs and targets
Optimizations combine bottom-up targets (e.g., bond/angle distributions) a nd top-down targets (APL, DHH, transition temperature scan settings) defined per experiment in experiments/targets/. Each `--targets-dir` points to a self-contained folder describing the training systems and their observables in hte form of yaml files.

### Outputs and checkpoints
Runs write to `output/` under a run-specific folder (based on `--run-name`). If a run is interrupted, you can usually resume from the latest checkpoint using `--continue-opt`.

## Citation
If you use this repository in your work work, please cite:
- **TBA**.

## Contact / Issues
Please use the repository issue tracker for:
- bug reports
- questions about reproducing experiments
- suggestions for new targets, lipids, or observables

