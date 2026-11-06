![Error Distribution Smoothing — representative sample selection](assets/hero.svg)

# Error Distribution Smoothing

[Project homepage](./) · [Quick start](#quick-start) · [Repository](https://github.com/PaoShou-China/Error-Distribution-Smoothing)

This repository contains the Error Distribution Smoothing (EDS) data-management implementation and experiments for imbalanced regression. The examples cover synthetic regression, Lorenz-system identification, polar moment of inertia, Cartpole, and Quadcopter tasks.

## Core components

- Shared EDS/SMOGN implementations and data-management utilities.
- Experiment-specific processing scripts and datasets.
- Regression, system-identification, and plotting examples.

## Installation

Use a virtual environment and install the shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Quick start

Start from a specific experiment directory and follow its local scripts and README/instructions, where present. For example:

```bash
cd 1_Motivation_Example
python process.py
```

The repository contains experiment-specific variants of `manager.py` and `process.py`; inspect each script's input/output paths before running. Some experiments expect data files already included in their `dataset/` folders.

## Repository map

- `src/`: shared EDS and SMOGN implementations.
- `manager.py`, `process.py`: root-level data-management example.
- `1_Motivation_Example/`: synthetic motivation example and plots.
- `2_Dynamics_System_Identification/`: Lorenz-system identification.
- `3_Polar_Moment_of_Inertia/`: synthetic image-feature regression.
- `4_Real_World_Experiment/4_1_Cartpole/`: Cartpole example.
- `4_Real_World_Experiment/4_2_Quadcopter/`: Quadcopter example.
- `experiments/`: additional experiment and plotting scripts.

## Reproducibility notes

Run scripts from their documented working directory because some use relative paths. Dataset provenance and feature/label schemas are not consistently recorded in the repository; consult the experiment files and document those details before reusing datasets or comparing results. Seeds and model parameters may be experiment-specific.

## Citation

See the repository's paper/reference information where provided. Please use the final published bibliographic record when citing the method.
