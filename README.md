# Error Distribution Smoothing

This repository contains the Error Distribution Smoothing (EDS) data-management implementation and four groups of experiments for imbalanced regression.

## Repository map

- `src/`: shared EDS and SMOGN implementations.
- `manager.py`, `process.py`: root-level data-management example.
- `1_Motivation_Example/`: synthetic motivation example and plots.
- `2_Dynamics_System_Identification/`: Lorenz-system identification examples.
- `3_Polar_Moment_of_Inertia/`: synthetic image-feature regression example.
- `4_Real_World_Experiment/4_1_Cartpole/`: Cartpole experiment.
- `4_Real_World_Experiment/4_2_Quadcopter/`: Quadcopter experiment.
- `experiments/`: additional experiment and plotting scripts.

Many experiment directories contain their own `manager.py` and `process.py`; run those scripts from their respective directories unless the script explicitly accepts a path.

## Requirements

The shared processing examples use NumPy and SciPy. Some experiment scripts also require PyTorch, Matplotlib, scikit-learn, PySINDy, or other packages. Install the packages required by the experiment you intend to run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Reproduction workflow

1. Select an experiment directory.
2. Inspect its script's main block for the configured data paths and parameters.
3. Run its `process.py` from that directory to generate the managed training set.
4. Run the experiment's regression/evaluation script, then its plotting script if applicable.

For example:

```bash
cd 1_Motivation_Example
python process.py
```

The experiment scripts currently define many parameters and paths in source; check those before launching. Outputs are generally written relative to the experiment directory.

## Data notes

Datasets are stored as text files within the experiment directories. The repository README and scripts do not consistently identify external dataset sources, licenses, units, or all feature/label schemas. Treat these as undocumented until the corresponding experiment's generation/loading code or accompanying publication establishes them; do not assume a common schema across experiments. Before reusing a dataset, record its source, license, feature/label meaning, units, and train/test split.

## Core API

The `Manager` class manages training points using Delaunay triangulation and cKDTree-based operations. The per-experiment `process.py` files show how initial points and subsequent batches are passed into the manager. Refer to those scripts for experiment-specific `delta`, batch size, and initial batch size.
