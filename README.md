# PERC: Physics Enforced Reservoir Computing
Apply hard physical constraints directly to Reservoir Computers! This repository contains the supporting code for the L4DC 2025 paper Physics-Enforced Reservoir Computing for Forecasting Spatiotemporal Systems" by Tretiak, et al.

PERC enforces hard constraints by solving a constrained optimization problem during the Ridge Regression step of RC training. For a given time series $u(t)$, PERC guarantees adhereance to linear invariants (any linear conservation laws) of the form $Cu(t) = d$. The examples are split into categories where $d=0$ (linear homogenous constraints) and $d \neq 0$ (linear inhomogenous constraints). Furthermore, nonlinear constraints of the form $p(u(t)) = d$ can be also promoted with PERC.

## Installation 
To install `perc` please clone the repository then install via `pip`
```bash
git clone https://github.com/dtretiak/PhysicsEnforcedReservoirComputing.git
cd PhysicsEnforcedReservoirComputing
pip install .
```

## Examples
All examples from the paper are included as jupyter notebooks `./notebooks/`. 
- For a basic unconstrained RC example please see `basic_RC_example.ipynb`
- For linear homogeneous constraints of the form $Cu = 0$, please see `kol_flow.ipynb` and `ks.ipynb`
- For linear inhomogenous constraints of the form $Cu = d$, please see `lv.ipynb` and `heat_eq.ipynb`
- For nonlinear constraints, please see `hamiltonian.ipynb`

## Data Availability 
Most of the examples shown in the paper contain simple enough datasets that can be simulated directly prior to training. All of the integrators can be found in `perc.integrators`. However, the Kolmogorov Flow simulation is performed separately via the integrator found [here](https://github.com/smokbel/Controlling-Kolmogorov-Flow). Some data is provided below in order to demonstrate the PERC algorithm, but please note the usage guidelines for the simulator and contact the owner of that repository prior to using the data for other purposes. 

In order to download the fluids data, first install the hugging face CLI
```bash
pip install -U "huggingface_hub[cli]"
```

Then download the data set into `./data/`
```bash
huggingface-cli download dtretiak/PERC_data --include "kol_flow/*" --repo-type dataset --local-dir ./data
```

## Citation 
If you find this work useful in your endeavors, please consider citing it: 

```bibtex
@InProceedings{tretiak2025perc,
    title={Physics-Enforced Reservoir Computing for Forecasting Spatiotemporal Systems},
    author={Tretiak, Dima and Anastasia, Bizyaeva and Kutz, J. Nathan and Brunton, Steven L.},
    booktitle={Learning For Dynamics and Controls},
    series={Proceedings of Machine Learning Research},
    publisher={{PMLR}},
    year={2025},
    volume={283},
}
```