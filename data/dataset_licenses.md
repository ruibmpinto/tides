# Dataset and code-asset licenses

This document records the provenance and license of every external asset used in the paper, in the format requested by the NeurIPS reproducibility checklist (item 12).

## Datasets

### UEA Time Series Classification Archive

* Citation: Bagnall, A., Dau, H. A., Lines, J., Flynn, M., Large, J., Bostrom, A., Southam, P., & Keogh, E. (2018). The UEA Multivariate Time Series Classification Archive, 2018. arXiv:1811.00075.
* Source: http://www.timeseriesclassification.com
* Loader used: `aeon.datasets.load_classification`
* Distribution: research-use terms from the archive maintainers; redistributed unmodified.
* Datasets used: `EigenWorms`, `EthanolConcentration`, `Heartbeat`, `MotorImagery`, `SelfRegulationSCP1`, `SelfRegulationSCP2`.

### Physiome-ODE

* Citation: Klötergens, C., Yalavarthi, V. K., Stubbemann, M., & Schmidt-Thieme, L. (2025). Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting based on ODEs. ICLR 2025.
* Source: https://zenodo.org/records/11492058
* License: Apache-2.0
* All 50 datasets are used as published; no modifications.

## External code

### Backbone and SSM utilities (S5)

* Source: https://github.com/lindermanlab/S5
* License: Apache-2.0
* Use: associative scan, HiPPO/DPLR initialization, ZOH/bilinear discretization. Adapted into PyTorch and used inside `tides/tides.py`.

### Baseline implementations (Physiome-ODE table)

These are referenced by URL only and are NOT redistributed in this repository. Reported numbers are copied from the public Physiome-ODE leaderboard.

| Baseline | Repository | License |
|----------|------------|---------|
| GRU-ODE-Bayes | https://github.com/edebrouwer/gru_ode_bayes | MIT |
| Neural Flows | https://github.com/mbilos/neural-flows-experiments | MIT |
| CRU | https://github.com/boschresearch/Continuous-Recurrent-Units | AGPL-3.0 |
| LinODENet | https://github.com/randolf-scholz/linodenet | MIT |
| GraFITi | https://github.com/yalavarthivk/GraFITi | MIT |

### Baseline implementations (UEA table)

| Baseline | Source paper | License |
|----------|--------------|---------|
| Rough Transformer (RFormer) | Moreno-Pino et al., NeurIPS 2024 | upstream repository released without an explicit LICENSE file; only architectural ideas reused, no source files redistributed |
| Log-NCDE | Walker et al., 2024 | MIT |

## Code license for this repository

All source code under this repository is released under the MIT License (see `LICENSE`), with the exception of files whose headers explicitly attribute external authorship.