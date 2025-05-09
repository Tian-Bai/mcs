# Multivariate Conformal Selection

This repository contains the reproduction codes and related materials presented in the paper [Multivariate Conformal Selection](https://arxiv.org/abs/2505.00917).

## Folders

- `simulation/`: Simulation Studies (Section 5)
    - `simulation/dist_score.py`: implementation of `mCS-dist`.
    - `simulation/DL_single_score.py`: implementation of `mCS-learn` with p-value loss (eq. 16).
    - `simulation/DL_mult_score.py`: implementation of `mCS-learn` with selection size loss (eq. 15).
    - `simulation/plot_and_table.ipynb`: reproducing the plots and tables in Section 5 and Appendix C.

- `drug/`: Real Data Application (Section 6)
    - `drug/dist_score.py` and `drug/DL_single_score.py`: similar to the corresponding files in the `simulation` folder.
    - `drug/training_mu.ipynb`: demonstration of QSAR model training and prediction.
    - `drug/plot_and_table.ipynb`: reproducing the plots and tables in Section 6 and Appendix D.