# Notebooks

This directory contains the Jupyter notebooks used to evaluate the uncertainty quantification (UQ) methods presented in:

> **Multiobjective Evolutionary Calibration of Uncertainty Quantification for Large Language Models**
> GECCO Companion 2026

## Overview

The notebooks implement the evaluation pipeline for the optimized embedding-based uncertainty estimators. They compute uncertainty scores from the optimized Pareto-front solutions and evaluate their ability to distinguish correct from incorrect answers using multiple correctness definitions.

The evaluation includes:

- AUROC (Area Under the ROC Curve)
- AURC (Area Under the Risk–Coverage Curve)
- Rejection accuracy at multiple coverage levels
- Evaluation on both in-distribution and out-of-distribution datasets

---

## Contents

### `UQ_eval_V5(AUROC-AURC)(Llama).ipynb`

Evaluation notebook for **Llama 3.1 8B**.

This notebook:

- Loads the optimized Pareto-front solutions.
- Computes uncertainty scores using the embedding-space estimator.
- Evaluates AUROC and AURC.
- Computes rejection accuracy under different coverage levels.
- Reports the experimental results presented in the paper.

---

### `UQ_eval_V5(AUROC-AURC)(DeepSeek).ipynb`

Evaluation notebook for **DeepSeek 7B**.

The evaluation protocol is identical to the Llama notebook, allowing direct comparison between the two models.

---

## Evaluation Protocol

Each notebook performs repeated evaluation using:

- Multiple calibration/evaluation splits
- Percentile-based correctness thresholds
- In-distribution (MedQuAD) evaluation
- Out-of-distribution evaluation
- Aggregation over Pareto-front solutions

The reported statistics correspond to the averages across repeated runs described in the paper.

---


## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mora2026gecco,
  author    = {Maria Mora-Cross and Sa{\'u}l Calder{\'o}n-Ram{\'\i}rez and Sebasti{\'a}n Rojas Gonz{\'a}lez},
  title     = {Multiobjective Evolutionary Calibration of Uncertainty Quantification for Large Language Models},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO Companion '26)},
  year      = {2026},
  address   = {San Jose, Costa Rica},
  publisher = {ACM},
  doi        = {10.1145/3795101.3805422}
}
```
