# Multiobjective Evolutionary Calibration of Uncertainty Quantification for Large Language Models

Large language models (LLMs) are increasingly used in high impact applications such as clinical decision support, where fluent outputs can hide critical errors. Uncertainty quantification (UQ) should discriminate correct from incorrect answers and remain informative under dataset shift. We formulate UQ design as a multiobjective evolutionary optimization problem over unsupervised estimators derived from hidden state embeddings. Candidate estimators rely on clustering algorithms followed by a kNN based uncertainty signal. We evaluate Llama 3.1 8B, and DeepSeek 7B on MedQuAD, with a suite of out of distribution datasets (OOD). Objective functions minimize Pearson correlation between UQ and text quality metrics (BERTScore F1 and ROUGE L). Using RVEA, we analyze the search dynamics and the resulting Pareto front of the trade off between semantic and lexical quality. Preliminary results show competitive AUROC and AURC, and improved AUROC relative to a Monte Carlo Dropout (MCD) baseline, highlighting the potential of evolutionary multiobjective optimization as a robust and scalable strategy for building reliable, risk aware generative AI systems.

Official implementation of the methods presented in the GECCO 2026 Companion paper.

## Overview

This repository contains the implementation of evolutionary optimization methods for uncertainty quantification in Large Language Models using latent-space representations.

## Repository Structure

```
src/
data/
notebooks/
``

## Citation
@inproceedings{mora2026gecco,
  author    = {Maria Mora-Cross and Sa{\'u}l Calder{\'o}n-Ram{\'\i}rez and Sebasti{\'a}n Rojas Gonz{\'a}lez},
  title     = {Multiobjective Evolutionary Calibration of Uncertainty Quantification for Large Language Models},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO Companion '26)},
  year      = {2026},
  address   = {San Jose, Costa Rica},
  publisher = {ACM},
  doi        = {10.1145/3795101.3805422}
}


