# Convergence and Accuracy of PINNs vs. Classical Finite Difference Schemes for BVPs

**MATH 449 Final Project** — University of Victoria, April 2026

## Overview

This project compares Physics-Informed Neural Networks (PINNs) with classical Finite Difference Methods (FDM) for solving boundary value problems (BVPs). We evaluate convergence rates, accuracy, conditioning, and computational cost across three test problems — two linear and one nonlinear.

## Repository Structure

```
├── code/                   # Implementation
│   ├── main.ipynb          # Main notebook (FDM, PINNs, comparisons)
│   ├── Outputs/            # Generated figures and results
│   └── requirements.txt    # Python dependencies
├── presentation/           # Beamer slides (LaTeX)
│   ├── slides.tex
│   ├── slides.pdf
│   └── beamerthemeUVic.sty
└── report/                 # Final paper (LaTeX)
    ├── main.tex
    ├── main.pdf
    └── refs.bib
```

## Setup

```bash
pip install -r code/requirements.txt
```

Dependencies: NumPy, SciPy, Matplotlib, PyTorch, Jupyter

## Methods

- **FDM**: Second-order central differences, Richardson extrapolation, condition number analysis
- **Nonlinear solvers**: Newton shooting, bisection shooting, Picard iteration, Newton FD
- **PINNs**: Fully connected networks trained with Adam + L-BFGS, collocation point sweeps

## Author

Evan Strasdin (V00907185)
