# Adaptive PDE Option Pricing

This project implements **adaptive finite difference solvers** for the Black–Scholes PDE for European call and put options, with automatic grid refinement near the strike price. It compares adaptive and standard uniform-grid methods, and provides visualizations and CSV outputs.

## Features

- **Finite Difference Solver** (uniform grid)
- **Adaptive Grid Solver** (refines near strike)
- **Automatic grid refinement** based on solution gradient/curvature
- **Time stepping**: fixed and adaptive Δt
- **Vectorized numpy** for performance
- **Plots**: price surface, final price, grid density
- **CSV output** of results
- **Unit tests** for accuracy

## Folder Structure

```
adaptive_pde_option_pricing/
├── main.py            # Entry point, runs examples
├── solvers.py         # PDE solver classes (adaptive & standard)
├── grids.py           # Grid generation and refinement logic
├── utils.py           # Helper functions (payoff, plots, Black-Scholes)
├── tests.py           # Unit tests for solvers
├── results/           # Stores plots and CSV results
└── README.md          # This file
```

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Usage

Run the main example:
```bash
python main.py
```

This will:
- Solve the Black–Scholes PDE for a European call (S ∈ [0, 200], K=100, r=0.05, σ=0.2, T=1)
- Compare adaptive and uniform solvers
- Save plots and CSVs in `results/`

## Testing

Run unit tests:
```bash
python tests.py
```

## Outputs

- **PNG plots**: price surfaces, final price, grid density
- **CSV**: option prices for S and t

## References

- Black–Scholes PDE: https://en.wikipedia.org/wiki/Black–Scholes_equation
- Adaptive mesh refinement: https://en.wikipedia.org/wiki/Adaptive_mesh_refinement

---

**Author:** [Your Name]
