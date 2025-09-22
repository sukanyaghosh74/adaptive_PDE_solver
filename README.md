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
- **Beautiful Streamlit frontend** for interactive exploration

## Folder Structure

```
adaptive_pde_option_pricing/
├── main.py            # Entry point, runs examples
├── solvers.py         # PDE solver classes (adaptive & standard)
├── grids.py           # Grid generation and refinement logic
├── utils.py           # Helper functions (payoff, plots, Black-Scholes)
├── tests.py           # Unit tests for solvers
├── app.py             # Streamlit frontend
├── results/           # Stores plots and CSV results
└── README.md          # This file
```

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib
- streamlit

Install dependencies:
```bash
pip install numpy scipy matplotlib streamlit
```

## Usage

### Run the main example (CLI):
```bash
python main.py
```

### Run the beautiful frontend (recommended!):
```bash
streamlit run app.py
```

This will open a web app in your browser where you can adjust parameters, run solvers, and visualize results interactively.

### Run unit tests:
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

**Author:** Sukanya Ghosh
