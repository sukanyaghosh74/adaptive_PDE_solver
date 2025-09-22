import os
import numpy as np
from solvers import FiniteDifferenceSolver, AdaptiveSolver
from utils import plot_surface, plot_final_price, plot_grid_density, save_csv, black_scholes_analytical

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_example():
    # Parameters
    S_max = 200
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    option_type = 'call'
    N_S = 100  # initial grid points in S
    N_t = 100  # time steps

    # Uniform grid solver
    print("Running uniform grid solver...")
    uniform_solver = FiniteDifferenceSolver(S_max, K, r, sigma, T, N_S, N_t, option_type)
    S_u, t_u, V_u = uniform_solver.solve()
    print("Uniform grid solver done.")

    # Adaptive grid solver
    print("Running adaptive grid solver...")
    S_a_list, t_a, V_a_list, grids = AdaptiveSolver(S_max, K, r, sigma, T, N_S, N_t, option_type,
                                     refine_tol=1e-2, max_refine_steps=5).solve(return_grids=True)
    print("Adaptive grid solver done.")

    # Analytical solution for final time
    V_exact = black_scholes_analytical(S_u, K, r, sigma, T, 0, option_type)

    # Save CSVs (for adaptive, save only the final time step)
    save_csv(S_u, t_u, V_u, os.path.join(RESULTS_DIR, "uniform_prices.csv"))
    save_csv(S_a_list[-1], t_a, [v for v in V_a_list], os.path.join(RESULTS_DIR, "adaptive_prices.csv"))

    # Plots
    plot_surface(S_u, t_u, V_u, "Uniform Grid Price Surface", os.path.join(RESULTS_DIR, "uniform_surface.png"))
    plot_surface(S_a_list, t_a, V_a_list, "Adaptive Grid Price Surface", os.path.join(RESULTS_DIR, "adaptive_surface.png"))
    plot_final_price(S_u, V_u[-1], V_exact, S_a_list[-1], V_a_list[-1], K, os.path.join(RESULTS_DIR, "final_price_comparison.png"))
    plot_grid_density(grids, K, os.path.join(RESULTS_DIR, "adaptive_grid_density.png"))

    # Print error metrics
    interp_adaptive = np.interp(S_u, S_a_list[-1], V_a_list[-1])
    l2_uniform = np.sqrt(np.mean((V_u[-1] - V_exact) ** 2))
    l2_adaptive = np.sqrt(np.mean((interp_adaptive - V_exact) ** 2))
    print(f"L2 error (uniform): {l2_uniform:.4e}")
    print(f"L2 error (adaptive): {l2_adaptive:.4e}")

if __name__ == "__main__":
    run_example()
