import numpy as np
from solvers import FiniteDifferenceSolver, AdaptiveSolver
from utils import black_scholes_analytical

def test_uniform_solver_accuracy():
    S_max = 200
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    N_S = 200  # Increased for better accuracy
    N_t = 200  # Increased for better accuracy
    option_type = 'call'
    solver = FiniteDifferenceSolver(S_max, K, r, sigma, T, N_S, N_t, option_type)
    S, t, V = solver.solve()
    V_exact = black_scholes_analytical(S, K, r, sigma, T, 0, option_type)
    l2 = np.sqrt(np.mean((V[-1] - V_exact) ** 2))
    print(f"Uniform solver L2 error: {l2:.4e}")
    assert l2 < 2.0, "Uniform solver error too high!"

def test_adaptive_solver_accuracy():
    S_max = 200
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    N_S = 50
    N_t = 50
    option_type = 'call'
    solver = AdaptiveSolver(S_max, K, r, sigma, T, N_S, N_t, option_type,
                            refine_tol=1e-2, max_refine_steps=5)
    S, t, V = solver.solve()
    V_exact = black_scholes_analytical(S[-1], K, r, sigma, T, 0, option_type)
    l2 = np.sqrt(np.mean((V[-1] - V_exact) ** 2))
    print(f"Adaptive solver L2 error: {l2:.4e}")
    assert l2 < 2.0, "Adaptive solver error too high!"

if __name__ == "__main__":
    test_uniform_solver_accuracy()
    test_adaptive_solver_accuracy()
    print("All tests passed.")
