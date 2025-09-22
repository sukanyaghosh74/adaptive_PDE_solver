import streamlit as st
import numpy as np
import os
import time
from solvers import FiniteDifferenceSolver, AdaptiveSolver
from utils import plot_surface, plot_final_price, plot_grid_density, save_csv, black_scholes_analytical

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

st.set_page_config(page_title="Adaptive PDE Option Pricing", layout="wide")
st.title("📈 Adaptive PDE Option Pricing for European Options")
st.markdown("""
This app solves the Black–Scholes PDE for European call/put options using both standard and adaptive finite difference methods. Adjust parameters, run the solvers, and visualize the results interactively!
""")

with st.sidebar:
    st.header("Model Parameters")
    S_max = st.number_input("Max Asset Price (S_max)", 100.0, 1000.0, 200.0, 10.0)
    K = st.number_input("Strike Price (K)", 1.0, 1000.0, 100.0, 1.0)
    r = st.number_input("Risk-free Rate (r)", 0.0, 1.0, 0.05, 0.01)
    sigma = st.number_input("Volatility (sigma)", 0.01, 2.0, 0.2, 0.01)
    T = st.number_input("Maturity (T, years)", 0.01, 10.0, 1.0, 0.01)
    option_type = st.selectbox("Option Type", ["call", "put"])
    st.header("Numerical Parameters")
    N_S = st.slider("Initial Grid Points (N_S)", 20, 400, 100, 10)
    N_t = st.slider("Time Steps (N_t)", 20, 400, 100, 10)
    st.header("Adaptive Settings")
    refine_tol = st.number_input("Refinement Tolerance", 1e-4, 1.0, 1e-2, format="%e")
    max_refine_steps = st.slider("Max Refinement Steps", 1, 10, 5, 1)
    st.markdown("---")
    run_uniform = st.button("Run Uniform Solver")
    run_adaptive = st.button("Run Adaptive Solver")

# Session state for results
if 'uniform_result' not in st.session_state:
    st.session_state['uniform_result'] = None
if 'adaptive_result' not in st.session_state:
    st.session_state['adaptive_result'] = None

col1, col2 = st.columns(2)

if run_uniform:
    with st.spinner("Running uniform grid solver..."):
        start = time.time()
        solver = FiniteDifferenceSolver(S_max, K, r, sigma, T, N_S, N_t, option_type)
        S_u, t_u, V_u = solver.solve()
        elapsed = time.time() - start
        V_exact = black_scholes_analytical(S_u, K, r, sigma, T, 0, option_type)
        l2_uniform = np.sqrt(np.mean((V_u[-1] - V_exact) ** 2))
        st.session_state['uniform_result'] = (S_u, t_u, V_u, V_exact, l2_uniform, elapsed)
        save_csv(S_u, t_u, V_u, os.path.join(RESULTS_DIR, "uniform_prices.csv"))
    st.success(f"Uniform solver done in {elapsed:.2f} seconds. L2 error: {l2_uniform:.4e}")

if run_adaptive:
    with st.spinner("Running adaptive grid solver..."):
        start = time.time()
        S_a_list, t_a, V_a_list, grids = AdaptiveSolver(S_max, K, r, sigma, T, N_S, N_t, option_type,
                                                        refine_tol=refine_tol, max_refine_steps=max_refine_steps).solve(return_grids=True)
        elapsed = time.time() - start
        V_exact = black_scholes_analytical(S_a_list[-1], K, r, sigma, T, 0, option_type)
        interp_adaptive = np.interp(st.session_state['uniform_result'][0] if st.session_state['uniform_result'] else S_a_list[-1], S_a_list[-1], V_a_list[-1])
        l2_adaptive = np.sqrt(np.mean((interp_adaptive - V_exact) ** 2))
        st.session_state['adaptive_result'] = (S_a_list, t_a, V_a_list, grids, l2_adaptive, elapsed)
        save_csv(S_a_list[-1], t_a, [v for v in V_a_list], os.path.join(RESULTS_DIR, "adaptive_prices.csv"))
    st.success(f"Adaptive solver done in {elapsed:.2f} seconds. L2 error: {l2_adaptive:.4e}")

# Show results
if st.session_state['uniform_result']:
    S_u, t_u, V_u, V_exact, l2_uniform, elapsed = st.session_state['uniform_result']
    with col1:
        st.subheader("Uniform Grid Results")
        st.write(f"L2 error: {l2_uniform:.4e}, Runtime: {elapsed:.2f} s")
        st.image(os.path.join(RESULTS_DIR, "uniform_surface.png"), caption="Uniform Grid Price Surface", use_column_width=True)
        st.download_button("Download Uniform CSV", data=open(os.path.join(RESULTS_DIR, "uniform_prices.csv"), "rb").read(), file_name="uniform_prices.csv")

if st.session_state['adaptive_result']:
    S_a_list, t_a, V_a_list, grids, l2_adaptive, elapsed = st.session_state['adaptive_result']
    with col2:
        st.subheader("Adaptive Grid Results")
        st.write(f"L2 error: {l2_adaptive:.4e}, Runtime: {elapsed:.2f} s")
        st.image(os.path.join(RESULTS_DIR, "adaptive_surface.png"), caption="Adaptive Grid Price Surface", use_column_width=True)
        st.image(os.path.join(RESULTS_DIR, "adaptive_grid_density.png"), caption="Adaptive Grid Density", use_column_width=True)
        st.download_button("Download Adaptive CSV", data=open(os.path.join(RESULTS_DIR, "adaptive_prices.csv"), "rb").read(), file_name="adaptive_prices.csv")

# Final price and comparison plot
if st.session_state['uniform_result'] and st.session_state['adaptive_result']:
    S_u, t_u, V_u, V_exact, l2_uniform, elapsed_u = st.session_state['uniform_result']
    S_a_list, t_a, V_a_list, grids, l2_adaptive, elapsed_a = st.session_state['adaptive_result']
    plot_final_price(S_u, V_u[-1], V_exact, S_a_list[-1], V_a_list[-1], K, os.path.join(RESULTS_DIR, "final_price_comparison.png"))
    st.image(os.path.join(RESULTS_DIR, "final_price_comparison.png"), caption="Final Option Price Comparison", use_column_width=True)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit. [Source code](https://github.com/your-repo)")
