import numpy as np
from grids import uniform_grid, adaptive_grid
from utils import payoff, apply_boundary_conditions

class FiniteDifferenceSolver:
    """
    Standard explicit finite difference solver for Black-Scholes PDE on a uniform grid.
    """
    def __init__(self, S_max, K, r, sigma, T, N_S, N_t, option_type='call'):
        self.S_max = S_max
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N_S = N_S
        self.N_t = N_t
        self.option_type = option_type

    def solve(self):
        S = uniform_grid(0, self.S_max, self.N_S)
        t = np.linspace(0, self.T, self.N_t + 1)
        dS = S[1] - S[0]
        dt = t[1] - t[0]

        V = np.zeros((self.N_t + 1, self.N_S))
        V[-1, :] = payoff(S, self.K, self.option_type)

        # Time stepping (backward in time)
        for n in reversed(range(self.N_t)):
            V[n, :] = V[n+1, :].copy()
            # Interior points
            for i in range(1, self.N_S - 1):
                Si = S[i]
                delta = (V[n+1, i+1] - V[n+1, i-1]) / (2 * dS)
                gamma = (V[n+1, i+1] - 2 * V[n+1, i] + V[n+1, i-1]) / (dS ** 2)
                V[n, i] = V[n+1, i] + dt * (
                    0.5 * self.sigma ** 2 * Si ** 2 * gamma +
                    self.r * Si * delta - self.r * V[n+1, i]
                )
            # Boundary conditions
            V[n, :] = apply_boundary_conditions(V[n, :], S, self.K, self.r, self.T - t[n], self.option_type)
        return S, t, V

class AdaptiveSolver(FiniteDifferenceSolver):
    """
    Adaptive finite difference solver with grid refinement near the strike.
    Stores S and V as lists for each time step to support variable grid sizes.
    """
    def __init__(self, S_max, K, r, sigma, T, N_S, N_t, option_type='call',
                 refine_tol=1e-2, max_refine_steps=5):
        super().__init__(S_max, K, r, sigma, T, N_S, N_t, option_type)
        self.refine_tol = refine_tol
        self.max_refine_steps = max_refine_steps

    def solve(self, return_grids=False):
        # Initial grid
        S = uniform_grid(0, self.S_max, self.N_S)
        t = np.linspace(0, self.T, self.N_t + 1)
        grids = [S.copy()]
        V_list = [None] * (self.N_t + 1)
        S_list = [None] * (self.N_t + 1)
        Vn = payoff(S, self.K, self.option_type)
        V_list[-1] = Vn.copy()
        S_list[-1] = S.copy()

        for n in reversed(range(self.N_t)):
            dt = t[1] - t[0]
            # Adaptive grid refinement
            S_curr = S.copy()
            V_curr = Vn.copy()
            for _ in range(self.max_refine_steps):
                grad = np.abs(np.gradient(V_curr, S_curr))
                refine_mask = grad > self.refine_tol
                if not np.any(refine_mask):
                    break
                S_new = adaptive_grid(S_curr, refine_mask)
                if len(S_new) == len(S_curr):
                    break
                V_curr = np.interp(S_new, S_curr, V_curr)
                S_curr = S_new
            grids.append(S_curr.copy())
            Vn1 = np.zeros_like(V_curr)
            dS = np.diff(S_curr)
            # Interior points
            for i in range(1, len(S_curr) - 1):
                Si = S_curr[i]
                dS_left = S_curr[i] - S_curr[i-1]
                dS_right = S_curr[i+1] - S_curr[i]
                # Non-uniform finite difference
                delta = (V_curr[i+1] - V_curr[i-1]) / (dS_right + dS_left)
                gamma = 2 * (
                    (V_curr[i+1] - V_curr[i]) / dS_right -
                    (V_curr[i] - V_curr[i-1]) / dS_left
                ) / (dS_right + dS_left)
                Vn1[i] = V_curr[i] + dt * (
                    0.5 * self.sigma ** 2 * Si ** 2 * gamma +
                    self.r * Si * delta - self.r * V_curr[i]
                )
            # Boundary conditions
            Vn1 = apply_boundary_conditions(Vn1, S_curr, self.K, self.r, self.T - t[n], self.option_type)
            Vn = Vn1.copy()
            V_list[n] = Vn.copy()
            S_list[n] = S_curr.copy()
        if return_grids:
            return S_list, t, V_list, grids
        return S_list, t, V_list
