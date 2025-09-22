import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def payoff(S, K, option_type='call'):
    """Payoff function for European call/put."""
    if option_type == 'call':
        return np.maximum(S - K, 0)
    else:
        return np.maximum(K - S, 0)

def apply_boundary_conditions(V, S, K, r, tau, option_type):
    """Apply boundary conditions for Black-Scholes PDE."""
    if option_type == 'call':
        V[0] = 0
        V[-1] = S[-1] - K * np.exp(-r * tau)
    else:
        V[0] = K * np.exp(-r * tau)
        V[-1] = 0
    return V

def plot_surface(S, t, V, title, filename):
    """Plot option price surface. Handles both uniform and adaptive (list) S and V."""
    import matplotlib.tri as mtri
    plt.figure(figsize=(8, 6))
    if isinstance(S, list) or (hasattr(S, '__len__') and isinstance(S[0], (np.ndarray, list))):
        # Adaptive: S is a list of arrays, V is list of arrays (len = N_t+1)
        points = []
        values = []
        times = []
        for i, (Si, Vi) in enumerate(zip(S, V)):
            ti = t[i] if hasattr(t, '__len__') else t
            points.extend([(s, ti) for s in Si])
            values.extend(Vi)
            times.extend([ti]*len(Si))
        points = np.array(points)
        values = np.array(values)
        triang = mtri.Triangulation(points[:,0], points[:,1])
        plt.tricontourf(triang, values, 50, cmap='viridis')
    else:
        T, S_grid = np.meshgrid(t, S)
        plt.contourf(S_grid, T, V.T, 50, cmap='viridis')
    plt.xlabel('Asset Price S')
    plt.ylabel('Time t')
    plt.title(title)
    plt.colorbar(label='Option Price')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_final_price(S_u, V_u, V_exact, S_a, V_a, K, filename):
    """Plot final option price vs asset price for both solvers and analytical."""
    plt.figure(figsize=(8, 5))
    plt.plot(S_u, V_u, label='Uniform Grid', lw=2)
    plt.plot(S_a, V_a, label='Adaptive Grid', lw=2)
    plt.plot(S_u, V_exact, '--', label='Analytical', lw=2)
    plt.axvline(K, color='k', ls=':', label='Strike')
    plt.xlabel('Asset Price S')
    plt.ylabel('Option Price')
    plt.title('Final Option Price vs Asset Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_grid_density(grids, K, filename):
    """Plot grid density vs asset price for adaptive solver."""
    plt.figure(figsize=(8, 5))
    for i, S in enumerate(grids[::max(1, len(grids)//10)]):
        plt.plot(S, np.ones_like(S) * i, '|', color='b', markersize=10, alpha=0.5)
    plt.axvline(K, color='r', ls='--', label='Strike')
    plt.xlabel('Asset Price S')
    plt.ylabel('Refinement Step')
    plt.title('Adaptive Grid Density Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_csv(S, t, V, filename):
    """Save option prices to CSV. Handles both uniform and adaptive (list) S and V."""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if isinstance(S, list) and isinstance(V, list):
            # Adaptive: S is list of arrays, V is list of arrays
            maxlen = max(len(s) for s in S)
            header = ['t'] + [f'S_{i}' for i in range(maxlen)]
            writer.writerow(header)
            for i, (ti, Si, Vi) in enumerate(zip(t, S, V)):
                row = [ti] + list(Vi) + [''] * (maxlen - len(Vi))
                writer.writerow(row)
        else:
            writer.writerow(['t'] + list(S))
            for i, ti in enumerate(t):
                writer.writerow([ti] + list(V[i]))

def black_scholes_analytical(S, K, r, sigma, T, t, option_type='call'):
    """Analytical Black-Scholes price for European option."""
    tau = T - t
    S = np.array(S)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        price = np.where(np.isnan(price), 0, price)
    return price
