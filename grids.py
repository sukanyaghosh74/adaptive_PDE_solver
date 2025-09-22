import numpy as np

def uniform_grid(S_min, S_max, N):
    """Generate a uniform grid in S."""
    return np.linspace(S_min, S_max, N)

def adaptive_grid(S, refine_mask):
    """
    Refine grid S where refine_mask is True.
    Insert new points between points where either is True.
    """
    S_new = [S[0]]
    for i in range(1, len(S)):
        S_new.append(S[i])
        if refine_mask[i-1] or refine_mask[i]:
            S_new.append(0.5 * (S[i-1] + S[i]))
    S_new = np.unique(S_new)
    S_new.sort()
    return S_new
