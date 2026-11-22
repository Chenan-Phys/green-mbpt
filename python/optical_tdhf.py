"""Long-wavelength optical response of a real, stable time-dependent HF kernel.

Inputs are thermally weighted forward transitions. A is the resonant block,
B couples the forward and reverse transitions, and j is the real current
matrix element times the square root of the occupation difference. A charge
channel h adds +h h.T to A and -h h.T to B. This sign convention describes
a real current and an imaginary, odd transition charge in the same basis.
"""

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular


def current_modes(a, b, current, charge_channels):
    """Return pole energies, optical amplitudes and transformed charge channels.

    The interband polarizability is v.T (diag(energy**2) + U U.T - z**2)^-1 v.
    A separate diamagnetic sum and the missing intraband weight determine the
    Drude term. The factorization keeps the common Hartree field available for
    coupling additional transitions after this solve.
    """
    a, b, current, charge_channels = map(np.asarray, (a, b, current, charge_channels))
    n = len(current)
    if (current.ndim != 1 or a.shape != (n, n) or b.shape != a.shape
            or charge_channels.ndim != 2 or charge_channels.shape[0] != n or not n):
        raise ValueError("Incompatible optical transition dimensions")
    if any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in (a, b, current, charge_channels)):
        raise ValueError("This optical kernel requires finite real inputs")
    if any(not np.allclose(x, x.T, atol=1e-10, rtol=1e-10) for x in (a, b)):
        raise ValueError("A and B must be symmetric")
    # In these coordinates the current susceptibility has a static term
    # 2*j.T @ inv(A+B) @ j. Subtracting it gives the optical polarizability.
    try:
        l = cholesky(a + b, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Unstable or singular A+B optical block") from exc
    squared, modes = eigh(l.T @ (a - b) @ l)
    if squared[0] <= 0:
        raise ValueError(f"Unstable optical mode, squared energy {squared[0]:.6g}")
    amplitude = np.sqrt(2.) * (modes.T @ solve_triangular(l, current, lower=True))
    channels = np.sqrt(2.) * (modes.T @ (l.T @ charge_channels))
    return np.sqrt(squared), amplitude, channels
