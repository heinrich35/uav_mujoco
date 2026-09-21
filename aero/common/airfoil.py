"""Airfoil parameterization for the 2D design loop: Kulfan CST + NACA helpers.

Coordinate convention (matches the xfoil-python binding's expectations):
    points run TE -> upper surface -> LE -> lower surface -> TE (clockwise),
    first/last points are the (open) trailing edge pair.
"""
import numpy as np
from scipy.special import binom


def cosine_spacing(n):
    """Cosine-spaced chord stations in [0, 1] (LE-clustered)."""
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n)))


def cst_basis(psi, n_coef):
    """Bernstein basis of the CST shape function, shape (len(psi), n_coef)."""
    psi = np.asarray(psi, float)
    r = np.arange(n_coef)
    K = binom(n_coef - 1, r)
    return K * psi[:, None] ** r * (1.0 - psi[:, None]) ** (n_coef - 1 - r)


def cst_surface(coeffs, n, n1=0.5, n2=1.0):
    """One CST surface. Returns (x_over_c, y_over_c)."""
    coeffs = np.asarray(coeffs, float)
    psi = cosine_spacing(n)
    C = psi ** n1 * (1.0 - psi) ** n2
    return psi, C * (cst_basis(psi, len(coeffs)) @ coeffs)


def cst_airfoil(c_upper, c_lower, n=160):
    """Full airfoil from lower/upper CST coefficient vectors -> (x, y)."""
    xu, yu = cst_surface(c_upper, n)
    xl, yl = cst_surface(c_lower, n)
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], -yl[1:]])
    return x, y


def fit_cst(psi, y_surface, n_coef):
    """Fit CST coefficients to one surface by linear least squares.

    Fit y DIRECTLY against the columns C*B_k — never divide by the class
    function. y/C is 0/0 at psi=0/1 and blows up like 1/(1-psi) near an open
    trailing edge; chasing that spike with lstsq wrecks the mid-chord
    (observed: fitted thickness 0.008 vs 0.053). Direct fit is well
    conditioned, and the fitted surface closes at the TE by construction
    (C=0 there — the NACA's 0.1%-chord open gap is simply not represented).
    """
    psi = np.asarray(psi, float)
    y_surface = np.asarray(y_surface, float)
    C = psi ** 0.5 * (1.0 - psi)
    A = C[:, None] * cst_basis(psi, n_coef)
    coeff, *_ = np.linalg.lstsq(A, y_surface, rcond=None)
    return coeff


def naca4_coords(t=0.12, n=160, sym=True):
    """NACA 4-digit (symmetric) coordinates, open trailing edge."""
    xc = cosine_spacing(n)
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                  + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)
    if sym:
        return np.concatenate([xc[::-1], xc[1:]]), np.concatenate([yt[::-1], -yt[1:]])
    raise NotImplementedError("cambered NACA not needed for the smoke loop")


def naca0012_cst(n_coef=6):
    """CST coefficient vector for a symmetric NACA0012 (upper == -lower)."""
    x, y = naca4_coords()
    n = (len(x) + 1) // 2
    xu, yu = x[:n], y[:n]
    xl, yl = x[n - 1:][::-1], -y[n - 1:][::-1]  # lower surface, LE -> TE, y>0
    c_up = fit_cst(xu, yu, n_coef)
    c_lo = fit_cst(xl, yl, n_coef)
    return c_up, c_lo


def thickness_camber(x, y):
    """(max_thickness, max_camber) from closed airfoil coordinates."""
    i_le = int(np.argmin(x))
    xu, yu = x[:i_le + 1], y[:i_le + 1]          # TE -> LE (upper)
    xl, yl = x[i_le:], y[i_le:]                  # LE -> TE (lower)
    yt = np.interp(xl[::-1], xu[::-1], yu[::-1])  # upper y at lower-surface x
    thickness = yt + yl[::-1]
    camber = 0.5 * (yt - yl[::-1])
    return float(thickness.max()), float(np.abs(camber).max())
