"""Airfoil polar evaluation backends.

Backends
--------
module     DARcorporation xfoil-python (in-process Fortran). Default.
           Installed from git with -fbounds-check/-ffpe-trap removed (see
           aero/setup_env_aero.sh) — upstream flags abort on XFoil's legacy
           out-of-bounds-by-1 Kutta-row write.
           NOTE: do NOT call repanel() on well-distributed input — it
           de-converges 319-point cosine coordinates (verified 2026-09-15);
           raw input matches XFoil's own paneled fixture to 3 decimals.
binary     apt /usr/bin/xfoil driven through a stdin script + polar parse.
           Fallback; requires the apt layer (aero/apt_layer.sh).
neuralfoil neuralfoil surrogate (--no-deps install). Microsecond evals for
           reward-shaping curricula; ~1% typical error vs XFoil.

All return a dict(cl=, cd=, cm=, converged=, backend=). NaN results are
normalized to converged=False.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

AERO_ROOT = Path(__file__).resolve().parents[1]
RESULT = lambda: AERO_ROOT / "results"  # noqa: E731


def eval_module(x, y, re, alpha, max_iter=100):
    from xfoil import XFoil
    from xfoil.model import Airfoil

    xf = XFoil()
    xf.print = False
    xf.airfoil = Airfoil(x=np.asarray(x, float), y=np.asarray(y, float))
    xf.Re = float(re)
    xf.max_iter = int(max_iter)
    cl, cd, cm, _cp = xf.a(float(alpha))
    conv = bool(np.isfinite(cl) and np.isfinite(cd) and cd > 0.0)
    return dict(cl=float(cl) if conv else np.nan,
                cd=float(cd) if conv else np.nan,
                cm=float(cm) if conv else np.nan,
                converged=conv, backend="module")


def eval_binary(x, y, re, alpha, max_iter=150):
    """Drive the apt xfoil executable; parse the polar file it writes."""
    xfoil_bin = "/usr/bin/xfoil"
    if not Path(xfoil_bin).exists():
        return dict(cl=np.nan, cd=np.nan, cm=np.nan, converged=False, backend="binary")

    with tempfile.TemporaryDirectory(dir=RESULT()) as td:
        td = Path(td)
        dat, pol = td / "af.dat", td / "af.pol"
        np.savetxt(dat, np.column_stack([x, y]), fmt="%.8f")
        script = (
            f"LOAD {dat}\n"
            "PANE\n"
            "OPER\n"
            f"VISC {re:.0f}\n"
            f"ITER {max_iter}\n"
            "PACC\n"
            f"{pol}\n\n"
            f"ALFA {alpha:.3f}\n"
            "PWRT\n"
            f"{pol}\n"
            "QUIT\n"
        )
        try:
            subprocess.run([xfoil_bin], input=script, text=True,
                           capture_output=True, timeout=30)
            lines = pol.read_text().strip().splitlines()
            hdr = next(i for i, ln in enumerate(lines) if ln.startswith("alpha"))
            cols = lines[hdr].split()
            row = lines[hdr + 2].split()  # hdr+1 is the dashes line
            rec = dict(zip(cols, map(float, row)))
        except Exception:
            return dict(cl=np.nan, cd=np.nan, cm=np.nan, converged=False, backend="binary")
    conv = np.isfinite(rec.get("CL", np.nan)) and rec.get("CD", 0.0) > 0
    return dict(cl=rec.get("CL", np.nan), cd=rec.get("CD", np.nan),
                cm=rec.get("CM", np.nan), converged=bool(conv), backend="binary")


def eval_neuralfoil(x, y, re, alpha):
    try:
        import neuralfoil as nf
    except ImportError:
        return dict(cl=np.nan, cd=np.nan, cm=np.nan, converged=False, backend="neuralfoil")
    aero = nf.get_aero_from_coordinates(
        np.stack([np.asarray(x, float), np.asarray(y, float)], axis=1),
        alpha=float(alpha), Re=float(re), model_size="xxlarge",
    )
    cl = float(np.asarray(aero["CL"]).item())  # vectorized API returns 0-d arrays
    cd = float(np.asarray(aero["CD"]).item())
    conv = np.isfinite(cl) and np.isfinite(cd) and cd > 0.0
    return dict(cl=cl, cd=cd, cm=float(np.asarray(aero["CM"]).item()),
                converged=conv, backend="neuralfoil")


_BACKENDS = {"module": eval_module, "binary": eval_binary, "neuralfoil": eval_neuralfoil}


def eval_polar(x, y, re, alpha, backend="module"):
    return _BACKENDS[backend](x, y, re, alpha)
