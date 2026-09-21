"""pytest wrappers around the smoke checks (each skips if its dep is absent)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfoil2d"))


def test_xfoil_eval():
    pytest.importorskip("xfoil")
    import smoke_test
    smoke_test.xfoil_eval()


def test_env_rollout():
    pytest.importorskip("gymnasium")
    import smoke_test
    smoke_test.env_rollout()


def test_pygem_ffd():
    pytest.importorskip("pygem")
    import smoke_test
    smoke_test.pygem_ffd()


def test_gmsh_cmsh():
    pytest.importorskip("gmsh")
    import smoke_test
    smoke_test.gmsh_cmsh()


def test_sb3_short():
    pytest.importorskip("stable_baselines3")
    import smoke_test
    smoke_test.sb3_short(150)
