# duct3d — placeholder

The 3D duct/loitering-munition airframe loop lands here in the next pass:

- `duct_ffd.py` — PyGeM FFD control lattice over the baseline duct/airframe STL
  (assets already in the repo: `assets/shahed-136*`, `assets/propeller.blend`).
- `gmsh_duct.py` — gmsh 3D mesh of the morphed geometry (or snappyHexMesh via
  the OpenFOAM layer for internal duct flow).
- `su2_runner.py` — SU2_CFD wrapper (serial + `mpirun --bind-to none -np N`);
  the v8.5.0 build at `~/SU2-v8.5.0` already has MPI + `SU2_DOT`/`SU2_DEF`
  (adjoint gradients + mesh deformation) compiled in.

Nothing in this directory runs yet.
