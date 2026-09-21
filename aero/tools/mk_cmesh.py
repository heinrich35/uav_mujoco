"""Build a 2D C-mesh around an airfoil: gmsh -> meshio -> SU2 .su2 format.

meshio reads the gmsh file, but its .su2 writer drops the physical-marker
names in current versions — so the .su2 text is written here directly from
the meshio mesh (format: NDIME/NPOIN/NELEM/NMARK with MARKER_TAG sections).

Usage:
    python mk_cmesh.py out_dir            # NACA0012 demo mesh
    python mk_cmesh.py out_dir coords.dat # two-column airfoil coordinates
"""
import sys
from pathlib import Path

import numpy as np


def build_cmsh(x, y, out_msh, far=15.0, le_char=0.01):
    """Circle-farfield mesh (good enough for Euler smoke cases)."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cmesh")

        pts = [gmsh.model.geo.addPoint(float(px), float(py), 0.0, le_char)
               for px, py in zip(x, y)]
        foil = gmsh.model.geo.addSpline(pts)                    # TE -> LE -> TE
        te = gmsh.model.geo.addLine(pts[-1], pts[0])            # close at the TE
        foil_loop = gmsh.model.geo.addCurveLoop([foil, te])

        center = gmsh.model.geo.addPoint(0.5, 0.0, 0.0, far * 0.2)
        circ_pts = [gmsh.model.geo.addPoint(0.5 + far * np.cos(a), far * np.sin(a),
                                            0.0, far * 0.2)
                    for a in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)]
        arcs = [gmsh.model.geo.addCircleArc(circ_pts[i], center, circ_pts[(i + 1) % 4])
                for i in range(4)]
        circ_loop = gmsh.model.geo.addCurveLoop(arcs)

        surf = gmsh.model.geo.addPlaneSurface([circ_loop, foil_loop])
        gmsh.model.geo.synchronize()

        # GOTCHA: with any physical group defined, gmsh.write drops elements
        # that belong to NO group — the triangles must be grouped or the .msh
        # will contain only the boundary lines.
        gmsh.model.addPhysicalGroup(2, [surf], 1)
        gmsh.model.setPhysicalName(2, 1, "fluid")

        gmsh.model.addPhysicalGroup(1, [foil, te], 1)
        gmsh.model.addPhysicalGroup(1, arcs, 2)
        gmsh.model.setPhysicalName(1, 1, "airfoil")
        gmsh.model.setPhysicalName(1, 2, "farfield")

        fld = gmsh.model.mesh.field
        fld.add("Distance", 1)
        fld.setNumbers(1, "CurvesList", [foil, te])
        fld.add("Threshold", 2)
        fld.setNumber(2, "InField", 1)
        fld.setNumber(2, "SizeMin", le_char)
        fld.setNumber(2, "SizeMax", far * 0.05)
        fld.setNumber(2, "DistMin", 0.5)
        fld.setNumber(2, "DistMax", far)
        fld.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay for 2D
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_msh))
    finally:
        gmsh.finalize()


def write_su2(msh_path, su2_path):
    """meshio read -> minimal .su2 writer (keeps physical marker names)."""
    import meshio

    mesh = meshio.read(msh_path)
    phys = mesh.cell_data.get("gmsh:physical")
    # field_data values are [tag, dim]; physical tags are PER-DIMENSION in
    # gmsh, so the dim-1 lookup for line markers must ignore the dim-2 group
    # (both can carry tag 1).
    names = {v[0]: k for k, v in (mesh.field_data or {}).items() if v[1] == 1}

    points = mesh.points[:, :2]
    with open(su2_path, "w") as f:
        f.write("NDIME= 2\n")
        f.write(f"NPOIN= {len(points)}\n")
        for i, (px, py) in enumerate(points):
            f.write(f"{px:.10e}\t{py:.10e}\t{i + 1}\n")

        tris = [(cb.data, pd) for cb, pd in
                zip(mesh.cells, phys) if cb.type == "triangle"]
        n_tri = sum(len(d) for d, _ in tris)
        f.write(f"NELEM= {n_tri}\n")
        for data, _ in tris:
            for a, b, c in data:
                f.write(f"5\t{a}\t{b}\t{c}\n")  # 5 = VTK_TRIANGLE (su2 uses VTK ids)

        markers = [(cb.data, pd) for cb, pd in zip(mesh.cells, phys) if cb.type == "line"]
        f.write(f"NMARK= {len(markers)}\n")
        for data, pd in markers:
            tag = names[int(np.atleast_1d(pd)[0])]
            f.write(f"MARKER_TAG= {tag}\n")
            f.write(f"MARKER_ELEMS= {len(data)}\n")
            for a, b in data:
                f.write(f"3\t{a}\t{b}\n")  # 3 = VTK_LINE


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/cmsh")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2:
        xy = np.loadtxt(sys.argv[2])
        x, y = xy[:, 0], xy[:, 1]
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from common.airfoil import naca4_coords
        x, y = naca4_coords()

    msh, su2 = out_dir / "cmesh.msh", out_dir / "cmesh.su2"
    build_cmsh(x, y, msh)
    write_su2(msh, su2)
    print(f"wrote {msh} and {su2}")


if __name__ == "__main__":
    main()
