#!/usr/bin/env python3
"""Step 2: Fix materials, add articulation APIs, collision spheres."""
import argparse, os
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
sim_app = app.app

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

ROOT = Path("/home/heinz/isaaclab_uav")
BIRDS = {"gosling": (4.0, 0.2), "duck": (5.0, 0.3), "swan": (8.0, 0.4)}

for name, (mass, r) in BIRDS.items():
    d = ROOT / "assets/pond/glb/usd" / name
    mesh_path = str(d / f"{name}.usd")
    art_path = str(d / f"{name}_articulation.usd")
    tex_dir = str(d / "textures")

    print(f"\n=== {name} ===")
    stage = Usd.Stage.Open(mesh_path)
    if not stage:
        print("  ERROR: cannot open stage")
        continue

    root = stage.GetPrimAtPath(f"/{name}")
    rp = str(root.GetPath())

    # Collect texture references
    textures = {}
    mesh_target = None
    prims = list(stage.TraverseAll())  # materialize once
    for prim in prims:
        pt = prim.GetTypeName()
        if pt == "Mesh" and mesh_target is None:
            mesh_target = prim.GetPath()
        if pt == "Material":
            for child in prim.GetAllChildren():
                if child.GetTypeName() == "Shader":
                    ta = child.GetAttribute("inputs:texture")
                    if ta:
                        try:
                            v = ta.Get()
                            if v and hasattr(v, "path") and v.path:
                                textures[child.GetName()] = f"textures/{Path(v.path).name}"
                        except: pass

    print(f"  Found {len(textures)} textures, mesh at {mesh_target}")

    # Fix texture paths
    for prim in prims:
        for attr in prim.GetAttributes():
            try:
                v = attr.Get()
                if isinstance(v, Sdf.AssetPath) and hasattr(v, "path") and v.path:
                    p = v.path
                    if tex_dir in p:
                        rel = f"./textures/{Path(p).name}"
                        attr.Set(Sdf.AssetPath(rel))
            except: pass

    # Create new material at /{name}/Looks/PBR_Material
    looks_path = Sdf.Path(f"{rp}/Looks")
    mat_path = looks_path.AppendPath("PBR_Material")
    mat = UsdShade.Material.Define(stage, mat_path)

    # Preview shader
    pv_path = mat_path.AppendPath("preview")
    pv = UsdShade.Shader.Define(stage, pv_path)
    pv.CreateIdAttr("UsdPreviewSurface")
    (pv.GetInput("metallic") or pv.CreateInput("metallic", Sdf.ValueTypeNames.Float)).Set(0.0)
    (pv.GetInput("roughness") or pv.CreateInput("roughness", Sdf.ValueTypeNames.Float)).Set(0.7)
    pv_surface = pv.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(pv_surface)

    tk = "baseColorTex" if "baseColorTex" in textures else (list(textures.keys())[0] if textures else None)
    if tk:
        tex_path = mat_path.AppendPath("diffuseTex")
        tex = UsdShade.Shader.Define(stage, tex_path)
        tex.CreateIdAttr("UsdUVTexture")
        (tex.GetInput("file") or tex.CreateInput("file", Sdf.ValueTypeNames.Asset)).Set(Sdf.AssetPath(f"./{textures[tk]}"))
        (tex.GetInput("wrapS") or tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token)).Set("repeat")
        (tex.GetInput("wrapT") or tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token)).Set("repeat")
        tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        dc_in = pv.GetInput("diffuseColor") or pv.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        dc_in.ConnectToSource(tex_out)

        rd_path = mat_path.AppendPath("st_reader")
        rd = UsdShade.Shader.Define(stage, rd_path)
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        (rd.GetInput("varname") or rd.CreateInput("varname", Sdf.ValueTypeNames.Token)).Set("st")
        rd_out = rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        st_in = tex.GetInput("st") or tex.CreateInput("st", Sdf.ValueTypeNames.Float2)
        st_in.ConnectToSource(rd_out)

    # Bind mesh to new material
    if mesh_target:
        mesh = stage.GetPrimAtPath(mesh_target)
        if mesh:
            mesh.CreateRelationship("material:binding", False).SetTargets([mat_path])

    # Articulation APIs
    UsdPhysics.ArticulationRootAPI.Apply(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)
    I_val = 0.4 * mass * r * r
    ma = UsdPhysics.MassAPI(root)
    ma.CreateMassAttr().Set(mass)
    ma.CreateCenterOfMassAttr().Set((0, 0, -0.02))
    ma.CreateDiagonalInertiaAttr().Set((I_val, I_val, I_val))
    for an, at, v in [
        ("physxArticulationAPI:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, False),
        ("physxArticulationAPI:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 8),
        ("physxArticulationAPI:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 2),
        ("physxRigidBodyAPI:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, 5.0),
        ("physxRigidBodyAPI:enableGyroscopicForces", Sdf.ValueTypeNames.Bool, True),
        ("physxRigidBodyAPI:angularConstraint", Sdf.ValueTypeNames.Token, "none"),
        ("physxRigidBodyAPI:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 8),
        ("physxRigidBodyAPI:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 2),
    ]:
        a = root.GetAttribute(an) or root.CreateAttribute(an, at, False); a.Set(v)

    # Mesh render only
    for prim in prims:
        if prim.GetTypeName() == "Mesh":
            (prim.GetAttribute("purpose") or prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token, False)).Set("render")
            for api in ("PhysxCollisionAPI", "PhysicsCollisionAPI"):
                try: prim.RemoveAPI(api)
                except: pass

    # Collision sphere
    sp = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{rp}/collision_sphere"))
    sp.CreateRadiusAttr().Set(r)
    UsdGeom.Xformable(sp).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set((0, 0, 0))
    UsdPhysics.CollisionAPI.Apply(sp.GetPrim())
    sp.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False).Set(True)
    sp.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")

    stage.Export(art_path)
    print(f"  Saved: {art_path}")

print("\nDone.")
sim_app.close()
