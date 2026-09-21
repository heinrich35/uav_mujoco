#!/usr/bin/env python3
"""Convert distractor GLBs (log, lilypad, turtle, rock) to textured USD.

Steps:
  1. Convert GLB → USD via MeshConverter
  2. Replace MDL shaders → UsdPreviewSurface + relative texture paths
  3. Add collision geometry

Usage: ./isaaclab.sh -p scripts/pond/convert_distractors.py --headless
"""
import argparse, os, shutil
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
sim_app = app.app

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

ROOT = Path("/home/heinz/isaaclab_uav")
GLB_DIR = ROOT / "assets/pond/glb"
USD_DIR = GLB_DIR / "usd"

DISTRACTORS = ["log", "lilypad", "turtle", "rock"]

# Collision specs per distractor
#   type: "sphere" → radius,  "box" → (sx, sy, sz) scale on unit cube
COLLISION = {
    "turtle":  {"type": "sphere", "radius": 0.3},
    "log":     {"type": "box",    "scale": (0.5, 0.2, 0.2)},
    "lilypad": {"type": "sphere", "radius": 0.3},
    "rock":    {"type": "box",    "scale": (0.5, 0.5, 0.5)},
}

for name in DISTRACTORS:
    glb_path = GLB_DIR / f"{name}.glb"
    obj_dir = USD_DIR / name

    # Step 1 — Convert GLB → USD
    if obj_dir.exists():
        shutil.rmtree(obj_dir)
    obj_dir.mkdir(parents=True)

    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")

    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    c = MeshConverter(MeshConverterCfg(
        asset_path=str(glb_path),
        usd_dir=str(obj_dir),
        usd_file_name=f"{name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
    ))
    mesh_usd = c.usd_path
    print(f"  [1/5] Converted GLB → USD: {mesh_usd}")

    # Step 2 — Fix materials: MDL → UsdPreviewSurface + texture bindings
    stage = Usd.Stage.Open(mesh_usd)
    if not stage:
        print(f"  ERROR: cannot open stage for {name}")
        continue

    tex_dir = str(obj_dir / "textures")

    # Collect textures and color factors from original MDL materials
    textures = {}
    color_factor = None  # (r, g, b) fallback when no textures
    metallic_val = 0.0
    roughness_val = 0.7
    mesh_targets = []
    prims = list(stage.TraverseAll())  # materialize once
    for prim in prims:
        pt = prim.GetTypeName()
        if pt == "Mesh":
            mesh_targets.append(prim.GetPath())
        if pt == "Material":
            for child in prim.GetAllChildren():
                if child.GetTypeName() == "Shader":
                    # Check for texture
                    ta = child.GetAttribute("inputs:texture")
                    if ta:
                        try:
                            v = ta.Get()
                            if v and hasattr(v, "path") and v.path:
                                textures[child.GetName()] = f"textures/{Path(v.path).name}"
                        except Exception:
                            pass
                    # Check for base color factor (used when no texture)
                    if color_factor is None:
                        bcf = child.GetAttribute("inputs:base_color_factor")
                        if bcf:
                            try:
                                v = bcf.Get()
                                if v:
                                    color_factor = v
                            except Exception:
                                pass
                    # Extract metallic/roughness from first material
                    if metallic_val == 0.0:
                        mf = child.GetAttribute("inputs:metallic_factor")
                        if mf:
                            try:
                                v = mf.Get()
                                if v is not None:
                                    metallic_val = v
                            except Exception:
                                pass
                    if roughness_val == 0.7:
                        rf = child.GetAttribute("inputs:roughness_factor")
                        if rf:
                            try:
                                v = rf.Get()
                                if v is not None:
                                    roughness_val = v
                            except Exception:
                                pass

    has_tex = len(textures) > 0
    has_color = color_factor is not None
    print(f"  [2/5] Textures: {len(textures)}, color_factor: {color_factor}, "
          f"meshes: {len(mesh_targets)}, metallic={metallic_val:.2f}, roughness={roughness_val:.2f}")

    # Fix absolute texture paths → relative (from original conversion)
    fixed_paths = 0
    for prim in prims:
        for attr in prim.GetAttributes():
            try:
                v = attr.Get()
                if isinstance(v, Sdf.AssetPath) and hasattr(v, "path") and v.path:
                    p = v.path
                    if tex_dir in p:
                        rel = f"./textures/{Path(p).name}"
                        attr.Set(Sdf.AssetPath(rel))
                        fixed_paths += 1
            except Exception:
                pass
    if fixed_paths:
        print(f"         Fixed {fixed_paths} absolute texture path(s)")

    # Create new UsdPreviewSurface material at /{name}/Looks/PBR_Material
    root = stage.GetPrimAtPath(f"/{name}")
    rp = str(root.GetPath()) if root else f"/{name}"

    looks_path = Sdf.Path(f"{rp}/Looks")
    mat_path = looks_path.AppendPath("PBR_Material")
    mat = UsdShade.Material.Define(stage, mat_path)

    # Preview surface shader
    pv_path = mat_path.AppendPath("preview")
    pv = UsdShade.Shader.Define(stage, pv_path)
    pv.CreateIdAttr("UsdPreviewSurface")
    (pv.GetInput("metallic") or pv.CreateInput("metallic", Sdf.ValueTypeNames.Float)).Set(metallic_val)
    (pv.GetInput("roughness") or pv.CreateInput("roughness", Sdf.ValueTypeNames.Float)).Set(roughness_val)
    pv_surface = pv.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(pv_surface)

    # Link diffuse texture or set solid base color
    tk = "baseColorTex" if "baseColorTex" in textures else (list(textures.keys())[0] if textures else None)
    if tk:
        tex_path = mat_path.AppendPath("diffuseTex")
        tex = UsdShade.Shader.Define(stage, tex_path)
        tex.CreateIdAttr("UsdUVTexture")
        (tex.GetInput("file") or tex.CreateInput("file", Sdf.ValueTypeNames.Asset)).Set(
            Sdf.AssetPath(f"./{textures[tk]}")
        )
        (tex.GetInput("wrapS") or tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token)).Set("repeat")
        (tex.GetInput("wrapT") or tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token)).Set("repeat")
        tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        dc_in = pv.GetInput("diffuseColor") or pv.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        dc_in.ConnectToSource(tex_out)

        # UV reader
        rd_path = mat_path.AppendPath("st_reader")
        rd = UsdShade.Shader.Define(stage, rd_path)
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        (rd.GetInput("varname") or rd.CreateInput("varname", Sdf.ValueTypeNames.Token)).Set("st")
        rd_out = rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        st_in = tex.GetInput("st") or tex.CreateInput("st", Sdf.ValueTypeNames.Float2)
        st_in.ConnectToSource(rd_out)
    elif has_color:
        # No texture — use extracted base_color_factor
        dc_in = pv.GetInput("diffuseColor") or pv.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        dc_in.Set(tuple(color_factor))

    # Bind all meshes to the new material
    for mesh_target in mesh_targets:
        mesh = stage.GetPrimAtPath(mesh_target)
        if mesh:
            mesh.CreateRelationship("material:binding", False).SetTargets([mat_path])

    # Set mesh purpose to "render" (collision is handled by dedicated prim below)
    for prim in prims:
        if prim.GetTypeName() == "Mesh":
            (prim.GetAttribute("purpose") or prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token, False)).Set("render")

    print(f"  [3/5] Material fixed: UsdPreviewSurface + texture bindings")

    # ------------------------------------------------------------------
    # Step 4 — Add RigidBodyAPI to root prim (needed for physics)
    # ------------------------------------------------------------------
    UsdPhysics.RigidBodyAPI.Apply(root)
    root.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, False).Set(True)
    root.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, False).Set(True)

    # PhysX-specific: immovable, no depenetration velocity
    for an, at, v in [
        ("physxRigidBodyAPI:kinematic", Sdf.ValueTypeNames.Bool, True),
        ("physxRigidBodyAPI:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, 0.0),
        ("physxRigidBodyAPI:enableGyroscopicForces", Sdf.ValueTypeNames.Bool, False),
    ]:
        a = root.GetAttribute(an) or root.CreateAttribute(an, at, False)
        a.Set(v)

    print(f"  [4/5] Added RigidBodyAPI (kinematic) to {rp}")

    # ------------------------------------------------------------------
    # Step 5 — Add collision geometry
    # ------------------------------------------------------------------
    col_spec = COLLISION[name]
    col_type = col_spec["type"]
    col_path = Sdf.Path(f"{rp}/collision")

    if col_type == "sphere":
        col_prim = UsdGeom.Sphere.Define(stage, col_path)
        col_prim.CreateRadiusAttr().Set(col_spec["radius"])
        desc = f"sphere r={col_spec['radius']}"
    else:  # box
        col_prim = UsdGeom.Cube.Define(stage, col_path)
        sx, sy, sz = col_spec["scale"]
        UsdGeom.Xformable(col_prim).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set((sx, sy, sz))
        desc = f"box {sx}×{sy}×{sz}"

    # Center collision at origin of the object
    UsdGeom.Xformable(col_prim).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set((0, 0, 0))

    # Apply collision API
    UsdPhysics.CollisionAPI.Apply(col_prim.GetPrim())
    col_prim.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False).Set(True)
    col_prim.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")

    print(f"  [5/5] Collision: {desc} at {col_path}")

    # Save
    stage.GetRootLayer().Export(mesh_usd)
    print(f"  Saved: {mesh_usd}")

print(f"\n{'='*50}")
print("  All distractors rebuilt successfully.")
print(f"{'='*50}")
sim_app.close()
