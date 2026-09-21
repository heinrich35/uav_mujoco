#!/usr/bin/env python3
"""Fix absolute texture paths → relative in articulation USDs."""
import argparse, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
sim_app = app.app

from pxr import Usd, Sdf
BASE = "/home/heinz/isaaclab_uav/assets/pond/glb/usd"
for name in ["gosling","duck","swan"]:
    path = f"{BASE}/{name}/{name}_articulation.usd"
    stage = Usd.Stage.Open(path)
    tex_dir = f"{BASE}/{name}/textures"
    fixed = 0
    for p in stage.TraverseAll():
        for attr in p.GetAttributes():
            try:
                v = attr.Get()
                if isinstance(v, Sdf.AssetPath) and hasattr(v, "path") and v.path and tex_dir in v.path:
                    attr.Set(Sdf.AssetPath(f"./textures/{os.path.basename(v.path)}"))
                    fixed += 1
            except: pass
    if fixed:
        stage.GetRootLayer().Export(path)
    print(f"{name}: {fixed} paths fixed")
sim_app.close()
