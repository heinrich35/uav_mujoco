#!/usr/bin/env python3
"""Step 1: Convert GLB→USD (no stage modifications)."""
import argparse, shutil
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
sim_app = app.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
ROOT = Path("/home/heinz/isaaclab_uav")
for name in ["gosling","duck","swan"]:
    d = ROOT / "assets/pond/glb/usd" / name
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    glb = ROOT / f"assets/pond/glb/{name}.glb"
    c = MeshConverter(MeshConverterCfg(
        asset_path=str(glb), usd_dir=str(d), usd_file_name=f"{name}.usd",
        force_usd_conversion=True, make_instanceable=False))
    print(f"{name}: {c.usd_path}")
print("Step 1 done.")
sim_app.close()
