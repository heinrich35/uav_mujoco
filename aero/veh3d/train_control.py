"""Fast loop: train the motor-control policy for ONE design — SB3 PPO, CPU.

Usage (source aero/env.sh first so CUDA stays invisible):
    python .../veh3d/train_control.py --steps 300000
    python .../veh3d/train_control.py --set geometry.span=1.6 --steps 60000
    python .../veh3d/train_control.py --no-train --model results/veh3d/control_x/model.zip
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_overrides(items):
    import yaml
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def train_once(cfg, overrides, steps, n_envs, seed, out_dir, verbose=1,
               eval_episodes=5):
    """Train one policy for one design; returns (model, metrics, out_dir, geom)."""
    import torch
    assert not torch.cuda.is_available(), (
        "GPU visible — source aero/env.sh (CUDA_VISIBLE_DEVICES='') first")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from env_veh import VehicleEnv
    from geometry import build, overrides_to_cfg

    cfg_o = overrides_to_cfg(cfg, overrides)
    geom = build(cfg_o)                      # one shared, read-only geometry
    assert geom.polar_ok, (
        f"section polar failed under {sys.executable} — NeuralFoil missing "
        f"(wrong python?) — source aero/env.sh")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def make():
        return VehicleEnv(geom=geom, cfg=cfg_o, seed=seed)

    env = (SubprocVecEnv([make] * n_envs, start_method="fork")
           if n_envs > 1 else make())
    t0 = time.perf_counter()
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=verbose,
                n_steps=200, gamma=0.99, batch_size=400)
    model.learn(total_timesteps=steps, progress_bar=False)
    train_s = time.perf_counter() - t0
    model.save(str(out_dir / "model"))

    metrics = evaluate(model, cfg, overrides, seed + 1, eval_episodes)
    metrics.update(train_seconds=round(train_s, 1), steps=steps)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return model, metrics, out_dir, geom


def evaluate(model, cfg, overrides, seed, episodes=5):
    """Deterministic eval of a trained policy -> tracking metrics."""
    import numpy as np

    from env_veh import VehicleEnv

    env = VehicleEnv(cfg=cfg, overrides=overrides, seed=seed)
    rets, errs, ys, psis, finals, steps_used = [], [], [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        total, done, x_last, n = 0.0, False, 0.0, 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            total += float(r)
            x_last, n = info["x"], n + 1
            errs.append(info["speed_err"]); ys.append(abs(info["y"]))
            psis.append(abs(info["psi"]))
            done = term or trunc
        rets.append(total); finals.append(x_last); steps_used.append(n)
    return dict(return_mean=float(np.mean(rets)),
                return_std=float(np.std(rets)),
                speed_err_rms=float(np.sqrt(np.mean(np.square(errs)))),
                lateral_rms=float(np.sqrt(np.mean(np.square(ys)))),
                yaw_rms_deg=float(np.degrees(np.sqrt(np.mean(np.square(psis))))),
                progress_mean=float(np.mean(finals)),
                episodes_survived=int(sum(s >= env.n_steps for s in steps_used)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", default=str(Path(__file__).parent / "design.yaml"))
    p.add_argument("--set", action="append", default=[],
                   help="dotted.path=value design override, repeatable")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--model", default=None, help="model.zip for --no-train")
    args = p.parse_args()

    import yaml
    from geometry import export_stl, load_design, render_png

    cfg = load_design(args.design)
    overrides = parse_overrides(args.set)
    tc = cfg["train_control"]
    steps = args.steps or tc["steps"]
    n_envs = args.n_envs or tc["n_envs"]
    seed = args.seed if args.seed is not None else tc["seed"]
    out = args.out or str(ROOT / "results" / "veh3d" /
                          f"control_{time.strftime('%H%M%S')}")

    if args.no_train:
        from stable_baselines3 import PPO

        from env_veh import VehicleEnv
        env = VehicleEnv(cfg=cfg, overrides=overrides, seed=seed)
        model = PPO.load(args.model, device="cpu")
        metrics = evaluate(model, cfg, overrides, seed, args.eval_episodes)
        print(json.dumps(metrics, indent=2))
        return

    model, metrics, out, geom = train_once(cfg, overrides, steps, n_envs,
                                           seed, out)
    print(json.dumps(metrics, indent=2))
    export_stl(geom, Path(out) / "design.stl")
    render_png(geom, Path(out) / "design.png")
    print(f"design artifacts -> {out}")


if __name__ == "__main__":
    main()
