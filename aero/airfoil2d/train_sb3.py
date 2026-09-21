"""Train SAC/TD3 on AirfoilDesignEnv — CPU only.

Usage (source aero/env.sh first so CUDA stays invisible):
    python /home/heinz/isaaclab_uav/aero/airfoil2d/train_sb3.py --steps 2000 --n-envs 4
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "td3"], default="sac")
    p.add_argument("--steps", type=int, default=2000, help="total timesteps")
    p.add_argument("--n-envs", type=int, default=4,
                   help="SubprocVecEnv workers (4 for smoke; keep < cores/2 while "
                        "the live training job is running)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "sb3_smoke"))
    args = p.parse_args()

    import torch
    torch.set_num_threads(1)  # N env procs × N threads would thrash the box
    assert not torch.cuda.is_available(), (
        "GPU is visible — source aero/env.sh (CUDA_VISIBLE_DEVICES='') before training")

    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import airfoil2d.env_xfoil as env_mod  # noqa: F401  (import for worker pickling)

    def make_env():
        return env_mod.AirfoilDesignEnv(seed=args.seed)

    if args.n_envs > 1:
        env = SubprocVecEnv([make_env] * args.n_envs, start_method="fork")
    else:
        env = make_env()

    cls = {"sac": SAC, "td3": TD3}[args.algo]
    model = cls("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                buffer_size=50_000 if args.algo == "sac" else 200_000)
    model.learn(total_timesteps=args.steps, progress_bar=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.algo}_airfoil_{args.steps}"
    model.save(str(path))
    print(f"saved {path}.zip")

    # one deterministic episode to show the learned refinement
    import numpy as np
    obs, _ = env.reset()
    total = 0.0
    for _ in range(8):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total += float(np.sum(reward)) if hasattr(reward, "__len__") else float(reward)
        if np.any(done) if hasattr(done, "__len__") else done:
            break
    print(f"deterministic episode return: {total:.3f}")


if __name__ == "__main__":
    main()
