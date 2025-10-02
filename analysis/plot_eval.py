"""Generate diagnostic plots for a trained AWAC policy."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydrogen_env import HydrogenEnv, VoltageCurrentLookup
from train_awac import AWAC, AWACConfig, load_dataset


def collect_episode(agent: AWAC, env: HydrogenEnv) -> dict[str, list[float]]:
    obs = env.reset()
    traj: dict[str, list[float] | float] = {
        "temp": [],
        "voltage": [],
        "current": [],
        "target": [],
        "delta": [],
        "reward": [],
        "boundary": [],
    }
    total_reward = 0.0
    prev_current = env.state["current"]
    for _ in range(env.max_steps):
        action = agent.predict(obs[np.newaxis, :])[0]
        obs, reward, done, info = env.step(action)
        total_reward += reward

        traj["temp"].append(env.state["temp"])
        traj["voltage"].append(env.state["voltage"])
        traj["current"].append(env.state["current"])
        traj["target"].append(env.moving_avg_current)
        traj["delta"].append(info.get("delta_current", env.state["current"] - prev_current))
        traj["reward"].append(reward)
        traj["boundary"].append(info.get("boundary_count", 0))

        prev_current = env.state["current"]
        if done:
            break
    traj["total_reward"] = total_reward
    traj["steps"] = len(traj["current"])
    return traj


def prepare_agent(model_dir: Path, dataset_path: Path) -> tuple[AWAC, HydrogenEnv]:
    dataset = load_dataset(dataset_path)
    config = AWACConfig(batch_size=256, gamma=0.99, n_critics=2)
    agent = AWAC(config=config, device="cpu", enable_ddp=False)
    agent.build_with_dataset(dataset)
    agent.load_model(str(model_dir / "awac_model.pt"))

    lookup = VoltageCurrentLookup.from_csv("data/interpolated_hydrogen_data.csv")
    env = HydrogenEnv(lookup)
    return agent, env


def plot_episode(traj: dict[str, list[float]], out_dir: Path, idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = range(traj["steps"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, traj["current"], label="Current")
    axes[0].plot(steps, traj["target"], label="Moving Target", linestyle="--")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, traj["temp"], label="Temperature (°C)")
    axes[1].plot(steps, traj["voltage"], label="Voltage (V)")
    axes[1].set_ylabel("Temp / Volt")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, traj["delta"], label="Δ Current")
    axes[2].bar(steps, traj["boundary"], alpha=0.3, label="Boundary Count")
    axes[2].set_ylabel("ΔI & Boundary")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Episode {idx} (Reward {traj['total_reward']:.2f})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"episode_{idx}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation episode")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/awac_quick"))
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/offline_dataset_small.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    agent, env = prepare_agent(args.model_dir, args.dataset)
    traj = collect_episode(agent, env)
    plot_episode(traj, args.output, idx=1)
    print(f"Saved plot to {args.output / 'episode_1.png'}")


if __name__ == "__main__":
    main()
