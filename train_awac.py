"""Train an AWAC agent on the offline hydrogen dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from d3rlpy.algos import AWAC, AWACConfig
from d3rlpy.dataset import MDPDataset

from hydrogen_env import HydrogenEnv, VoltageCurrentLookup
from build_offline_dataset import HeuristicPolicy, collect_transitions


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "interpolated_hydrogen_data.csv"
DATASETS_DIR = DATA_DIR / "datasets"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DEFAULT_ENV_KWARGS = dict(
    current_noise=0.015,
    degradation_rate=8e-4,
    degradation_std=3e-4,
    max_degradation=0.3,
    bias_std=0.02,
)


def rollout_episode(awac: AWAC, env: HydrogenEnv) -> dict[str, float]:
    obs = env.reset()
    start_temp = float(env.state["temp"])
    start_voltage = float(env.state["voltage"])
    total_reward = 0.0
    info = {"current_error": 0.0}
    for _ in range(env.max_steps):
        action = awac.predict(obs[np.newaxis, :])[0]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return {
        "episode_reward": float(total_reward),
        "final_current_error": float(info["current_error"]),
        "final_temp": float(env.state["temp"]),
        "final_voltage": float(env.state["voltage"]),
        "start_temp": start_temp,
        "start_voltage": start_voltage,
        "degradation_factor": float(info.get("degradation_factor", 1.0)),
        "boundary_count": int(info.get("boundary_count", 0)),
        "delta_current": float(info.get("delta_current", 0.0)),
        "change_ratio": float(info.get("change_ratio", 0.0)),
    }


def ensure_dataset(dataset_path: Path, episodes: int, seed: int) -> None:
    if dataset_path.exists():
        return
    print(f"Dataset {dataset_path} not found. Generating {episodes} episodes...")
    np.random.seed(seed)
    lookup = VoltageCurrentLookup.from_csv(RAW_DATA_PATH)
    env = HydrogenEnv(lookup, **DEFAULT_ENV_KWARGS)
    policy = HeuristicPolicy(env)
    data = collect_transitions(env, policy, n_episodes=episodes)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **data)
    print(f"Saved dataset to {dataset_path}")


def load_dataset(dataset_path: Path) -> MDPDataset:
    arrays = np.load(dataset_path)
    observations = arrays["observations"].astype(np.float32)
    actions = arrays["actions"].astype(np.float32)
    rewards = arrays["rewards"].astype(np.float32)
    terminals = arrays["terminals"].astype(np.float32)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_space="continuous",
        action_size=actions.shape[-1],
    )
    return dataset


def train_awac(
    dataset: MDPDataset,
    output_dir: Path,
    seed: int,
    n_steps: int,
    batch_size: int,
) -> AWAC:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    config = AWACConfig(batch_size=batch_size, gamma=0.99, n_critics=2)
    awac = AWAC(config=config, device="cpu", enable_ddp=False)
    awac.build_with_dataset(dataset)
    awac.fit(
        dataset=dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
    )
    awac.save_model(str(output_dir / "awac_model.pt"))
    return awac


def evaluate_awac(
    awac: AWAC,
    lookup: VoltageCurrentLookup,
    eval_episodes: int,
    noise: float,
    seed: int,
) -> list[dict[str, float]]:
    np.random.seed(seed)
    env_kwargs = DEFAULT_ENV_KWARGS.copy()
    if noise is not None:
        env_kwargs["current_noise"] = max(noise, 1e-6)
    env = HydrogenEnv(lookup, **env_kwargs)
    results = []
    for _ in range(eval_episodes):
        metrics = rollout_episode(awac, env)
        results.append(metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AWAC offline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASETS_DIR / "offline_dataset.npz",
        help="Path to offline dataset (npz).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=400,
        help="Episodes to simulate if dataset is missing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Training steps for AWAC.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for AWAC updates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "awac_runs",
        help="Directory to store logs and checkpoints.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes to run after training.",
    )
    parser.add_argument(
        "--eval-noise",
        type=float,
        default=0.0,
        help="Gaussian noise std.dev. for evaluation environment (amps).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation rollout.",
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Reduce dataset size and training steps for quick smoke tests.",
    )
    args = parser.parse_args()

    if args.fast_test:
        original_steps = args.steps
        original_episodes = args.episodes
        args.steps = min(args.steps, 5_000)
        args.episodes = min(args.episodes, 120)
        args.eval_episodes = min(args.eval_episodes, 2)
        print(
            "Running in fast-test mode: "
            f"steps {original_steps} -> {args.steps}, "
            f"episodes {original_episodes} -> {args.episodes}, "
            f"eval_episodes -> {args.eval_episodes}"
        )

    dataset_path = args.dataset if args.dataset.is_absolute() else (ROOT_DIR / args.dataset)
    output_dir = args.output if args.output.is_absolute() else (ROOT_DIR / args.output)

    ensure_dataset(dataset_path, args.episodes, args.seed)
    dataset = load_dataset(dataset_path)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    awac = train_awac(dataset, output_dir, args.seed, args.steps, args.batch_size)

    if not args.skip_eval:
        lookup = VoltageCurrentLookup.from_csv(RAW_DATA_PATH)
        results = evaluate_awac(
            awac,
            lookup,
            eval_episodes=args.eval_episodes,
            noise=args.eval_noise,
            seed=args.seed,
        )
        print("\nEvaluation summary:")
        for idx, metrics in enumerate(results, start=1):
            print(
                f"Episode {idx}: reward={metrics['episode_reward']:.3f}, "
                f"final drift={metrics['final_current_error']:.4f} A, last Δ={metrics['delta_current']:.4f} A, "
                f"start (T={metrics['start_temp']:.0f}C, V={metrics['start_voltage']:.3f} V) -> "
                f"end (T={metrics['final_temp']:.0f}C, V={metrics['final_voltage']:.3f} V), "
                f"degradation={metrics.get('degradation_factor', 1.0):.3f}, boundary_count={metrics.get('boundary_count', 0)}"
            )


if __name__ == "__main__":
    main()
