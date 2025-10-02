"""Generate an offline dataset for AWAC training.

The script rolls out a heuristic controller inside ``HydrogenEnv`` to produce
trajectories that cover the action space and produce positive as well as
negative rewards. The resulting arrays are saved into ``offline_dataset.npz``
which can be consumed by d3rlpy's ``MDPDataset`` API.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from hydrogen_env import HydrogenEnv, VoltageCurrentLookup


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "interpolated_hydrogen_data.csv"
DATASETS_DIR = DATA_DIR / "datasets"


class HeuristicPolicy:
    """Simple behavior policy used to populate the offline replay buffer."""

    def __init__(self, env: HydrogenEnv, random_prob: float = 0.1) -> None:
        self.env = env
        self.random_prob = random_prob

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        temp, voltage, current, resistance, _error, delta = obs
        tolerance = self.env.tolerance
        band = tolerance * 0.5

        if np.random.rand() < self.random_prob:
            return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)

        action = np.zeros(2, dtype=np.float32)

        if abs(delta) <= band:
            # When changes are within tolerance, hold with mild dithering.
            action += np.random.normal(0.0, 0.05, size=2)
            action *= 0.3
            return self._respect_bounds(temp, voltage, current, action)

        gain = np.clip(delta / (tolerance * 4.0), -1.0, 1.0)
        action[1] = float(np.clip(-gain, -1.0, 1.0))

        temp_correction = -0.4 * gain
        if abs(delta) > tolerance:
            temp_correction += -0.2 * np.sign(delta)
        action[0] = float(np.clip(temp_correction, -1.0, 1.0))

        # Add light exploration noise.
        action += np.random.normal(0.0, 0.05, size=2)
        return self._respect_bounds(temp, voltage, current, action).astype(np.float32)

    def _respect_bounds(
        self, temp: float, voltage: float, current: float, action: np.ndarray
    ) -> np.ndarray:
        temp = float(temp)
        adjusted = action.copy()

        if temp <= self.env.temp_min + self.env.temp_step:
            adjusted[0] = max(adjusted[0], 1.0)
        if temp >= self.env.temp_max - 2 * self.env.temp_step:
            adjusted[0] = min(adjusted[0], -1.0)

        temp_int = int(np.clip(round(temp / self.env.temp_step) * self.env.temp_step, self.env.temp_min, self.env.temp_max))
        v_min, v_max = self.env.lookup.voltage_bounds(temp_int)
        current_voltage = float(voltage)
        projected_voltage = current_voltage + adjusted[1] * self.env.voltage_step
        if projected_voltage <= v_min + 0.05:
            adjusted[1] = max(adjusted[1], 1.0)
        if projected_voltage >= v_max - 0.05:
            adjusted[1] = min(adjusted[1], -1.0)

        # If already near target current, damp the action
        if abs(current - self.env.target_current) < self.env.tolerance:
            adjusted *= 0.4

        adjusted[0] = float(np.clip(adjusted[0], -1.0, 1.0))
        adjusted[1] = float(np.clip(adjusted[1], -1.0, 1.0))
        return adjusted


def collect_transitions(
    env: HydrogenEnv,
    policy: Callable[[np.ndarray], int],
    n_episodes: int,
) -> dict[str, np.ndarray]:
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(env.max_steps):
            raw_action = policy(obs)
            if isinstance(raw_action, (int, np.integer)):
                action_vec = HydrogenEnv.discrete_action_vector(int(raw_action))
            else:
                action_vec = np.asarray(raw_action, dtype=np.float32)
            next_obs, reward, done, _ = env.step(raw_action)

            observations.append(obs)
            actions.append(action_vec)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(done)

            obs = next_obs
            if done:
                break

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.bool_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline dataset")
    parser.add_argument(
        "--episodes",
        type=int,
        default=400,
        help="Number of episodes to roll out (default: 400).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASETS_DIR / "offline_dataset.npz",
        help="Output .npz path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    lookup = VoltageCurrentLookup.from_csv(RAW_DATA_PATH)
    env = HydrogenEnv(
        lookup,
        current_noise=0.015,
        degradation_rate=8e-4,
        degradation_std=3e-4,
        max_degradation=0.3,
        bias_std=0.02,
    )
    policy = HeuristicPolicy(env)

    output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
    data = collect_transitions(env, policy, n_episodes=args.episodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    total_steps = len(data["observations"])
    print(f"Saved {total_steps} transitions to {output_path}")


if __name__ == "__main__":
    main()
