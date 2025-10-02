"""Utilities for simulating a simplified hydrogen production control task.

The code loads the interpolated physics data provided in
``interpolated_hydrogen_data.csv`` and exposes a light-weight environment that
matches the scenario described in the project slides. It is purposely simple so
that it can be used to generate offline trajectories for AWAC training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class VoltageCurrentLookup:
    """Wraps lookup tables extracted from the interpolated CSV."""

    temp_to_curves: Dict[int, Tuple[np.ndarray, np.ndarray]]

    @classmethod
    def from_csv(cls, path: Path | str) -> "VoltageCurrentLookup":
        df = pd.read_csv(path)
        temp_to_curves: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for temp, group in df.groupby("Temp"):
            ordered = group.sort_values("Current")
            currents = ordered["Current"].to_numpy(dtype=np.float32)
            voltages = ordered["Voltage"].to_numpy(dtype=np.float32)
            temp_to_curves[int(temp)] = (currents, voltages)
        return cls(temp_to_curves=temp_to_curves)

    def current_from_voltage(self, temp: int, voltage: float) -> float:
        currents, voltages = self.temp_to_curves[temp]
        clipped_voltage = np.clip(voltage, voltages[0], voltages[-1])
        return float(np.interp(clipped_voltage, voltages, currents))

    def voltage_from_current(self, temp: int, current: float) -> float:
        currents, voltages = self.temp_to_curves[temp]
        clipped_current = np.clip(current, currents[0], currents[-1])
        return float(np.interp(clipped_current, currents, voltages))

    def voltage_bounds(self, temp: int) -> Tuple[float, float]:
        _, voltages = self.temp_to_curves[temp]
        return float(voltages[0]), float(voltages[-1])

    @property
    def temperatures(self) -> np.ndarray:
        temps = np.array(sorted(self.temp_to_curves.keys()), dtype=np.int32)
        return temps


class HydrogenEnv:
    """Simplified environment for hydrogen production control.

    The observation vector contains:
    ``[temp, voltage, current, resistance, current_error, delta_current]``
    where ``current_error`` is relative to the target production rate and
    ``delta_current`` is the change from the previous step.
    """

    ACTIONS = {
        0: "hold",
        1: "temp_up",
        2: "temp_down",
        3: "volt_up",
        4: "volt_down",
    }

    _DISCRETE_TO_VECTOR = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([-1.0, 0.0], dtype=np.float32),
        3: np.array([0.0, 1.0], dtype=np.float32),
        4: np.array([0.0, -1.0], dtype=np.float32),
    }

    @classmethod
    def discrete_action_vector(cls, action_id: int) -> np.ndarray:
        return cls._DISCRETE_TO_VECTOR[int(action_id)].copy()

    def __init__(
        self,
        lookup: VoltageCurrentLookup,
        target_current: float = 0.93,
        voltage_step: float = 0.05,
        temp_step: int = 5,
        tolerance: float = 0.01,
        max_steps: int = 96,
        current_noise: float = 0.01,
        degradation_rate: float = 5e-4,
        degradation_std: float = 2e-4,
        max_degradation: float = 0.25,
        bias_std: float = 0.01,
    ) -> None:
        self.lookup = lookup
        self.default_target_current = target_current
        self.target_current = target_current
        self.voltage_step = voltage_step
        self.temp_step = temp_step
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.current_noise = current_noise
        self.degradation_rate = degradation_rate
        self.degradation_std = degradation_std
        self.max_degradation = max_degradation
        self.bias_std = bias_std

        self.temp_min = int(lookup.temperatures.min())
        self.temp_max = int(lookup.temperatures.max())
        self.reset()

    def _calc_current(self, temp: int, voltage: float) -> float:
        current = self.lookup.current_from_voltage(temp, voltage)
        current *= self.degradation_factor
        if self.current_noise:
            current += np.random.normal(0.0, self.current_noise)
        if self.bias_std:
            current += self.measurement_bias
        return float(np.clip(current, 0.3, 3.2))

    def _calc_voltage_bounds(self, temp: int) -> Tuple[float, float]:
        return self.lookup.voltage_bounds(temp)

    def reset(self) -> np.ndarray:
        temps = self.lookup.temperatures
        temp = int(np.random.choice(temps))
        self.degradation_factor = float(
            np.clip(1.0 - np.random.uniform(0.0, min(self.max_degradation, 0.05)), 0.7, 1.05)
        )
        self.measurement_bias = float(np.random.normal(0.0, self.bias_std))

        base_voltage = self.lookup.voltage_from_current(temp, self.default_target_current)
        voltage_bounds = self._calc_voltage_bounds(temp)
        jitter = np.random.uniform(-0.05, 0.05)
        voltage = np.clip(base_voltage + jitter, *voltage_bounds)
        current = self._calc_current(temp, voltage)

        self.boundary_count = 0
        self.prev_abs_change = 0.0
        self.prev_delta = 0.0
        self.moving_avg_current = current
        self.prev_target_error = 0.0
        self.target_current = current

        self.state = {
            "temp": temp,
            "voltage": voltage,
            "current": current,
            "prev_current": current,
            "steps": 0,
        }
        return self._get_obs(0.0)

    def _get_obs(self, delta_current: float) -> np.ndarray:
        temp = self.state["temp"]
        voltage = self.state["voltage"]
        current = self.state["current"]
        resistance = voltage / max(current, 1e-6)
        current_error = current - self.target_current
        obs = np.array(
            [
                float(temp),
                float(voltage),
                float(current),
                float(resistance),
                float(current_error),
                float(delta_current),
            ],
            dtype=np.float32,
        )
        return obs

    def _vector_from_action(
        self, action: Union[int, np.ndarray, Sequence[float]]
    ) -> np.ndarray:
        if isinstance(action, (int, np.integer)):
            assert action in self._DISCRETE_TO_VECTOR, f"Invalid action {action}"
            return self._DISCRETE_TO_VECTOR[int(action)]
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (2,):
            raise ValueError(
                f"Continuous action must have shape (2,), got {action_arr.shape}"
            )
        return np.clip(action_arr, -1.0, 1.0)

    def step(
        self, action: Union[int, np.ndarray, Sequence[float]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        action_vec = self._vector_from_action(action)
        temp = self.state["temp"]
        voltage = self.state["voltage"]

        temp += int(np.round(action_vec[0])) * self.temp_step
        temp = int(np.clip(temp, self.temp_min, self.temp_max))
        voltage += float(action_vec[1]) * self.voltage_step

        v_min, v_max = self._calc_voltage_bounds(temp)
        voltage = float(np.clip(voltage, v_min, v_max))
        current = self._calc_current(temp, voltage)

        prev_current = self.state["current"]
        delta_current = current - prev_current
        current_error = current - self.target_current

        abs_change = abs(delta_current)
        change_ratio = abs_change / self.tolerance
        reward = 1.0 - change_ratio**2

        action_penalty = 0.1 * (abs(action_vec[0]) + abs(action_vec[1]))
        reward -= action_penalty

        bound_penalty = 0.0
        if temp in (self.temp_min, self.temp_max):
            bound_penalty += 1.0
        if voltage <= v_min + 1e-6 or voltage >= v_max - 1e-6:
            bound_penalty += 1.0
        reward -= bound_penalty

        temp_mid = (self.temp_max + self.temp_min) / 2.0
        temp_span = (self.temp_max - self.temp_min) / 2.0
        reward -= 0.3 * (abs(temp - temp_mid) / max(temp_span, 1e-6))
        volt_mid = (v_max + v_min) / 2.0
        volt_span = (v_max - v_min) / 2.0
        reward -= 0.3 * (abs(voltage - volt_mid) / max(volt_span, 1e-6))

        near_boundary = (
            temp in (self.temp_min, self.temp_max)
            or voltage <= v_min + 1e-6
            or voltage >= v_max - 1e-6
        )
        if near_boundary:
            self.boundary_count += 1
            reward -= min(1.5 * self.boundary_count, 10.0)
        else:
            if self.boundary_count > 0:
                retreat_bonus = float(np.clip(self.boundary_count * 0.6, 0.0, 2.0))
                reward += retreat_bonus
            self.boundary_count = 0

        # Update moving average target to follow slow drift.
        alpha = 0.05  # smoothing factor for moving average
        self.moving_avg_current = (1 - alpha) * self.moving_avg_current + alpha * current
        adaptive_target = self.moving_avg_current
        target_error = current - adaptive_target
        target_ratio = abs(target_error) / self.tolerance

        improvement = self.prev_abs_change - abs_change
        if improvement > 0:
            reward += float(np.clip(improvement / (self.tolerance * 2.0), 0.0, 0.5))

        # Bonus for reducing error relative to moving average.
        if abs(target_error) < abs(self.prev_target_error):
            reward += 0.3

        # Bonus when the agent reverses the drift direction meaningfully.
        if np.sign(delta_current) != np.sign(self.prev_delta) and abs_change > 0:
            reward += 0.4

        reward -= min(target_ratio, 5.0) * 0.5

        if change_ratio > 5.0:
            reward -= min((change_ratio - 5.0) * 0.5, 4.0)

        reward = float(np.clip(reward, -10.0, 1.0))

        self.state.update(
            {
                "temp": temp,
                "voltage": voltage,
                "current": current,
                "prev_current": prev_current,
                "steps": self.state["steps"] + 1,
            }
        )

        self._update_latent_drift()
        self.prev_abs_change = abs_change
        self.prev_delta = delta_current
        self.prev_target_error = target_error

        forced_done = self.boundary_count >= 3
        if forced_done:
            reward -= 12.0

        done = self.state["steps"] >= self.max_steps or forced_done
        obs = self._get_obs(delta_current)
        info = {
            "current_error": current_error,
            "delta_current": delta_current,
            "change_ratio": change_ratio,
            "action_vector": action_vec,
            "degradation_factor": self.degradation_factor,
            "boundary_count": self.boundary_count,
            "forced_done": forced_done,
        }
        return obs, reward, done, info

    def _update_latent_drift(self) -> None:
        delta = -self.degradation_rate + np.random.normal(0.0, self.degradation_std)
        self.degradation_factor = float(
            np.clip(
                self.degradation_factor + delta,
                1.0 - self.max_degradation,
                1.05,
            )
        )
        bias_delta = np.random.normal(0.0, self.bias_std * 0.05)
        self.measurement_bias = float(
            np.clip(self.measurement_bias + bias_delta, -self.bias_std, self.bias_std)
        )


def rollout(
    env: HydrogenEnv,
    policy,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect a single trajectory using the provided policy callable.

    The ``policy`` must accept the current observation and return an action id.
    """

    obs_list = []
    act_list = []
    rew_list = []
    next_obs_list = []
    done_list = []

    obs = env.reset()
    for _ in range(n_steps):
        action = int(policy(obs))
        next_obs, reward, done, _ = env.step(action)

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        next_obs_list.append(next_obs)
        done_list.append(done)

        obs = next_obs
        if done:
            break

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        np.array(rew_list, dtype=np.float32),
        np.array(done_list, dtype=np.bool_),
        np.array(next_obs_list, dtype=np.float32),
    )
