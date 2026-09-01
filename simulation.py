from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SimulatedSignals:
    availability: np.ndarray
    behavioral_compatibility: np.ndarray
    collaboration_compatibility: np.ndarray
    experience_relevance: np.ndarray
    learning_velocity: np.ndarray
    project_complexity: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.column_stack(
            [
                self.availability,
                self.behavioral_compatibility,
                self.collaboration_compatibility,
                self.experience_relevance,
                self.learning_velocity,
                self.project_complexity,
            ]
        ).astype(np.float32)

    @staticmethod
    def names():
        return [
            "availability",
            "behavioral_compatibility",
            "collaboration_compatibility",
            "experience_relevance",
            "learning_velocity",
            "project_complexity",
        ]


def generate_simulated_signals(
    n: int,
    rng: np.random.Generator,
    cfg: Dict,
    skill_similarity: Optional[np.ndarray] = None,
) -> SimulatedSignals:
    """
    Generate variables absent from the static source dataset.

    These values are explicitly semi-synthetic and must not be described as
    observed workforce measurements.
    """
    lo_a, hi_a = float(cfg["availability_min"]), float(cfg["availability_max"])
    lo_b, hi_b = float(cfg["behavior_min"]), float(cfg["behavior_max"])
    lo_c, hi_c = float(cfg["collaboration_min"]), float(cfg["collaboration_max"])
    lo_e, hi_e = float(cfg["experience_min"]), float(cfg["experience_max"])
    lo_l, hi_l = float(cfg["learning_rate_min"]), float(cfg["learning_rate_max"])

    availability = rng.uniform(lo_a, hi_a, size=n)
    behavior = rng.uniform(lo_b, hi_b, size=n)
    collaboration = rng.uniform(lo_c, hi_c, size=n)
    experience = rng.uniform(lo_e, hi_e, size=n)
    learning_velocity = rng.uniform(lo_l, hi_l, size=n)
    project_complexity = np.clip(rng.beta(2.0, 2.0, size=n), 0.0, 1.0)

    # Skill similarity may weakly condition experience relevance, but never labels.
    if skill_similarity is not None:
        experience = np.clip(
            0.65 * experience + 0.35 * np.asarray(skill_similarity), 0.0, 1.0
        )

    return SimulatedSignals(
        availability=availability.astype(np.float32),
        behavioral_compatibility=behavior.astype(np.float32),
        collaboration_compatibility=collaboration.astype(np.float32),
        experience_relevance=experience.astype(np.float32),
        learning_velocity=learning_velocity.astype(np.float32),
        project_complexity=project_complexity.astype(np.float32),
    )


class DynamicPairEnvironment:
    """
    Controlled semi-synthetic environment for DDQN assignment learning.

    State:
        fixed learned pair embedding + six simulated operational signals.

    Action:
        0 = reject/do not assign
        1 = assign

    Reward:
        primarily rewards agreement with the declared compatibility label while
        adding a bounded operational term only for assignment actions.

    The environment is a feasibility simulation, not a model of observed
    longitudinal workforce transitions.
    """

    def __init__(
        self,
        pair_embeddings: np.ndarray,
        labels: np.ndarray,
        base_signals: np.ndarray,
        rng: np.random.Generator,
        cfg: Dict,
    ):
        self.emb = np.asarray(pair_embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.base = np.asarray(base_signals, dtype=np.float32)
        self.rng = rng
        self.cfg = cfg
        self.order = np.arange(len(self.labels))
        self.ptr = 0
        self.current_idx = 0
        self.current_state = None

    @property
    def state_dim(self) -> int:
        return int(self.emb.shape[1] + self.base.shape[1])

    def _drift(self, signals: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(
            0.0, float(self.cfg.get("project_drift_std", 0.05)), size=signals.shape
        )
        drifted = np.asarray(signals, dtype=np.float32) + noise.astype(np.float32)
        # learning velocity is naturally a smaller-range signal, but clipping all
        # simulation channels to [0,1] keeps the RL state bounded.
        return np.clip(drifted, 0.0, 1.0).astype(np.float32)

    def _state_for(self, idx: int) -> np.ndarray:
        signals = self._drift(self.base[idx])
        return np.concatenate([self.emb[idx], signals], axis=0).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.rng.shuffle(self.order)
        self.ptr = 0
        self.current_idx = int(self.order[self.ptr])
        self.current_state = self._state_for(self.current_idx)
        return self.current_state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        idx = self.current_idx
        y = int(self.labels[idx])
        action = int(action)

        classification_reward = 1.0 if action == y else -1.0

        availability, behavior, collab, experience, learning, complexity = self.base[idx]
        operational_quality = (
            0.25 * availability
            + 0.20 * behavior
            + 0.20 * collab
            + 0.20 * experience
            + 0.15 * learning
            - 0.15 * complexity
        )
        # Keep the operational term secondary so the agent is not rewarded for
        # simulation variables at the expense of the declared evaluation label.
        operational_term = 0.25 * float(operational_quality) if action == 1 else 0.0
        reward = float(classification_reward + operational_term)

        self.ptr += 1
        done = self.ptr >= min(
            len(self.order), int(self.cfg.get("max_episode_steps", len(self.order)))
        )
        if done:
            next_state = np.zeros_like(self.current_state)
        else:
            self.current_idx = int(self.order[self.ptr])
            next_state = self._state_for(self.current_idx)
            self.current_state = next_state

        info = {
            "index": idx,
            "heuristic_label": y,
            "classification_reward": classification_reward,
            "simulated_operational_term": operational_term,
        }
        return next_state.copy(), reward, done, info
