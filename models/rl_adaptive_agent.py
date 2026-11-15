"""
models/rl_adaptive_agent.py
===========================
Online Reinforcement Learning (RL) Adaptive Policy Agent for Sign Language Personalization.

Architecture:
- Paradigm: Contextual Bandit / Policy Gradient (REINFORCE with Baseline Subtraction)
- State Space: Concatenation of normalized 3D hand spatial landmarks (63-dim) + baseline model logits (39-dim)
- Action Space: Discrete selection / probability shift across all 39 sign language classes
- Policy: Parameterized Linear-Softmax Policy pi_theta(a | s) with L2 regularization
- Reward Structure:
    * +1.0 : Positive verification (user continues signing, word accepted by dictionary, auto-commit confirmed)
    * -1.0 : Immediate correction (user clicks Backspace / Delete or signs 'del' immediately after)
    * +0.2 : High-confidence steady hold without correction
- Persistence: Serializes policy weights theta and cumulative statistics to models/rl_policy.npz
"""
import os
import json
import time
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import numpy as np


class AdaptiveSignRLAgent:
    """
    Contextual Policy-Gradient Reinforcement Learning Agent.
    Continuously adapts the gesture decision boundaries to the unique anatomical
    proportions, bone lengths, and signing style of the individual user.
    """

    def __init__(
        self,
        num_classes: int = 39,
        classes: Optional[List[str]] = None,
        state_dim: int = 102,  # 63 spatial features + 39 base probabilities
        learning_rate: float = 0.015,
        l2_reg: float = 0.001,
        policy_path: str = "models/rl_policy.npz",
        enabled: bool = True
    ):
        self.num_classes = num_classes
        self.classes = classes or [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
        ]
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.policy_path = policy_path
        self.enabled = enabled

        # Policy weight matrix: shape (num_classes, state_dim)
        # Initialized with zeros (identity / zero-bias relative to base supervised model)
        self.weights = np.zeros((self.num_classes, self.state_dim), dtype=np.float32)
        self.bias = np.zeros(self.num_classes, dtype=np.float32)

        # Experience buffer for rolling policy gradient updates
        self.last_state: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None
        self.last_probs: Optional[np.ndarray] = None
        self.history: List[Dict[str, Any]] = []

        # Running Telemetry Statistics
        self.total_episodes: int = 0
        self.cumulative_reward: float = 0.0
        self.positive_rewards: int = 0
        self.negative_rewards: int = 0
        self.last_feedback_message: str = "RL Agent Initialized"
        self.last_update_timestamp: float = time.time()

        # Load existing policy if present
        self.load_policy()

    def build_state_vector(
        self,
        raw_coords: Optional[List[Tuple[float, float, float]]],
        base_probs: np.ndarray
    ) -> np.ndarray:
        """
        Constructs the normalized state vector s_t combining spatial kinematics
        and the supervised model's probability distribution.
        """
        if raw_coords is None or len(raw_coords) == 0:
            spatial = np.zeros(63, dtype=np.float32)
        else:
            flat = []
            w = raw_coords[0]
            palm = float(np.linalg.norm(np.array(raw_coords[9]) - np.array(w))) + 1e-6
            for pt in raw_coords[:21]:
                # Relative translation to wrist, scaled by palm length for scale invariance
                flat.extend([
                    (pt[0] - w[0]) / palm,
                    (pt[1] - w[1]) / palm,
                    (pt[2] - w[2]) / palm if len(pt) > 2 else 0.0
                ])
            spatial = np.array(flat[:63], dtype=np.float32)

        if len(base_probs) != self.num_classes:
            p_pad = np.zeros(self.num_classes, dtype=np.float32)
            p_pad[:len(base_probs)] = base_probs[:self.num_classes]
            base_probs = p_pad

        state = np.concatenate([spatial, base_probs.astype(np.float32)])
        if len(state) != self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:min(len(state), self.state_dim)] = state[:self.state_dim]
            return padded
        return state

    def get_policy_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the policy softmax distribution pi_theta(a | s).
        """
        logits = np.dot(self.weights, state) + self.bias
        # Numerical stability clamp
        logits = np.clip(logits, -20.0, 20.0)
        exp_logits = np.exp(logits - np.max(logits))
        policy_probs = exp_logits / (np.sum(exp_logits) + 1e-9)
        return policy_probs

    def adapt_probabilities(
        self,
        raw_coords: Optional[List[Tuple[float, float, float]]],
        base_probs: np.ndarray,
        temperature: float = 0.5
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Blends the supervised model's probabilities with the RL policy distribution:
        P_adapted(y) proportional to P_base(y) * (pi_theta(y|s))^temperature
        """
        if not self.enabled:
            return base_probs, None

        state = self.build_state_vector(raw_coords, base_probs)
        policy_probs = self.get_policy_distribution(state)

        # Cache step for subsequent reward assignment
        self.last_state = state
        self.last_probs = policy_probs

        # Blend supervised probability with learned RL contextual modulation
        # Using logarithmic Bayesian weighting
        log_base = np.log(np.clip(base_probs, 1e-6, 1.0))
        log_rl = np.log(np.clip(policy_probs, 1e-6, 1.0))

        # Weight ratio: 75% supervised backbone + 25% personal RL adaptation
        adapted_logits = 0.75 * log_base + 0.25 * (log_rl * temperature)
        exp_adapted = np.exp(adapted_logits - np.max(adapted_logits))
        adapted_probs = exp_adapted / np.sum(exp_adapted)

        # Determine if RL altered the top prediction
        top_base = int(np.argmax(base_probs))
        top_adapted = int(np.argmax(adapted_probs))
        self.last_action_idx = top_adapted

        note = None
        if top_base != top_adapted and top_adapted < len(self.classes):
            note = f"RL Personalization shifted '{self.classes[top_base]}' -> '{self.classes[top_adapted]}'"

        return adapted_probs, note

    def record_feedback(
        self,
        label: str,
        reward: float,
        reason: str = "explicit"
    ) -> Dict[str, Any]:
        """
        Applies a Policy Gradient (REINFORCE) update with the observed reward.
        """
        if not self.enabled or self.last_state is None:
            return {"status": "skipped", "reason": "RL disabled or no active state"}

        if label not in self.classes:
            return {"status": "error", "reason": f"Unknown label '{label}'"}

        action_idx = self.classes.index(label)
        state = self.last_state
        policy_probs = self.get_policy_distribution(state)

        # Compute Policy Gradient:
        # grad_theta = (1(a == y) - pi(a|s)) * state * reward
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)

        for a in range(self.num_classes):
            delta = (1.0 if a == action_idx else 0.0) - policy_probs[a]
            grad_weights[a] = delta * state * reward
            grad_bias[a] = delta * reward

        # Update policy weights with L2 regularization
        self.weights += self.learning_rate * grad_weights - (self.learning_rate * self.l2_reg * self.weights)
        self.bias += self.learning_rate * grad_bias

        # Update telemetry
        self.total_episodes += 1
        self.cumulative_reward += reward
        if reward > 0:
            self.positive_rewards += 1
        else:
            self.negative_rewards += 1

        self.last_update_timestamp = time.time()
        self.last_feedback_message = f"Sign '{label}': {('+' if reward > 0 else '')}{reward:.1f} ({reason})"

        # Persist policy checkpoint
        self.save_policy()

        return {
            "status": "updated",
            "sign": label,
            "reward": reward,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "episodes": self.total_episodes,
            "reason": reason
        }

    def save_policy(self, path: Optional[str] = None):
        """Saves RL policy matrix and metadata to disk."""
        target_path = path or self.policy_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        try:
            meta = {
                "total_episodes": self.total_episodes,
                "cumulative_reward": float(self.cumulative_reward),
                "positive_rewards": self.positive_rewards,
                "negative_rewards": self.negative_rewards,
                "last_update": self.last_update_timestamp
            }
            np.savez_compressed(
                target_path,
                weights=self.weights,
                bias=self.bias,
                meta=json.dumps(meta)
            )
        except Exception as e:
            print(f"[!] Warning: Failed to save RL policy: {e}")

    def load_policy(self, path: Optional[str] = None) -> bool:
        """Loads RL policy matrix from disk if present."""
        target_path = path or self.policy_path
        if not os.path.exists(target_path):
            return False

        try:
            data = np.load(target_path, allow_pickle=True)
            if "weights" in data and "bias" in data:
                w = data["weights"]
                b = data["bias"]
                if w.shape == self.weights.shape and b.shape == self.bias.shape:
                    self.weights = w
                    self.bias = b
                    if "meta" in data:
                        meta = json.loads(str(data["meta"]))
                        self.total_episodes = meta.get("total_episodes", 0)
                        self.cumulative_reward = meta.get("cumulative_reward", 0.0)
                        self.positive_rewards = meta.get("positive_rewards", 0)
                        self.negative_rewards = meta.get("negative_rewards", 0)
                    return True
        except Exception as e:
            print(f"[!] Warning: Failed to load RL policy: {e}")
        return False

    def reset_policy(self):
        """Resets the RL policy back to neutral factory zero weights."""
        self.weights.fill(0.0)
        self.bias.fill(0.0)
        self.total_episodes = 0
        self.cumulative_reward = 0.0
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.last_feedback_message = "Policy reset to factory baseline."
        if os.path.exists(self.policy_path):
            try:
                os.remove(self.policy_path)
            except Exception:
                pass

    def get_metrics(self) -> Dict[str, Any]:
        """Returns live telemetry metrics for Streamlit UI display."""
        avg_reward = (
            round(self.cumulative_reward / max(1, self.total_episodes), 3)
            if self.total_episodes > 0 else 0.0
        )
        return {
            "enabled": self.enabled,
            "episodes": self.total_episodes,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "average_reward": avg_reward,
            "positive_count": self.positive_rewards,
            "negative_count": self.negative_rewards,
            "weight_norm": round(float(np.linalg.norm(self.weights)), 4),
            "status": self.last_feedback_message
        }
