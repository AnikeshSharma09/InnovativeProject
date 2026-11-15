"""
tests/test_rl_adaptive_agent.py
===============================
Unit test suite verifying the Reinforcement Learning Adaptive Policy Agent.
"""
import unittest
import numpy as np
import os
import json
from models.rl_adaptive_agent import AdaptiveSignRLAgent


class TestAdaptiveSignRLAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("data/canonical_signs_3d.json", "r") as f:
            cls.canon = json.load(f)

    def setUp(self):
        self.temp_policy = "models/test_rl_policy.npz"
        self.agent = AdaptiveSignRLAgent(
            policy_path=self.temp_policy,
            learning_rate=0.05,
            enabled=True
        )

    def tearDown(self):
        if os.path.exists(self.temp_policy):
            try:
                os.remove(self.temp_policy)
            except Exception:
                pass

    def _dummy_probs(self, top_cls: str = "A", top_p: float = 0.50):
        probs = np.full(len(self.agent.classes), 0.01)
        if top_cls in self.agent.classes:
            probs[self.agent.classes.index(top_cls)] = top_p
        return probs / np.sum(probs)

    def test_state_vector_shape(self):
        """Verifies state vector concatenation: 63 spatial + 39 base probs = 102."""
        coords = self.canon["K"]
        probs = self._dummy_probs("K", 0.70)
        state = self.agent.build_state_vector(coords, probs)
        self.assertEqual(len(state), 102)
        self.assertEqual(state.dtype, np.float32)

    def test_neutral_policy_initialization(self):
        """Zero initial weights must produce uniform policy distribution."""
        coords = self.canon["V"]
        probs = self._dummy_probs("V", 0.60)
        adapted, note = self.agent.adapt_probabilities(coords, probs)
        # Top class before and after should match under neutral policy
        self.assertEqual(np.argmax(probs), np.argmax(adapted))
        self.assertIsNone(note)

    def test_positive_reward_reinforces_class(self):
        """Positive reward (+1.0) must increase class logit and posterior probability."""
        coords = self.canon["K"]
        probs = self._dummy_probs("K", 0.40)
        k_idx = self.agent.classes.index("K")

        # Initial forward pass to cache state
        p_init, _ = self.agent.adapt_probabilities(coords, probs)

        # Apply 3 positive reinforcement updates
        for _ in range(3):
            self.agent.record_feedback("K", reward=1.0, reason="user_confirmed")

        p_after, _ = self.agent.adapt_probabilities(coords, probs)
        self.assertGreater(p_after[k_idx], p_init[k_idx])
        self.assertEqual(self.agent.positive_rewards, 3)
        self.assertGreater(self.agent.cumulative_reward, 0.0)

    def test_negative_reward_penalizes_class(self):
        """Negative reward (-1.0) on wrong class must decrease its posterior probability."""
        coords = self.canon["V"]
        # Model wrongly output K instead of V
        probs = self._dummy_probs("K", 0.55)
        k_idx = self.agent.classes.index("K")

        p_init, _ = self.agent.adapt_probabilities(coords, probs)

        # Apply negative feedback on K
        for _ in range(3):
            self.agent.record_feedback("K", reward=-1.0, reason="user_deleted")

        p_after, _ = self.agent.adapt_probabilities(coords, probs)
        self.assertLess(p_after[k_idx], p_init[k_idx])
        self.assertEqual(self.agent.negative_rewards, 3)

    def test_policy_save_load_persistence(self):
        """Trained weights and telemetry must persist across disk save/load cycles."""
        coords = self.canon["I"]
        probs = self._dummy_probs("I", 0.50)
        self.agent.adapt_probabilities(coords, probs)
        self.agent.record_feedback("I", reward=1.0, reason="test")

        # Load fresh agent pointing to same policy file
        agent2 = AdaptiveSignRLAgent(policy_path=self.temp_policy, enabled=True)
        self.assertEqual(agent2.total_episodes, 1)
        self.assertEqual(agent2.positive_rewards, 1)
        np.testing.assert_array_almost_equal(agent2.weights, self.agent.weights)

    def test_policy_reset(self):
        """Resetting policy must return weights to zero and clear telemetry."""
        coords = self.canon["H"]
        probs = self._dummy_probs("H", 0.50)
        self.agent.adapt_probabilities(coords, probs)
        self.agent.record_feedback("H", reward=1.0, reason="test")

        self.agent.reset_policy()
        self.assertEqual(self.agent.total_episodes, 0)
        self.assertEqual(self.agent.cumulative_reward, 0.0)
        self.assertEqual(np.count_nonzero(self.agent.weights), 0)


if __name__ == "__main__":
    unittest.main()
