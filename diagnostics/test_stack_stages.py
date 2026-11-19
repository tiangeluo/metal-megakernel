"""The split diagnostic must reproduce the ordinary connected CPU call."""
import unittest

from bench.protocol import Case
from decoder.model import Config
from diagnostics.stack_stages import difference, isolate


class StageReplayTests(unittest.TestCase):
    def test_matched_inputs_on_a_wrapped_cache(self):
        result = isolate(Case("stack", 2, 6), 20261118, "cpu", config=Config.compact())
        for checks in result["layers"].values():
            for key in ("prepare", "attention_on_observed_token", "finish_on_observed_heads",
                        "expert_components", "replay_against_original_call"):
                for name, check in checks[key].items():
                    self.assertTrue(check if isinstance(check, bool) else check["passed"], (key, name))
            self.assertEqual(checks["split_replay_vs_compiled_partial"]["different"], 0)

    def test_exact_routes_even_with_large_ids(self):
        self.assertFalse(difference([300], [301])["passed"])
