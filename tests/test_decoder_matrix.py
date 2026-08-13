from __future__ import annotations

import unittest
from pathlib import Path

from compatibility.run_decoder_matrix import build_command, hybrid_command


class DecoderMatrixTests(unittest.TestCase):
    def test_hidden_state_builder_option_is_explicit(self):
        without_hidden = build_command("conda", "ryzen-ai-1.8.0", Path("model"), Path("out"), False)
        with_hidden = build_command("conda", "ryzen-ai-1.8.0", Path("model"), Path("out"), True)
        self.assertNotIn("include_hidden_states=true", without_hidden)
        self.assertIn("include_hidden_states=true", with_hidden)

    def test_sdk_specific_hybrid_interfaces(self):
        old = hybrid_command("conda", "ryzen-ai-1.7.1", "1.7.1", Path("input"), Path("output"))
        current = hybrid_command("conda", "ryzen-ai-1.8.0", "1.8.0", Path("input"), Path("output"))
        self.assertEqual(old[-3:], ["--hybrid", "output", "input"])
        self.assertIn("--input", current)
        self.assertIn("--output", current)
        self.assertIn("--eager", current)


if __name__ == "__main__":
    unittest.main()
