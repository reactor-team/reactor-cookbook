# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Offline consistency checks for the robotics quickstarts."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "xwam": ("xwam_quickstart.md", "xwam_quickstart.py"),
    "lingbot-va": ("lingbot_va_quickstart.md", "lingbot_va_quickstart.py"),
    "cosmos-nano-policy-droid": (
        "cosmos_droid_quickstart.md",
        "cosmos_droid_quickstart.py",
    ),
    "groot-n17": ("groot_n17_quickstart.md", "groot_n17_quickstart.py"),
    "dreamzero": ("dreamzero_quickstart.md", "dreamzero_quickstart.py"),
    "xr1-robocasa365": (
        "xr1_robocasa365_quickstart.md",
        "xr1_robocasa365_quickstart.py",
    ),
}

FIXTURE_ACTIONS = {
    "xwam_examples.npz": ("expected_actions", (5, 32, 14)),
    "lingbot_va_examples.npz": ("expected_actions", (5, 16, 7)),
    "cosmos_droid_examples.npz": ("expected_actions", (5, 32, 8)),
    "groot_n17_examples.npz": ("expected_joint_position", (5, 40, 7)),
    "dreamzero_examples.npz": ("expected_actions", (5, 24, 8)),
    "xr1_robocasa365_examples.npz": ("expected_actions", (5, 16, 60)),
}


class DocumentationConsistencyTest(unittest.TestCase):
    def test_every_model_has_a_guide_and_script(self) -> None:
        readme = (ROOT / "README.md").read_text()
        for model, (guide, script) in MODELS.items():
            with self.subTest(model=model):
                self.assertTrue((ROOT / guide).is_file())
                self.assertTrue((ROOT / script).is_file())
                self.assertIn(model, readme)
                self.assertIn(guide, readme)

    def test_all_relative_markdown_links_resolve(self) -> None:
        link_pattern = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
        for markdown in ROOT.rglob("*.md"):
            text = markdown.read_text()
            for match in link_pattern.finditer(text):
                target = match.group(1).split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                with self.subTest(document=markdown.name, target=target):
                    self.assertTrue((markdown.parent / target).resolve().exists())

    def test_recorded_action_shapes_match_the_guides(self) -> None:
        for filename, (key, shape) in FIXTURE_ACTIONS.items():
            with self.subTest(fixture=filename):
                with np.load(ROOT / "examples" / filename, allow_pickle=False) as data:
                    self.assertEqual(data[key].shape, shape)


if __name__ == "__main__":
    unittest.main()
