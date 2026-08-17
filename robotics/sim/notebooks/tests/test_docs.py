# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Offline consistency checks for the robotics quickstarts."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOTICS_ROOT = ROOT.parents[1]

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
    "dreamzero-yam-molmoact2": (
        "dreamzero_yam_bridge.md",
        "dreamzero_yam_bridge.py",
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
        heading_pattern = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
        for markdown in ROBOTICS_ROOT.rglob("*.md"):
            text = markdown.read_text()
            for match in link_pattern.finditer(text):
                raw_target = match.group(1)
                target, separator, anchor = raw_target.partition("#")
                if "://" in target or target.startswith("mailto:"):
                    continue
                destination = markdown if not target else (markdown.parent / target)
                destination = destination.resolve()
                with self.subTest(document=markdown.name, target=target):
                    self.assertTrue(destination.exists())
                    if separator:
                        anchor_document = (
                            destination / "README.md"
                            if destination.is_dir()
                            else destination
                        )
                        self.assertTrue(anchor_document.is_file())
                        headings = {
                            re.sub(r"\s+", "-", re.sub(r"[^\w\- ]", "", heading.lower()))
                            for heading in heading_pattern.findall(
                                anchor_document.read_text()
                            )
                        }
                        self.assertIn(anchor, headings)

    def test_recorded_action_shapes_match_the_guides(self) -> None:
        for filename, (key, shape) in FIXTURE_ACTIONS.items():
            with self.subTest(fixture=filename):
                with np.load(ROOT / "examples" / filename, allow_pickle=False) as data:
                    self.assertEqual(data[key].shape, shape)


if __name__ == "__main__":
    unittest.main()
