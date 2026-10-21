import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_depth import depth_to_rgb
from render import save_report_and_reset


class PreparationTests(unittest.TestCase):
    def test_fixed_bounds_and_invalid_pixels(self):
        depth = np.array([[0.5, 1.5, 2.5, 0, np.nan, np.inf, -1]], dtype=np.float32)
        rgb = depth_to_rgb(depth, 0.5, 2.5)
        self.assertEqual(rgb[0, :, 0].tolist(), [255, 128, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])
        self.assertEqual(depth_to_rgb(np.array([[1.5, 9.0]]), .5, 2.5)[0, 0, 0], 128)

    def test_rejects_ambiguous_input(self):
        for depth, near, far in [(np.ones((2, 2), dtype=np.uint16), .1, 2),
                                 (np.ones((2, 2)), 2, 1),
                                 (np.ones((2, 2, 3)), .1, 2)]:
            with self.assertRaises(ValueError):
                depth_to_rgb(depth, near, far)

    def test_cleanup_failure_retains_successful_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            class Client:
                async def send_command(self, name, data):
                    self.saved = json.loads((output / "report.json").read_text())
                    raise ConnectionError("disconnected")
            client = Client()
            asyncio.run(save_report_and_reset(client, output, {"generated_frames": 121}))
            self.assertEqual(client.saved["generated_frames"], 121)
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved["generated_frames"], 121)
            self.assertIn("ConnectionError", saved["cleanup_warning"])


if __name__ == "__main__":
    unittest.main()
