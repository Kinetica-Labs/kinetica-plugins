"""The terrain plugin's ground without a window (python3 -m unittest test_terrain, in this folder): the same seed the same
hills, flat about the middle, never below the floor, slopes under 12°."""
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kinetica-editor", "clients", "python"))
sys.path.insert(0, HERE)
import plugin  # noqa: E402


class TerrainTests(unittest.TestCase):
    def test_ground(self):
        hills = plugin.hills_of(1, 6, 30.0, 1.0, 3.0)
        self.assertEqual(hills, plugin.hills_of(1, 6, 30.0, 1.0, 3.0))
        self.assertLess(plugin.height_at(1.0, 2.0, hills, 3.0), 0.005)  # the flat middle: only the hills' feet, millimetres
        step, steepest = 0.1, 0.0
        for i in range(-140, 140, 3):
            for k in range(-140, 140, 3):
                x, z = i * step, k * step
                h = plugin.height_at(x, z, hills, 3.0)
                self.assertGreaterEqual(h, 0.0)
                steepest = max(steepest, abs(plugin.height_at(x + step, z, hills, 3.0) - h) / step, abs(plugin.height_at(x, z + step, hills, 3.0) - h) / step)
        self.assertLess(math.degrees(math.atan(steepest)), 12.0)

    def test_obj(self):
        with tempfile.TemporaryDirectory() as d:
            low, high = plugin.write_obj(os.path.join(d, "t.obj"), 10.0, 1.0, plugin.hills_of(2, 2, 10.0, 0.5, 2.0), 2.0)
            text = open(os.path.join(d, "t.obj")).read()
        self.assertEqual(text.count("\nv "), 121)  # 11 x 11 points
        self.assertEqual(text.count("\nf "), 200)
        self.assertTrue(0.0 <= low <= high)


if __name__ == "__main__":
    unittest.main()
