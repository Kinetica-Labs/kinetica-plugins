"""The «Пульт» plugin's arithmetic without a window or a board (python3 -m unittest test_teleop, in this folder): the
tilt from the zero pose, the dead zone, the stick's signs, the joystick's raw events."""
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kinetica-editor", "clients", "python"))
sys.path.insert(0, HERE)
import plugin  # noqa: E402


def about(axis, degrees):
    h = math.radians(degrees) / 2
    return (math.cos(h), *(math.sin(h) * a for a in axis))


class TeleopTests(unittest.TestCase):
    def test_tilt_from_any_zero(self):
        zero = about((0.0, 0.0, 1.0), 146.0)  # held turned about the vertical, as the board lay
        q = _mul(zero, about((1.0, 0.0, 0.0), 20.0))  # then tilted 20° about its own x
        roll, pitch = plugin.tilt(plugin.relative(q, zero))
        self.assertAlmostEqual(roll, 20.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)

    def test_dead_zone_and_full(self):
        self.assertEqual(plugin.shaped(3.0, 30.0), 0.0)
        self.assertAlmostEqual(plugin.shaped(17.0, 30.0), 0.5)
        self.assertEqual(plugin.shaped(-60.0, 30.0), -1.0)

    def test_stick_signs(self):
        turn, speed = plugin.command_of(1.0, 1.0, 1.0, 2.0)  # right and forward
        self.assertEqual((turn, speed), (-1.0, 2.0))  # turning right is negative (positive: left)
        self.assertAlmostEqual(plugin.command_of(0.0, -1.0, 1.0, 3.0)[1], -1.0)  # back: a third

    def test_joystick_events(self):
        data = struct.pack("<IhBB", 10, 32767, 2, 0) + struct.pack("<IhBB", 11, -16384, 0x82, 1) + struct.pack("<IhBB", 12, 1, 1, 3)
        self.assertEqual([(k, n, round(v, 3)) for k, n, v in plugin.js_events(data)], [(2, 0, 1.0), (2, 1, -0.5), (1, 3, 0.0)])


def _mul(a, b):
    w0, x0, y0, z0 = a
    w, x, y, z = b
    return (w0 * w - x0 * x - y0 * y - z0 * z, w0 * x + x0 * w + y0 * z - z0 * y, w0 * y - x0 * z + y0 * w + z0 * x, w0 * z + x0 * y - y0 * x + z0 * w)


if __name__ == "__main__":
    unittest.main()
