"""The «Пульт» plugin without a window, a board or a pad (python3 -m unittest test_teleop, in this folder): the tilt from
the zero pose, the dead zone, the signs; a pad's raw events by its own map, a controller's bindings from them."""
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kinetica-editor", "clients", "python"))
sys.path.insert(0, HERE)
import gamepad  # noqa: E402
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

    def test_pad_by_its_own_map(self):
        pad = gamepad.Pad([0x00, 0x01, 0x03, 0x04, 0x10, 0x11], [0x131, 0x130, 0x133, 0x134, 0x136, 0x137, 0x138, 0x139])  # a Switch Pro's
        pad.feed(_event(2, 0, 32767) + _event(2, 1, -16384) + _event(2, 3, 1000) + _event(1, 7, 1) + _event(1 | 0x80, 1, 1) + _event(2, 4, 0))
        self.assertEqual(tuple(round(v, 3) for v in pad.value("left_stick")), (1.0, 0.5))  # up is +y
        self.assertEqual(pad.value("right_x"), 0.0)  # in the dead zone
        self.assertEqual((pad.value("right_trigger"), pad.value("a")), (1.0, 1.0))  # ZR, and the south button (B on a Switch) as of opening
        xbox = gamepad.Pad([0x00, 0x01, 0x02], [])
        xbox.feed(_event(2, 2, -32767))
        self.assertEqual(xbox.value("left_trigger"), 0.0)  # an analog trigger at rest

    def test_bindings_as_the_keys_do(self):
        pad = gamepad.Pad([0x00, 0x01], [0x130, 0x138, 0x139])
        panel = plugin.Panel({"steer": "axis", "throttle": "axis", "move": "vector2", "grip": "button"},
                             [("steer", ["left_x"], False, False), ("throttle", ["left_y"], False, False),
                              ("throttle", ["right_trigger"], False, False), ("throttle", ["left_trigger"], True, False),
                              ("move", ["left_stick"], False, False), ("grip", ["a"], False, True)])
        self.assertEqual(panel.changed(pad), {})  # a still pad says nothing
        pad.feed(_event(2, 0, -32767) + _event(1, 2, 1) + _event(1, 0, 1))
        self.assertEqual(panel.changed(pad), {"steer": -1.0, "throttle": 1.0, "move": [-1.0, 0.0], "grip": 1.0})
        pad.feed(_event(1, 0, 0) + _event(1, 1, 1))  # A let go: the toggle holds; ZL with ZR held: no throttle
        self.assertEqual(panel.changed(pad), {"throttle": 0.0})
        pad.feed(_event(1, 0, 1))
        self.assertEqual(panel.changed(pad), {"grip": 0.0})


def _event(kind, number, value):
    return struct.pack("<IhBB", 0, value, kind, number)


def _mul(a, b):
    w0, x0, y0, z0 = a
    w, x, y, z = b
    return (w0 * w - x0 * x - y0 * y - z0 * z, w0 * x + x0 * w + y0 * z - z0 * y, w0 * y - x0 * z + y0 * w + z0 * x, w0 * z + x0 * y - y0 * x + z0 * w)


if __name__ == "__main__":
    unittest.main()
