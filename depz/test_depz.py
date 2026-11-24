"""The BNO086 protocol's tests (python3 -m unittest test_depz, in this folder), against bytes worked by hand: the CRC's
check value, packets cut from a stream in any pieces after noise, a Set Feature frame, the reports of an input frame with
their times, the angles of a quaternion. No board needed."""
import math
import struct
import unittest

import bno086
import hand


class ProtocolTests(unittest.TestCase):
    def test_crc8_maxim(self):
        self.assertEqual(bno086.crc8(b"123456789"), 0xA1)  # the catalogue's check value of CRC-8/MAXIM

    def test_packets_any_pieces(self):
        one, two = bno086.packet(0x81, 7, b"\x03DEPZ\0"), bno086.packet(0x80, 8, b"\x34\x00")
        broken = bytearray(one)
        broken[6] ^= 1  # a false start: its header fails the CRC
        stream = b"\x00\xa5" + bytes(broken) + one + two
        cut, got = bno086.Packets(), []
        for i in range(len(stream)):
            got += cut.feed(stream[i:i + 1])
        self.assertEqual(got, [(0x81, 7, b"\x03DEPZ\0"), (0x80, 8, b"\x34\x00")])

    def test_set_feature(self):
        frame = bno086.set_feature(0x08, 10000, 5)
        self.assertEqual(len(frame), 21)
        self.assertEqual(struct.unpack_from("<HBB", frame), (21, 2, 5))
        self.assertEqual(frame[4:6], b"\xfd\x08")
        self.assertEqual(struct.unpack_from("<I", frame, 9)[0], 10000)

    def test_samples_and_times(self):
        s = math.sqrt(0.5)
        quat = bytes([0x08, 1, 0b111, 0x02]) + struct.pack("<4h", 0, 0, round(s * 16384), round(s * 16384))
        accel = bytes([0x01, 1, 0b10, 0]) + struct.pack("<3h", 0, 0, round(9.81 * 256))
        unknown = bytes([0x7E, 0, 0, 0])
        body = b"\xfb" + struct.pack("<i", 10) + quat + accel + unknown
        frame = struct.pack("<HBB", 4 + len(body), 3, 0) + body
        (q, a) = bno086.samples(frame, 1_000_000)
        self.assertEqual((q[0], q[1], q[3]), ("quat", 1_000_000 - 1000 + (1 << 8 | 2) * 100, 3))
        for got, want in zip(q[2], (s, 0, 0, s)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertEqual((a[0], a[1], a[3]), ("accel", 999_000, 2))
        self.assertAlmostEqual(a[2][2], 9.81, places=2)
        self.assertEqual(bno086.samples(struct.pack("<HBB", 4 + len(body), 2, 0) + body, 0), [], "the control channel has no samples")

    def test_euler(self):
        s = math.sqrt(0.5)
        for q, want in (((s, 0, 0, s), (0, 0, 90)), ((s, s, 0, 0), (90, 0, 0)), ((math.sqrt(0.75), 0, 0.5, 0), (0, 60, 0))):
            for got, w in zip(map(math.degrees, bno086.euler(q)), want):
                self.assertAlmostEqual(got, w, places=6)


class HandTests(unittest.TestCase):
    """The board as the arm's puppet strings (hand.py): its turn from the zero pose to the joints' targets."""

    @staticmethod
    def about(axis, radians):
        half = radians / 2
        return (math.cos(half), *[v * math.sin(half) for v in axis])

    def test_the_zero_pose_leaves_the_arm_where_it_stood(self):
        q = (0.9, 0.1, -0.3, 0.3)
        n = math.sqrt(sum(v * v for v in q))
        q = tuple(v / n for v in q)
        rest = [0.1, -0.2, 0.3, 0.0, 0.0, 0.5]
        self.assertEqual([round(v, 9) for v in hand.targets(hand.angles(hand.relative(q, q)), rest)], rest)

    def test_each_angle_leads_its_joints(self):
        zero, rest = (1.0, 0.0, 0.0, 0.0), [0.0] * 6
        yaw = hand.targets(hand.angles(hand.relative(self.about((0, 0, 1), 0.5), zero)), rest)
        self.assertAlmostEqual(yaw[0], 0.5)
        self.assertEqual(yaw[1:], [0.0] * 5)
        pitch = hand.targets(hand.angles(self.about((0, 1, 0), 0.3)), rest)
        self.assertAlmostEqual(pitch[1], 0.3)
        self.assertAlmostEqual(pitch[2], -0.18)  # the elbow bent back by 0.6 of it
        roll = hand.targets(hand.angles(self.about((1, 0, 0), -0.7)), rest, flip=("roll",))
        self.assertAlmostEqual(roll[5], 0.7)     # reversed
        self.assertEqual(len(hand.targets(hand.angles(self.about((1, 0, 0), 1.0)), [0.0] * 3)), 3)  # no wrist: skipped

    def test_led_no_further_than_the_limit_and_smoothly(self):
        far = hand.targets({"yaw": 9.0, "pitch": 0.0, "roll": 0.0}, [0.2] * 6)
        self.assertAlmostEqual(far[0], 0.2 + hand.LIMIT)
        shown = [0.0]
        for _ in range(60):  # a second of frames
            shown = hand.smooth(shown, [1.0], 1 / 60)
        self.assertGreater(shown[0], 0.99)
        self.assertLess(hand.smooth([0.0], [1.0], 1 / 60)[0], 0.25)  # no jump in one frame


if __name__ == "__main__":
    unittest.main()
