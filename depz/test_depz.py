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
    """The board turns the wrist (hand.py): its turn since zero, carried into the world by how it is held, solved into
    three joint angles."""

    def test_the_zero_pose_asks_no_turn(self):
        q = hand.about((0.6, 0.0, 0.8), 0.9)
        self.assertLess(max(abs(v) for v in hand.log(hand.wanted(q, q, hand.held((0, 0, -1))))), 1e-9)

    def test_held_x_into_the_screen_z_up(self):
        mount = hand.held((0.3, -0.5, -0.9))  # a view looking down a little: laid flat
        x, z = hand.rotate(mount, (1, 0, 0)), hand.rotate(mount, (0, 0, 1))
        n = math.hypot(0.3, 0.9)
        self.assertAlmostEqual(x[0], 0.3 / n), self.assertAlmostEqual(x[1], 0.0), self.assertAlmostEqual(x[2], -0.9 / n)
        self.assertAlmostEqual(z[1], 1.0)

    def test_a_turn_about_the_board_up_turns_the_hand_about_world_up(self):
        turn = hand.wanted(hand.about((0, 0, 1), 0.7), (1.0, 0.0, 0.0, 0.0), hand.held((0, 0, -1)))
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(hand.log(turn), (0.0, 0.7, 0.0))))

    def test_the_wrist_solved_from_its_last_angles(self):
        axes = [(-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)]  # the UR5e's wrist at rest
        th = [0.4, -0.7, 1.1]
        turn = hand.mul(hand.mul(hand.about(axes[0], th[0]), hand.about(axes[1], th[1])), hand.about(axes[2], th[2]))
        got = hand.solve(turn, axes, [0.3, -0.6, 1.0])  # the last frame's: near
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(got, th)))

    def test_smoothly(self):
        shown = [0.0]
        for _ in range(60):  # a second of frames
            shown = hand.smooth(shown, [1.0], 1 / 60)
        self.assertGreater(shown[0], 0.99)
        self.assertLess(hand.smooth([0.0], [1.0], 1 / 60)[0], 0.3)  # no jump in one frame


if __name__ == "__main__":
    unittest.main()
