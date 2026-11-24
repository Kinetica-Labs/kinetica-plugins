"""The chess plugin's planner without an editor: a move is eight points over, down, grip, up, over, down, open, up at
the squares' centres; a capture first carries the taken piece off the board onto the floor; the pieces' places follow;
what is not a move is said in words; the time is the lead speed's and the settles'."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kinetica-editor", "clients", "python"))
sys.path.insert(0, HERE)
import plugin  # noqa: E402

CENTRE = (0.0, -0.5)


class Chess(unittest.TestCase):
    def test_a_move_is_eight_points_at_the_squares(self):
        steps, after = plugin.plan("e2e4", CENTRE, {"e2": 7})
        self.assertEqual([what for what, _, _ in steps], ["go", "go", "grip", "go", "go", "go", "grip", "go"])
        (x2, z2), (x4, z4) = plugin.square("e2", CENTRE), plugin.square("e4", CENTRE)
        self.assertEqual((steps[1][1][0], steps[1][1][2]), (x2, z2))
        self.assertEqual((steps[5][1][0], steps[5][1][2]), (x4, z4))
        self.assertAlmostEqual(z2 - z4, 2 * plugin.SQUARE)       # two ranks away from the arm
        self.assertEqual((steps[2][1], steps[6][1]), (True, False))
        self.assertEqual(after, {"e4": 7})
        self.assertGreater(plugin.seconds(steps, steps[0][1]), 8 * plugin.SETTLE / 2)

    def test_a_capture_carries_the_taken_piece_off_first(self):
        steps, after = plugin.plan("e4d5", CENTRE, {"e4": 7, "d5": 9})
        self.assertEqual(len(steps), 16)
        off = steps[5][1]
        self.assertGreater(off[0] - CENTRE[0], 4 * plugin.SQUARE)               # beside the board
        self.assertAlmostEqual(off[1], steps[13][1][1] - plugin.BOARD_TOP)      # set down on the floor, not at the board's height
        self.assertEqual(after, {"d5": 7, "x0": 9})

    def test_what_is_no_move_is_said(self):
        for move, words in (("e9e4", "e2e4"), ("e3e4", "e3"), ("e2e2", "e2")):
            with self.assertRaises(ValueError) as said:
                plugin.plan(move, CENTRE, {"e2": 7})
            self.assertIn(words, str(said.exception))


if __name__ == "__main__":
    unittest.main()
