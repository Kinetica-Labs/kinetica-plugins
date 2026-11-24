"""The narration plugin without an editor: lines type out at TYPE letters a second and only the newest is typed, the plate
keeps LINES, a long sentence wraps, a part's light rises, settles and goes out when the next line's part takes it over."""
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kinetica-editor", "clients", "python"))
sys.path.insert(0, HERE)
import plugin  # noqa: E402


class Narration(unittest.TestCase):
    def setUp(self):
        plugin._said.clear(), plugin._lit.clear()
        plugin._wake = lambda: None  # no editor to draw in

    def test_the_newest_line_types_out(self):
        plugin.say(None, "Ставлю шасси", target=3)
        born = plugin._said[-1][1]
        text, lights, moving = plugin.frame(born + 4.5 / plugin.TYPE)
        self.assertEqual(text, "Став▍")
        self.assertTrue(moving)
        text, _, _ = plugin.frame(born + 10)
        self.assertEqual(text, "Ставлю шасси")

    def test_the_plate_keeps_whole_phrases(self):
        for k in range(5):
            plugin.say(None, f"шаг {k}", target=0)
        self.assertEqual([phrase for phrase, _ in plugin._said], ["шаг 2", "шаг 3", "шаг 4"])
        plugin.say(None, "слово " * 20, target=0)  # three lines wrapped: one short phrase above it fits, not two
        text, _, _ = plugin.frame(time.monotonic() + 60)
        lines = text.split("\n")
        self.assertEqual(lines[0], "шаг 4")
        self.assertTrue(len(lines) <= plugin.MOST and all(len(line) <= plugin.WIDE for line in lines))
        self.assertEqual(plugin.say(None, "   ", target=0)["said"], False)

    def test_the_light_passes_to_the_next_part(self):
        self.assertAlmostEqual(plugin.light(plugin.RISE), 1.0)
        self.assertAlmostEqual(plugin.light(10), plugin.HOLD)
        self.assertAlmostEqual(plugin.light(10, plugin.FADE), 0.0)
        plugin.say(None, "кабина", target=5)
        plugin.say(None, "мотор", target=7)
        self.assertIsNotNone(plugin._lit[5][1])  # going out
        self.assertIsNone(plugin._lit[7][1])     # the newest's
        _, lights, _ = plugin.frame(time.monotonic() + 10)
        self.assertEqual(lights[5], 0.0)
        self.assertNotIn(5, plugin._lit)
        self.assertAlmostEqual(lights[7], plugin.HOLD)


if __name__ == "__main__":
    unittest.main()
