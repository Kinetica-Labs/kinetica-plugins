"""The narration plugin without an editor: the card's label counts the steps, its words type out at TYPE letters a second after
WORDS_AT, it slides in and shows within SLIDE, the same card said again stays; a part's light rises, settles and goes out
when the next card's part takes it over."""
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
        plugin._card.clear(), plugin._lit.clear()
        plugin._wake = lambda: None  # no editor to draw in

    def test_a_card_slides_in_and_types_its_words(self):
        plugin.say(None, "Ставлю шасси", title="Шасси", target=3)
        born = plugin._card["born"]
        card, _, moving = plugin.frame(born)
        self.assertEqual((card["title"], card["opacity"], card["text"]), ("Шаг 1 · Шасси", 0.0, "▍"))
        self.assertLess(card["place"][0], plugin.PLACE[0])
        card, _, moving = plugin.frame(born + plugin.WORDS_AT + 4.5 / plugin.TYPE)
        self.assertEqual(card["text"], "Став▍")
        self.assertTrue(moving)
        card, _, moving = plugin.frame(born + 10)
        self.assertEqual((card["text"], card["opacity"], card["place"][0], moving), ("Ставлю шасси", 1.0, plugin.PLACE[0], False))

    def test_steps_count_and_a_repeat_is_the_same_card(self):
        plugin.say(None, "раз", title="", target=0)
        plugin.say(None, "раз", title="", target=0)
        plugin.say(None, "слово " * 20, title="Плечо", target=0)
        card, _, _ = plugin.frame(time.monotonic() + 60)
        self.assertEqual(card["title"], "Шаг 2 · Плечо")
        self.assertTrue(all(len(line) <= plugin.WIDE for line in card["text"].split("\n")))
        self.assertEqual(plugin.say(None, "   ", title="", target=0)["said"], False)

    def test_the_light_passes_to_the_next_part(self):
        self.assertAlmostEqual(plugin.light(plugin.RISE), 1.0)
        self.assertAlmostEqual(plugin.light(10), plugin.HOLD)
        self.assertAlmostEqual(plugin.light(10, plugin.FADE), 0.0)
        plugin.say(None, "кабина", title="", target=5)
        plugin.say(None, "мотор", title="", target=7)
        self.assertIsNotNone(plugin._lit[5][1])  # going out
        self.assertIsNone(plugin._lit[7][1])     # the newest's
        _, lights, _ = plugin.frame(time.monotonic() + 10)
        self.assertEqual(lights[5], 0.0)
        self.assertNotIn(5, plugin._lit)
        self.assertAlmostEqual(lights[7], plugin.HOLD)


if __name__ == "__main__":
    unittest.main()
