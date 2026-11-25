"""The replay plugin without a window (python3 -m unittest test_replay, in this folder): the film of the editor's
example journal - where it starts, which lines become steps and the Play it ends with - and the steps in words."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EDITOR = os.path.join(HERE, "..", "..", "kinetica-editor")
sys.path.insert(0, os.path.join(EDITOR, "clients", "python"))
sys.path.insert(0, HERE)
import plugin  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Replay(unittest.TestCase):
    def setUp(self):
        self.lines = read(os.path.join(EDITOR, "examples", "films", "pendulum-tower.jsonl"))
        ops = json.load(open(os.path.join(EDITOR, "tests", "data", "ops-list.json"), encoding="utf-8"))
        self.edits = {o["name"] for o in (ops["ops"] if isinstance(ops, dict) else ops) if o.get("kind") == "edit"}

    def test_the_film_starts_at_its_snapshot_and_ends_with_its_play(self):
        start, steps, played = plugin.plan(self.lines, self.edits)
        self.assertEqual(self.lines[start]["op"], "journal_snapshot")
        ops = [op for _, op, _ in steps]
        self.assertEqual(ops.count("scene_add"), 9)  # a post, a beam, a rod, a ball, five blocks
        self.assertEqual(ops.count("assembly_joint"), 2)
        self.assertEqual(ops[-1], "sim_play")  # the session pressed Play once, at its end
        self.assertNotIn("app_quit", ops)      # a window's own action is no step of the building
        self.assertGreater(played, 3.0)        # it ran about 4 s

    def test_a_film_with_no_snapshot_has_nowhere_to_begin(self):
        self.assertEqual(plugin.plan([{"op": "scene_add", "ok": True}], self.edits), (None, [], 0.0))

    def test_the_steps_in_words(self):
        names = {3: "Стержень", 4: "Перекладина", 5: "Шар"}
        self.assertEqual(plugin.words("scene_add", {"shape": "sphere", "name": "Шар"}, names), ("Шар", "Ставлю шар «Шар»", 0))
        self.assertEqual(plugin.words("assembly_joint", {"a": 3, "b": 4, "type": "hinge"}, names),
                         ("Шарнир", "Шарнир: «Стержень» к «Перекладина»", 3))
        self.assertEqual(plugin.words("scene_set", {"id": 3, "key": "rotation", "values": [0, 0, -60]}, names),
                         ("Стержень", "«Стержень»: поворот 0 0 -60", 3))


if __name__ == "__main__":
    unittest.main()
