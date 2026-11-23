"""The brain's tests (python3 -m unittest test_brain, in this folder): the net against numbers worked by hand, its memory,
the weights as the scene keeps them, the balancer's wiring, and selection's result the same for 1 and 4 workers. The
editor's Python package and kinetica-core are found as evolution.py finds them (KE_EDITOR, else ../../kinetica-editor)."""
import json
import math
import os
import tempfile
import unittest

import drive
import evolution
import network
import scenes
from robot import Wiring


class ExamStatisticsTests(unittest.TestCase):
    def test_clean_share_exact(self):
        low, high = evolution.clean_share(100, True)
        self.assertAlmostEqual(low, 0.025 ** (1 / 100), places=12)   # 100 of 100: clean in 96.4-100 %, not «always»
        self.assertEqual(high, 1.0)
        low, high = evolution.clean_share(3.4, False)                  # 3 clean laps, the fourth with its error
        self.assertAlmostEqual(1 - high, 1 - 0.975 ** 0.25, places=9)  # Clopper-Pearson for 1 error in 4: q in [0.0063, 0.806]
        self.assertAlmostEqual(low, 0.194, places=3)
        self.assertEqual(evolution.clean_share(0.2, False), (0.0, 0.975))  # the first lap's error
        self.assertEqual(evolution.clean_share(0, True), (0.0, 1.0))


class NetTests(unittest.TestCase):
    def test_forward_by_hand(self):
        net = network.Net(2, [1], 1)  # x -> h = tanh(w1 x1 + w2 x2 + b1) -> y = tanh(w3 h + b2)
        w = [0.5, -1.0, 0.1, 2.0, -0.2]
        h = math.tanh(0.5 * 0.3 - 1.0 * 0.6 + 0.1)
        self.assertAlmostEqual(net.forward(w, [0.3, 0.6])[0], math.tanh(2.0 * h - 0.2), places=12)
        self.assertEqual(net.count(), 5)

    def test_memory(self):
        net = network.Net(1, [2], 1, recurrent=True)
        w = network.Net(1, [2], 1, recurrent=True).random_weights(network.new_rng(3))
        first = net.forward(w, [1.0])
        self.assertNotEqual(net.forward(w, [1.0]), first, "a recurrent net remembers the step before")
        net.reset()
        self.assertEqual(net.forward(w, [1.0]), first, "reset: the run begins again")

    def test_echo(self):
        net = network.Net(1, [2], 2, echo=True)  # it sees 1 input and its 2 outputs of the step before
        w = net.random_weights(network.new_rng(5))
        self.assertEqual(net.sizes[0], 3)
        first = net.forward(w, [0.5])
        self.assertEqual(net.seen[0], [0.5, 0.0, 0.0], "a run begins with nothing told")
        net.forward(w, [0.5])
        self.assertEqual(net.seen[0], [0.5] + first, "then it hears what it told")

    def test_a_brain_file_round_trip(self):
        net = network.Net(3, [4], 2, echo=True)
        w = net.random_weights(network.new_rng(1))
        doc = network.read_brain_file(json.dumps(network.brain_file(net, w, {"drive": "car"}, "3 входа")))
        self.assertEqual((doc["net"]["inputs"], doc["net"]["layers"], doc["net"]["outputs"], doc["net"]["echo"]), (3, [4], 2, True))
        self.assertEqual(doc["weights"], w)
        with self.assertRaises(ValueError):
            network.read_brain_file(json.dumps({**doc, "weights": w[:-1]}))

    def test_weights_as_the_scene_keeps_them(self):
        w = [network.stored(x) for x in (0.123456789, -3.14159265, 1e-7)]
        as_float32 = [float(f"{x:.9g}") for x in w]  # the editor reads them back single precision
        self.assertEqual(network.weights_of(network.weights_text(w)), w)
        self.assertEqual(network.weights_of(as_float32), w)


class RobotTests(unittest.TestCase):
    def test_balancer_wiring(self):
        with evolution._kinetica().Scene() as scene:
            cart = scenes.balance(scene)
            wiring = Wiring(scene, cart)
            kinds = [kind for _, _, kind, _ in wiring.inputs]
            self.assertEqual(kinds, ["encoder", "imu", "lidar"])
            self.assertEqual((wiring.size, len(wiring.outputs), len(wiring.joints)), (8, 1, 1))  # 1 + 6 + 1 in; the push is no output


class EvolutionTests(unittest.TestCase):
    def test_one_seed_one_result_whatever_the_workers(self):
        with evolution._kinetica().Scene() as scene, tempfile.TemporaryDirectory() as folder:
            cart = scenes.balance(scene)
            file = os.path.join(folder, "balance.kscene")
            scene.ed.scene_save(file=file)
            brain = {"layers": "4", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 15.0}
            runs = [evolution.train(file, cart, brain, population=6, generations=2, seconds=1.0, workers=n, seed=7) for n in (1, 4)]
            self.assertEqual(runs[0][0], runs[1][0])
            self.assertEqual(runs[0][1], runs[1][1])
            self.assertGreater(runs[0][1], 0.0)


class RaceTests(unittest.TestCase):
    def test_cars_side_by_side_one_result_whatever_the_workers(self):
        """Five cars a world on the mesh track: the same scores for 1 and 3 workers; the track is one body."""
        with evolution._kinetica().Scene() as scene, tempfile.TemporaryDirectory() as folder:
            car = scenes.race(scene)
            self.assertEqual(len(scene.ed.scene_list(kind="entity")["objects"]), 3)  # the floor, the track, the car
            file = os.path.join(folder, "race.kscene")
            scene.ed.scene_save(file=file)
            brain = {"layers": "4", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 6.0, "drive": "car", "speed": 4.0}
            runs = [evolution.train(file, car, brain, population=10, generations=2, seconds=2.0, workers=n, seed=3, judge="track", cars=5)
                    for n in (1, 3)]
            self.assertEqual(runs[0][2], runs[1][2])

    def test_more_tracks_one_result_whatever_the_workers(self):
        """A random track besides the scene's: the scores the worst of the two, the same for 1 and 3 workers."""
        with evolution._kinetica().Scene() as scene, tempfile.TemporaryDirectory() as folder:
            car = scenes.race(scene)
            brain = {"layers": "4", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 6.0, "drive": "car", "speed": 4.0,
                     "reflex": 6.0, "echo": True}
            scene.ed.component_set(id=car, type="brain.net", values=brain)
            file = os.path.join(folder, "race.kscene")
            scene.ed.scene_save(file=file)
            more = evolution.courses(brain, 1, 3, folder)
            runs = [evolution.train(file, car, brain, population=10, generations=2, seconds=2.0, workers=n, seed=3, judge="track", cars=5,
                                    tracks=more) for n in (1, 3)]
            self.assertEqual(runs[0][2], runs[1][2])
            alone = evolution.train(file, car, brain, population=10, generations=1, seconds=2.0, workers=1, seed=3, judge="track", cars=5)
            self.assertLessEqual(runs[0][2][0][0], alone[2][0][0])  # the worst of two tracks is no more than this one alone

    def test_cars_ignore_each_other_in_a_scene_tagged_before(self):
        """A scene whose car already has the tag (saved after a training): its copies still pass through each other."""
        with evolution._kinetica().Scene() as scene, tempfile.TemporaryDirectory() as folder:
            car = scenes.race(scene)
            scene.ed.scene_tag(ids=[car], tag=evolution.TAG, on=True)
            file = os.path.join(folder, "tagged.kscene")
            scene.ed.scene_save(file=file)
            arena = evolution.Arena(file, car, "track", 3, "car")
            try:
                tagged = [o["id"] for o in arena.scene.ed.scene_list(tag=evolution.TAG)["objects"]]
                self.assertEqual(sorted(w.body for w in arena.wirings), sorted(tagged))  # each tagged, each ignoring the tag
            finally:
                arena.close()

    def test_a_touch_is_a_red_card(self):
        """The judge rules a car out the frame a corner of it reaches a wall, though its middle is well on the track."""
        with evolution._kinetica().Scene() as scene:
            car = scenes.race(scene)
            scene.ed.sim_step(frames=0)
            at = next(x for x in scene.ed.sim_state()["shapes"] if x["id"] == car)
            _, side = drive.axes(at["rotation"])
            judge = evolution.Track(scene, Wiring(scene, car, drive="car"))
            def looked(shift):  # the car moved sideways by `shift` m, as turned as it is
                x, z = at["position"][0] + shift * side[0], at["position"][2] + shift * side[2]
                return judge.step({f"{car}/centre x": x, f"{car}/centre z": z, f"{car}/rotation": at["rotation"]}, 1 / 60)
            self.assertTrue(looked(0.0))
            self.assertTrue(looked(judge.half - 0.25))  # its side 10 cm from the wall
            self.assertFalse(looked(judge.half - 0.14))  # its side at the wall: out

    def test_the_example_brain_drives_the_race(self):
        """The trained brain the race example comes with: two laps of the example's track, not a touch."""
        with evolution._kinetica().Scene() as scene, tempfile.TemporaryDirectory() as folder:
            car = scenes.race(scene)
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "race.brain.json"), encoding="utf-8") as f:
                doc = network.read_brain_file(f.read())
            scene.ed.component_set(id=car, type="brain.net", values={**doc["card"], "layers": "16, 8"})
            file = os.path.join(folder, "race.kscene")
            scene.ed.scene_save(file=file)
            brain = dict(doc["card"], layers="16, 8", recurrent=False, echo=True)
            laps, seconds, finished = evolution.exam(file, car, brain, doc["weights"], laps=2)
            self.assertTrue(finished, f"{laps:.2f} laps, then a red card")

    def test_the_circuit_is_drivable(self):
        """Its tightest turn takes the most speed of the race's card on the tyres' grip: v^2 / R <= grip."""
        self.assertLess(4.0 ** 2 / scenes.tightest(scenes.spline(scenes.CIRCUIT)), drive.GRIP)


class DriveTests(unittest.TestCase):
    def test_tyres_take_the_skid_out_and_the_motor_holds_the_speed(self):
        car = {"rotation": [0.0, 90.0, 0.0], "velocity": [0.0, 0.0, 0.0], "mass": 2.0}  # its x along the world's -z
        forward, _ = drive.axes(car["rotation"])
        self.assertAlmostEqual(forward[2], -1.0, places=9)
        push = drive.tyres(car, 4.0)
        self.assertAlmostEqual(push[2], -2.0 * drive.PUSH, places=9)  # short of 4 m/s: the motor's most, forward
        skid = drive.tyres(dict(car, velocity=[1.0, 0.0, 0.0]), 0.0)  # sliding sideways at 1 m/s
        self.assertAlmostEqual(skid[0], -2.0 * drive.GRIP, places=9)  # the grip's most, against the skid

    def test_the_plan(self):
        card = {"speed": 6.0, "speed_min": 1.0, "turn": 3.0}
        self.assertEqual(drive.plan([1.0, -1.0], card), (3.0, 1.0))
        self.assertEqual(drive.plan([-0.5, 1.0], card), (-1.5, 6.0))
        self.assertEqual(drive.plan([0.25, -1.0], dict(card, reflex=2.0), 0.25), (2.25, 1.0))  # the reflex turns it away more
        self.assertEqual(drive.plan([0.5, -1.0], dict(card, reflex=6.0), 0.5)[0], 3.0)  # no more than the steering gives


if __name__ == "__main__":
    unittest.main()
