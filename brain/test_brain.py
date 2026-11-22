"""The brain's tests (python3 -m unittest test_brain, in this folder): the net against numbers worked by hand, its memory,
the weights as the scene keeps them, the balancer's wiring, and selection's result the same for 1 and 4 workers. The
editor's Python package and kinetica-core are found as evolution.py finds them (KE_EDITOR, else ../../kinetica-editor)."""
import math
import os
import tempfile
import unittest

import evolution
import network
import scenes
from robot import Wiring


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


if __name__ == "__main__":
    unittest.main()
