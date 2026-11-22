"""Training a brain by natural selection (plugin.py starts it from the editor's panel; run this file to train with no
window): a population of weight lists, each played as the robot's brain for some seconds in a world of its own - a
kinetica-core process, the editor's physics without a window - and scored by a judge; the best kept as they are, the
rest born of two parents picked by tournament, their genes mixed and some of them nudged. The worlds run side by side,
one process a CPU core: physics is the cost, the nets are tiny, a graphics card would add nothing.

The judge is the laboratory, not the robot: it reads the physics' truth (sim_measure) to score a run - the brain reads
only its sensors. One seed, one result, whatever the number of workers: the scores come back by the individual's
place, the random numbers are drawn in one thread.

    python3 evolution.py scene.kscene --generations 20      # the body with a «Мозг» in the scene, trained, its weights put
                                                           # into a copy of the scene (--save) or printed
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from network import Net, hidden_of, new_rng, stored, weights_of, weights_text
from robot import Wiring


def _kinetica():
    """The editor's Python package: from the editor's plugin host it is on the path already; a run by hand finds it in
    KE_EDITOR or next to this repository."""
    try:
        import kinetica
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        editor = os.environ.get("KE_EDITOR", os.path.join(here, "..", "..", "kinetica-editor"))
        sys.path.insert(0, os.path.join(editor, "clients", "python"))
        import kinetica
    return kinetica


FRAME = 1.0 / 60  # the world's frame, s


def _channels(scene, **what):
    return {c["id"]: c["value"] for c in scene.ed.sim_measure(**what)["channels"]}


class Upright:
    """A pole kept up: the seconds its joint stays within `limit` rad of where it was built and the body within 2.5 m
    of its start, less a little for wandering. The run ends when either fails."""
    limit = 0.5

    def __init__(self, scene, wiring):
        if not wiring.joints:
            raise ValueError("«upright» judges a joint of the robot: the robot has none")
        self.scene, self.body, self.joint = scene, wiring.body, wiring.joints[0]
        self.alive = self.wander = 0.0
        self.start = None

    def step(self, dt):
        c = _channels(self.scene, objects=[self.body], joints=[self.joint])
        angle = next((v for k, v in c.items() if k.startswith(f"{self.joint}/") and k.endswith("coordinate")), 0.0)
        x, z = c.get(f"{self.body}/centre x", 0.0), c.get(f"{self.body}/centre z", 0.0)
        self.start = self.start or (x, z)
        away = ((x - self.start[0]) ** 2 + (z - self.start[1]) ** 2) ** 0.5
        if abs(angle) > self.limit or away > 2.5:
            return False
        self.alive += dt
        self.wander += away * dt
        return True

    def score(self):
        return self.alive - 0.1 * self.wander / max(self.alive, 1e-9) if self.alive > 0 else 0.0


JUDGES = {"upright": Upright}


class Arena:
    """One world with no window: the scene from its file, the robot wired, an episode played on demand."""

    def __init__(self, scene_file, body, judge):
        self.scene = _kinetica().Scene()
        self.scene.ed.scene_open(file=scene_file)
        self.wiring = Wiring(self.scene, body)
        self.judge = JUDGES[judge]

    def episode(self, net, weights, seconds, rate_hz, power):
        """The robot run for `seconds` with these weights as its brain: the judge's score. Built anew, then from the
        frame it starts from, every 1 / rate_hz: the sensors read, the net's outputs commanded, the world stepped - the
        order of the editor's run (a tick for the frame it starts from, then one after each frame, its commands before
        the next), so a brain trained here plays the same there."""
        ed, wiring = self.scene.ed, self.wiring
        ed.sim_stop()
        ed.sim_step(frames=0)
        judge = self.judge(self.scene, wiring)
        net.reset()
        every = max(1, round(60.0 / float(rate_hz)))
        ids = [s[0] for s in wiring.inputs]
        t = 0.0
        while t < seconds:
            readings = {r["id"]: r for r in ed.sensor_read(ids=ids)["sensors"]} if ids else {}
            outputs = net.forward(weights, wiring.numbers(readings))
            if wiring.outputs:
                ed.robot_command(commands=wiring.commands(outputs, power))
            ed.sim_step(frames=every)
            t += every * FRAME
            if not judge.step(every * FRAME):
                break
        return judge.score()

    def close(self):
        self.scene.close()


def _child(rng, a, b, mutation, rate=0.1):
    """Uniform crossover, then each gene nudged by a normal of `mutation` with probability `rate` - kept as the scene keeps it."""
    child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)]
    return [stored(w + rng.gauss(0.0, mutation)) if rng.random() < rate else w for w in child]


def _pick(rng, scores, k=3):
    """Tournament: the best of k picked at random."""
    return max((rng.randrange(len(scores)) for _ in range(k)), key=lambda i: scores[i])


def train(scene_file, body, brain, population=100, generations=30, seconds=10.0, mutation=0.2, elite=2, workers=0, seed=1,
          judge="upright", report=None, stop=None, first=None):
    """The best weights found, their score, and each generation's (best, mean). `brain`: the component's values
    (layers, activation, recurrent, rate_hz, power). `report(generation, best, mean)` after each generation; `stop`, a
    threading.Event, ends it after the generation it is set in. `first`: weights to seed the population with."""
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    population, elite = max(2, int(population)), max(0, int(elite))
    rng = new_rng(seed)
    local, arenas, lock = threading.local(), [], threading.Lock()

    def arena():
        if not hasattr(local, "arena"):
            local.arena = Arena(scene_file, body, judge)
            local.net = Net(local.arena.wiring.size, hidden_of(brain["layers"]), len(local.arena.wiring.outputs),
                            brain["activation"], brain["recurrent"])
            with lock:
                arenas.append(local.arena)
        return local.arena, local.net

    with _probe_scene(scene_file) as world:
        probe = Wiring(world, body)
    size = Net(probe.size, hidden_of(brain["layers"]), len(probe.outputs), brain["activation"], brain["recurrent"])
    if not probe.outputs:
        raise ValueError("the robot has no force of its own: a brain has nothing to drive")
    genomes = [size.random_weights(rng) for _ in range(population)]
    if first and len(first) == size.count():
        genomes[0] = weights_of(first)
    history, best, best_score = [], genomes[0], float("-inf")

    def score(i):
        a, net = arena()
        return a.episode(net, genomes[i], seconds, brain["rate_hz"], brain["power"])

    os.environ.setdefault("KE_PHYSICS_THREADS", "1")  # a world a core: no worker threads of its own
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for g in range(int(generations)):
                scores = list(pool.map(score, range(population)))
                order = sorted(range(population), key=lambda i: -scores[i])
                if scores[order[0]] > best_score:
                    best, best_score = list(genomes[order[0]]), scores[order[0]]
                mean = sum(scores) / population
                history.append((scores[order[0]], mean))
                if report:
                    report(g, scores[order[0]], mean)
                if stop is not None and stop.is_set():
                    break
                born = [list(genomes[i]) for i in order[:elite]]
                while len(born) < population:
                    born.append(_child(rng, genomes[_pick(rng, scores)], genomes[_pick(rng, scores)], float(mutation)))
                genomes = born
    finally:
        for a in arenas:
            a.close()
    return best, best_score, history


def _probe_scene(scene_file):
    """A world of the scene, to read it or to write a copy of it (a context manager: it closes)."""
    scene = _kinetica().Scene()
    scene.ed.scene_open(file=scene_file)
    return scene


def brain_of(scene, body=None):
    """(body, its brain's values) - the given body's, else the first with a «Мозг»."""
    rows = scene.ed.component_list(type="brain.net")["components"]
    for row in rows:
        if body in (None, row["id"]):
            v = {"layers": "8", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 20.0, "weights": ""}
            v.update(row["values"])
            v["layers"], v["recurrent"] = str(v["layers"]), v["recurrent"] in (True, 1, "1", "true")
            return row["id"], v
    raise ValueError("no body has a «Мозг»: put one on (brain_add)")


def main():
    p = argparse.ArgumentParser(description="Train the brain of a scene's robot by natural selection, with no window.")
    p.add_argument("scene")
    p.add_argument("--body", type=int, default=None, help="the body with the brain (default: the first with one)")
    for name, kind, default in (("population", int, 100), ("generations", int, 30), ("seconds", float, 10.0), ("mutation", float, 0.2),
                                ("elite", int, 2), ("workers", int, 0), ("seed", int, 1)):
        p.add_argument("--" + name, type=kind, default=default)
    p.add_argument("--judge", default="upright", choices=sorted(JUDGES))
    p.add_argument("--save", help="a copy of the scene with the best weights in its brain")
    a = p.parse_args()
    with _probe_scene(os.path.abspath(a.scene)) as scene:
        body, brain = brain_of(scene, a.body)
        report = lambda g, best, mean: print(f"generation {g + 1}: best {best:.3f}, mean {mean:.3f}", flush=True)
        weights, score, _ = train(os.path.abspath(a.scene), body, brain, a.population, a.generations, a.seconds, a.mutation, a.elite,
                                  a.workers, a.seed, a.judge, report)
        print(f"best {score:.3f}, {len(weights)} weights")
        if a.save:
            scene.ed.component_set(id=body, type="brain.net", values={**brain, "weights": weights_text(weights), "score": round(score, 4)})
            scene.ed.scene_save(file=os.path.abspath(a.save))
            print("saved", a.save)


if __name__ == "__main__":
    main()
