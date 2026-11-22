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
from robot import Wiring, _lines, _value, read_world


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


class Upright:
    """A pole kept up: the seconds its joint stays within `limit` rad of where it was built and the body within 2.5 m
    of its start, less a little for wandering. The run ends when either fails. Looked at every decision."""
    limit, every = 0.5, 0.0

    def __init__(self, scene, wiring):
        if not wiring.joints:
            raise ValueError("«upright» judges a joint of the robot: the robot has none")
        self.body, self.joint = wiring.body, wiring.joints[0]
        self.alive = self.wander = 0.0
        self.start = None

    def needs(self):
        return [self.body], [self.joint]

    def step(self, c, dt):
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


class Track:
    """A race: the metres along the track's middle line (its «brain.track», scenes.race) the robot made from where it
    began, laps counted on, back the other way taken off - the best of the run. The run ends when it has not gained
    half a metre for 3 s: stuck at a wall, or going round in circles - or when it is off the track, farther from the
    line than its walls. Looked at twice a second."""
    every = 0.5
    lines, widths = {}, {}  # a track's line and the way along it, and its width, by the scene: read once a world

    def __init__(self, scene, wiring):
        key = id(scene)
        if key not in Track.lines:
            rows = scene.ed.component_list(type="brain.track")["components"]
            if not rows:
                raise ValueError("«track» judges along a track: the scene has none («Пример: гонка» makes one)")
            flat = [float(v) for v in rows[0]["values"]["points"]]
            Track.widths[key] = float(rows[0]["values"].get("width", 1.8))
            line = list(zip(flat[0::2], flat[1::2]))
            along = [0.0]
            for k in range(len(line)):
                p, q = line[k], line[(k + 1) % len(line)]
                along.append(along[-1] + ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5)
            Track.lines[key] = (line, along)
        self.line, self.along = Track.lines[key]
        self.length, self.body, self.half = self.along[-1], wiring.body, Track.widths[key] / 2
        self.last, self.made, self.best, self.idle = None, 0.0, 0.0, 0.0

    def needs(self):
        return [self.body], []

    def where(self, x, z):
        """How far along the middle line the nearest point to (x, z) is, and how far from it, m."""
        best, at = float("inf"), 0.0
        for k in range(len(self.line)):
            (ax, az), (bx, bz) = self.line[k], self.line[(k + 1) % len(self.line)]
            dx, dz = bx - ax, bz - az
            t = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz or 1.0)))
            d = (ax + t * dx - x) ** 2 + (az + t * dz - z) ** 2
            if d < best:
                best, at = d, self.along[k] + t * (self.along[k + 1] - self.along[k])
        return at, best ** 0.5

    def step(self, c, dt):
        s, off = self.where(c.get(f"{self.body}/centre x", 0.0), c.get(f"{self.body}/centre z", 0.0))
        if off > self.half + 0.2:  # past a wall: off the track, the run is over
            return False
        if self.last is not None:
            self.made += (s - self.last + self.length / 2) % self.length - self.length / 2  # across the start line: on
        self.last = s
        if self.made > self.best + 0.5:
            self.best, self.idle = self.made, 0.0
        else:
            self.idle += dt
        return self.idle < 3.0

    def score(self):
        return max(self.best, self.made)


JUDGES = {"upright": Upright, "track": Track}
TAG = "brain"  # the robots a world trains side by side: each touches the world, not the others


class Arena:
    """One world with no window, `cars` robots in it side by side - copies of the scene's one, each touching the
    world and not the others (tagged and ignoring the tag), each sensing only what its body touches. A copy of a
    robot is of its body alone: one of joined bodies trains one a world."""

    def __init__(self, scene_file, body, judge, cars=1):
        self.scene = _kinetica().Scene()
        ed = self.scene.ed
        ed.scene_open(file=scene_file)
        self.judge = JUDGES[judge]
        bodies = [int(body)]
        if cars > 1:
            if Wiring(self.scene, body).joints:
                raise ValueError("a robot of joined bodies trains one a world: «cars» 1")
            ed.scene_tag(ids=[body], tag=TAG)
            ed.scene_set(id=body, line="ignores", values=[TAG])
            for id, _, kind, _ in Wiring(self.scene, body).inputs:  # every copy hears the noise the editor's one does:
                seed = int(_value(_lines(self.scene, id), kind, "seed", 0) or 0)  # its seed, or its own id, as a seed of 0 means
                if kind in ("imu", "sonar", "wrench", "lidar") and not seed:
                    ed.scene_set(id=id, line=kind, key="seed", values=[id])
            y = self._height(body)
            for _ in range(cars - 1):
                copy = ed.scene_duplicate(ids=[body])["ids"][0]
                ed.scene_move(ids=[copy], by=[0, y - self._height(copy), 0])  # where the original stands
                bodies.append(copy)
        world = read_world(self.scene)
        self.wirings = [Wiring(self.scene, b, world) for b in bodies]

    def _height(self, id):
        shape = next(w for w in self.scene.ed.scene_get(id=id)["lines"] if w and w[0] == "shape")
        return float(shape[shape.index("position") + 2])

    def episodes(self, nets, genomes, seconds, rate_hz, power):
        """Each robot run with its weights for `seconds`, all at once: the judges' scores. Built anew, then from the
        frame it starts from, every 1 / rate_hz: every sensor read, every net's outputs commanded, the world stepped -
        the order of the editor's run (a tick for the frame it starts from, then one after each frame, its commands
        before the next), so a brain trained here plays the same there. A robot the judge has ruled out stops."""
        ed = self.scene.ed
        ed.sim_stop()
        ed.sim_step(frames=0)
        n = len(genomes)
        judges = [self.judge(self.scene, self.wirings[i]) for i in range(n)]
        for net in nets[:n]:
            net.reset()
        every = max(1, round(60.0 / float(rate_hz)))
        look = max(every, round(self.judge.every * 60))  # frames between two looks of the judges
        alive, t, frame, since = list(range(n)), 0.0, 0, 0
        stopped = {}
        while t < seconds and alive:
            readings = {r["id"]: r for r in ed.sensor_read()["sensors"]}
            commands = dict(stopped)
            stopped.clear()
            for i in alive:
                commands.update(self.wirings[i].commands(nets[i].forward(genomes[i], self.wirings[i].numbers(readings)), power))
            if commands:
                ed.robot_command(commands=commands)
            ed.sim_step(frames=every, state=False)  # what the judges need they ask by sim_measure
            t, frame, since = t + every * FRAME, frame + every, since + every
            if since >= look:
                objects, joints = set(), set()
                for i in alive:
                    o, j = judges[i].needs()
                    objects.update(o), joints.update(j)
                c = {x["id"]: x["value"] for x in ed.sim_measure(objects=sorted(objects), joints=sorted(joints))["channels"]}
                still = []
                for i in alive:
                    if judges[i].step(c, since * FRAME):
                        still.append(i)
                    else:  # out: its thrusts off from the next frame on
                        stopped.update(self.wirings[i].commands([0.0] * len(self.wirings[i].outputs), power))
                alive, since = still, 0
        return [j.score() for j in judges]

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
          judge="upright", report=None, stop=None, first=None, cars=0):
    """The best weights found, their score, and each generation's (best, mean). `brain`: the component's values
    (layers, activation, recurrent, rate_hz, power). `report(generation, best, mean)` after each generation; `stop`, a
    threading.Event, ends it after the generation it is set in. `first`: weights to seed the population with. `cars`:
    robots a world - the population in worlds of that many, individual i always in world i // cars at its place
    i % cars (its sensors' noise its own), so the result is the seed's whatever the number of workers; 0: 20 for a robot
    of one body, 1 for one of joined bodies (its copies would lose their joints)."""
    workers = int(workers) or max(1, (os.cpu_count() or 2) - 1)
    population, elite = max(2, int(population)), max(0, int(elite))
    rng = new_rng(seed)
    local, arenas, lock = threading.local(), [], threading.Lock()
    with _probe_scene(scene_file) as world:
        probe = Wiring(world, body)
    cars = max(1, min(int(cars) or (1 if probe.joints else 20), population))
    if not probe.outputs:
        raise ValueError("the robot has no force of its own: a brain has nothing to drive")
    shape = (probe.size, hidden_of(brain["layers"]), len(probe.outputs), brain["activation"], brain["recurrent"])
    genomes = [Net(*shape).random_weights(rng) for _ in range(population)]
    if first and len(first) == Net(*shape).count():
        genomes[0] = weights_of(first)
    history, best, best_score = [], genomes[0], float("-inf")

    def world_scores(w):
        if not hasattr(local, "arena"):
            local.arena = Arena(scene_file, body, judge, cars)
            local.nets = [Net(*shape) for _ in range(cars)]
            with lock:
                arenas.append(local.arena)
        return local.arena.episodes(local.nets, genomes[w * cars:(w + 1) * cars], seconds, brain["rate_hz"], brain["power"])

    os.environ.setdefault("KE_PHYSICS_THREADS", "1")  # a world a core: no worker threads of its own
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for g in range(int(generations)):
                scores = [s for world in pool.map(world_scores, range((population + cars - 1) // cars)) for s in world]
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
            v = {"layers": "8", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 15.0, "weights": "", "judge": "upright", "seconds": 8.0}
            v.update(row["values"])
            v["layers"], v["recurrent"] = str(v["layers"]), v["recurrent"] in (True, 1, "1", "true")
            return row["id"], v
    raise ValueError("no body has a «Мозг»: put one on (brain_add)")


def main():
    p = argparse.ArgumentParser(description="Train the brain of a scene's robot by natural selection, with no window.")
    p.add_argument("scene")
    p.add_argument("--body", type=int, default=None, help="the body with the brain (default: the first with one)")
    for name, kind, default in (("population", int, 100), ("generations", int, 30), ("mutation", float, 0.2),
                                ("elite", int, 2), ("workers", int, 0), ("seed", int, 1)):
        p.add_argument("--" + name, type=kind, default=default)
    p.add_argument("--seconds", type=float, help="a run's length (default: the brain's)")
    p.add_argument("--judge", choices=sorted(JUDGES), help="what scores a run (default: the brain's)")
    p.add_argument("--cars", type=int, default=0, help="robots a world, side by side (0: 20 for one body, 1 for joined bodies)")
    p.add_argument("--save", help="a copy of the scene with the best weights in its brain")
    a = p.parse_args()
    with _probe_scene(os.path.abspath(a.scene)) as scene:
        body, brain = brain_of(scene, a.body)
        report = lambda g, best, mean: print(f"generation {g + 1}: best {best:.3f}, mean {mean:.3f}", flush=True)
        weights, score, _ = train(os.path.abspath(a.scene), body, brain, a.population, a.generations, a.seconds or float(brain["seconds"]),
                                  a.mutation, a.elite, a.workers, a.seed, a.judge or brain["judge"], report, cars=a.cars)
        print(f"best {score:.3f}, {len(weights)} weights")
        if a.save:
            scene.ed.component_set(id=body, type="brain.net", values={**brain, "weights": weights_text(weights), "score": round(score, 4)})
            scene.ed.scene_save(file=os.path.abspath(a.save))
            print("saved", a.save)


if __name__ == "__main__":
    main()
