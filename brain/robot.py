"""The robot as its brain sees it (plugin.py runs the brain in play, evolution.py trains it): the body the brain is on
and all that hangs on it through joints; the sensors on any of them, in the order of their ids - the inputs; the
forces of those bodies - the outputs. What a sensor reads becomes numbers from -1 to 1 for the net, an output from -1
to 1 a force's strength in newtons. The readings are the sensors' own (noise, counts, delays): the truth of the
physics never reaches a brain. Only the standard library; the scene is a kinetica.Scene - the editor's or one with
no window."""
import math

SENSOR_KINDS = ("encoder", "imu", "sonar", "touch", "wrench", "effort", "stereo", "lidar")


def _lines(scene, id):
    return scene.ed.scene_get(id=id)["lines"]


def _value(lines, line, key, default=None):
    for words in lines:
        if words and words[0] == line and key in words[1:]:
            return words[words.index(key, 1) + 1]
    return default


def _spec(lines):
    for words in lines:
        if words and words[0] == "spec":
            return {words[k]: words[k + 1] for k in range(1, len(words) - 1, 2)}
    return {}


def read_world(scene):
    """What the wirings are made of, read once: each joint's two bodies, each sensor's holder, kind and datasheet, each
    force's body and mode. Many robots in one world (evolution.Arena) are wired from one reading."""
    world = {"links": {}, "sensors": [], "forces": []}
    for j in scene.ed.scene_list(kind="joint")["objects"]:
        lines = _lines(scene, j["id"])
        world["links"][j["id"]] = (int(_value(lines, "link", "a", 0)), int(_value(lines, "link", "b", 0)))
    for s in sorted(scene.ed.scene_list(kind="sensor")["objects"], key=lambda o: o["id"]):
        lines = _lines(scene, s["id"])
        kind = next((w[0] for w in lines if w and w[0] in SENSOR_KINDS), None)
        world["sensors"].append((s["id"], s["name"], kind, _spec(lines), int(_value(lines, "object", "parent", 0))))
    for f in sorted(scene.ed.scene_list(kind="force")["objects"], key=lambda o: o["id"]):
        lines = _lines(scene, f["id"])
        world["forces"].append((f["id"], f["name"], int(_value(lines, "object", "parent", 0)), _value(lines, "effect", "mode")))
    return world


class Wiring:
    """inputs: (id, name, kind, spec) of each sensor; outputs: (id, name) of each force that acts all the time (a
    thrust - a push once is the world's, not the brain's); joints: the robot's own. A car (drive "car", drive.py) has
    two outputs - the planner's: the turn it wants and the speed - which its drive holds, the turn by its IMU's
    gyroscope (`gyro`), and its net hears them back the step after (network.Net's echo: the speed it asked for, which
    no sensor of a car without wheels reads); `count` - the net's outputs."""

    def __init__(self, scene, body, world=None, drive="thrusts"):
        world = world or read_world(scene)
        self.body = int(body)
        bodies, self.joints = {self.body}, []
        grown = True
        while grown:  # the bodies joined to the robot's, and theirs, ...
            grown = False
            for jid, (a, b) in sorted(world["links"].items()):
                if jid not in self.joints and (a in bodies or b in bodies):
                    self.joints.append(jid)
                    bodies |= {x for x in (a, b) if x}
                    grown = True
        self.bodies = sorted(bodies)
        holders = set(self.bodies) | set(self.joints)
        self.inputs = [(id, name, kind, spec) for id, name, kind, spec, parent in world["sensors"] if parent in holders and kind not in (None, "stereo")]
        self.outputs = [(id, name) for id, name, parent, mode in world["forces"] if parent in set(self.bodies) and mode == "constant"]
        self.size = len(self.numbers({}))
        self.car = drive == "car"
        self.gyro = next((id for id, _, kind, _ in self.inputs if kind == "imu"), None)
        self.count = 2 if self.car else len(self.outputs)
        self.sides = None  # the first fan of rays: where its right half and its left half are among the numbers
        at = 0
        for _, _, kind, spec in self.inputs:
            n = int(spec.get("rays_h", 1)) * int(spec.get("rays_v", 1)) if kind == "lidar" else {"imu": 6, "wrench": 6}.get(kind, 1)
            if kind == "lidar" and int(spec.get("rays_v", 1)) == 1 and n > 1 and self.sides is None:
                self.sides = (at, at + n // 2, at + n - n // 2, at + n)  # rays from its right to its left
            at += n
        self.footprint = None
        if self.car and scene is not None:
            shape = next(w for w in scene.ed.scene_get(id=self.body)["lines"] if w and w[0] == "shape")
            self.footprint = [float(shape[shape.index("size") + 1 + k]) for k in range(3)]

    def numbers(self, readings):
        """The net's inputs from the sensors' readings ({id: reading} as sensor_read and a run's tick give them). A ray's
    distance comes as nearness: 0 where it runs free, 1 at a wall touching the sensor - the empty way a neutral zero,
    the danger the signal."""
        out = []
        for id, _, kind, spec in self.inputs:
            r = readings.get(id) or {}
            if kind == "lidar":
                n, far = int(spec.get("rays_h", 1)) * int(spec.get("rays_v", 1)), float(spec.get("max_m", 12)) or 1.0
                ranges = list(r.get("ranges") or [])[:n]
                out += [1.0 - d / far if d >= 0 else 0.0 for d in ranges] + [0.0] * (n - len(ranges))  # no return: nothing near
            elif kind == "sonar":
                d, far = r.get("distance"), float(spec.get("max_m", 4)) or 1.0
                out.append(1.0 - d / far if d is not None and d >= 0 else 0.0)
            elif kind == "encoder":
                v = r.get("reading")
                out.append(math.tanh(2.0 * (r.get("relative", 0.0) if v is None else v)))  # a joint's useful tilt is a fraction of a radian
            elif kind == "imu":
                out += [math.tanh(a / 9.81) for a in r.get("accel", [0.0] * 3)] + [math.tanh(w / 2.0) for w in r.get("gyro", [0.0] * 3)]
            elif kind == "wrench":
                out += [math.tanh(f / 50.0) for f in r.get("force", [0.0] * 3)] + [math.tanh(t / 10.0) for t in r.get("torque", [0.0] * 3)]
            elif kind == "touch":
                out.append(math.tanh(r.get("force", 0.0) / 10.0))
            elif kind == "effort":
                out.append(math.tanh(r.get("effort", 0.0) / 10.0))
        return out

    def away(self, numbers):
        """How much nearer the walls on the right are than those on the left, -1..1 (its rays' mean nearness): the
        turn away from them, a reflex's (drive.plan) - 0 with no fan of rays."""
        if self.sides is None:
            return 0.0
        r0, r1, l0, l1 = self.sides
        return sum(numbers[r0:r1]) / (r1 - r0) - sum(numbers[l0:l1]) / (l1 - l0)

    def commands(self, outputs, power):
        """robot_command's commands: each force's strength, its output times `power` newtons."""
        return {str(id): {"strength": round(o * float(power), 6)} for (id, _), o in zip(self.outputs, outputs)}

    def said(self):
        """The wiring in words: «5 входов: Лидар 1 (5 лучей); 2 выхода: Тяга 1, Тяга 2»."""
        ins = []
        for id, name, kind, spec in self.inputs:
            n = int(spec.get("rays_h", 1)) * int(spec.get("rays_v", 1)) if kind == "lidar" else {"imu": 6, "wrench": 6}.get(kind, 1)
            ins.append(f"{name} ({n})" if n > 1 else name)
        outs = "поворот и скорость (привод машины)" if self.car else ", ".join(n for _, n in self.outputs) or "нет"
        return f"{self.size} входов: {', '.join(ins) or 'нет'}; {self.count} выходов: {outs}"
