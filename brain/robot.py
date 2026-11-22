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


class Wiring:
    """inputs: (id, name, kind, spec) of each sensor; outputs: (id, name) of each force that acts all the time (a
    thrust - a push once is the world's, not the brain's); joints: the robot's own."""

    def __init__(self, scene, body):
        self.body = int(body)
        links = {}
        for j in scene.ed.scene_list(kind="joint")["objects"]:
            lines = _lines(scene, j["id"])
            links[j["id"]] = (int(_value(lines, "link", "a", 0)), int(_value(lines, "link", "b", 0)))
        bodies, self.joints = {self.body}, []
        grown = True
        while grown:  # the bodies joined to the robot's, and theirs, ...
            grown = False
            for jid, (a, b) in sorted(links.items()):
                if jid not in self.joints and (a in bodies or b in bodies):
                    self.joints.append(jid)
                    bodies |= {x for x in (a, b) if x}
                    grown = True
        self.bodies = sorted(bodies)
        holders = set(self.bodies) | set(self.joints)
        self.inputs, self.outputs = [], []
        for s in sorted(scene.ed.scene_list(kind="sensor")["objects"], key=lambda o: o["id"]):
            lines = _lines(scene, s["id"])
            kind = next((w[0] for w in lines if w and w[0] in SENSOR_KINDS), None)
            if int(_value(lines, "object", "parent", 0)) in holders and kind not in (None, "stereo"):
                self.inputs.append((s["id"], s["name"], kind, _spec(lines)))
        for f in sorted(scene.ed.scene_list(kind="force")["objects"], key=lambda o: o["id"]):
            lines = _lines(scene, f["id"])
            if int(_value(lines, "object", "parent", 0)) in set(self.bodies) and _value(lines, "effect", "mode") == "constant":
                self.outputs.append((f["id"], f["name"]))
        self.size = len(self.numbers({}))

    def numbers(self, readings):
        """The net's inputs from the sensors' readings ({id: reading} as sensor_read and a run's tick give them)."""
        out = []
        for id, _, kind, spec in self.inputs:
            r = readings.get(id) or {}
            if kind == "lidar":
                n, far = int(spec.get("rays_h", 1)) * int(spec.get("rays_v", 1)), float(spec.get("max_m", 12)) or 1.0
                ranges = list(r.get("ranges") or [])[:n]
                out += [d / far if d >= 0 else 1.0 for d in ranges] + [1.0] * (n - len(ranges))  # no return: as far as it sees
            elif kind == "sonar":
                d, far = r.get("distance"), float(spec.get("max_m", 4)) or 1.0
                out.append(d / far if d is not None and d >= 0 else 1.0)
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

    def commands(self, outputs, power):
        """robot_command's commands: each force's strength, its output times `power` newtons."""
        return {str(id): {"strength": round(o * float(power), 6)} for (id, _), o in zip(self.outputs, outputs)}

    def said(self):
        """The wiring in words: «5 входов: Лидар 1 (5 лучей); 2 выхода: Тяга 1, Тяга 2»."""
        ins = []
        for id, name, kind, spec in self.inputs:
            n = int(spec.get("rays_h", 1)) * int(spec.get("rays_v", 1)) if kind == "lidar" else {"imu": 6, "wrench": 6}.get(kind, 1)
            ins.append(f"{name} ({n})" if n > 1 else name)
        return f"{self.size} входов: {', '.join(ins) or 'нет'}; {len(self.outputs)} выходов: {', '.join(n for _, n in self.outputs) or 'нет'}"
