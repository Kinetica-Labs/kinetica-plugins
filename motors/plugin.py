"""«Моторы»: a brushed DC motor's constant and resistance from how it is made, not from a datasheet's two numbers.

The physics, for a winding turning in a permanent magnet's field (radial, B in the air gap):
- Lorentz: a conductor of length L carrying i across B feels F = i L B; at the rotor's radius r it turns it by i L B r.
  2N active sides (N turns, two sides each) in `paths` parallel paths carry i = I / paths each; the share `cover` of
  them lies under a pole at any moment. Torque τ = cover · 2N · B L r · I / paths = K I.
- Faraday (Maxwell's ∮E·dl = -dΦ/dt): a side moving at v = ω r across B gets e = B L ω r; the sides of one path in
  series add, so the back-EMF E = cover · 2N · B L r · ω / paths = K ω - the same K: the power K I ω leaving the
  winding as EMF is the shaft's work, energy kept.
- Ohm: copper of resistivity ρ, a turn of mean length 2 L + 2 π r (two sides and two end turns over half the rotor),
  N turns over `paths` paths in parallel: R = ρ N (2 L + 2 π r) / (S paths²), S the wire's section π d² / 4.
- A gearbox of ratio G: the wheel sees K G (torque times G, speed over G); R unchanged; its losses left out.

Left out (said, not hidden): the field taken uniform in the gap (no finite-element solution of the magnet - B is the
magnet's datasheet number at its gap); copper at 20 °C (its resistance grows 0.4 %/K as it heats); no brush drop, no
iron losses, no inductance (L/R is a fraction of a millisecond, under a frame); the gearbox lossless. The running
world then solves the motor with its driver and battery each step (the editor's src/sim/RobotPower.cpp).

While the scene runs, each battery's charge and power stand beside the robot on a leader line to it, laid out with the
scene's other callouts (place aside), 4 times a second: «Батарея 1: 98,7 %» and «P = U·I = 12,43 В × 5,3 А = 66 Вт» -
the model's numbers (sim_measure), for a person: no controller reads them."""
import math
import threading

from kinetica.plugin import command, field, on_play, on_stop, tr

RHO_COPPER = 1.72e-8  # Ω·m at 20 °C


def design(gap, length, radius, turns, wire, paths=2, cover=0.7, gear=1.0):
    """K (N·m/A = V·s/rad, at the gearbox's output) and R (Ω) of a motor: gap field T, rotor length and radius m, turns,
    wire diameter m, parallel paths, the share of sides under a pole, the gear ratio; with the steps between."""
    k_motor = cover * 2 * turns * gap * length * radius / paths
    turn = 2 * length + 2 * math.pi * radius
    section = math.pi * wire * wire / 4
    resistance = RHO_COPPER * turns * turn / (section * paths * paths)
    return {"K": k_motor * gear, "K_motor": k_motor, "R": resistance, "turn_m": turn, "wire_m": turns * turn, "section_mm2": section * 1e6}


@command(label="Мотор по устройству")
def make(scene, joint=field(ref="joint", label="Сустав"), gap: float = field(0.45, unit="Тл", range=(0.05, 1.5), label="Поле в зазоре"),
         length: float = field(0.035, unit="м", range=(0.005, 0.3), label="Длина ротора"),
         radius: float = field(0.012, unit="м", range=(0.002, 0.2), label="Радиус ротора"),
         turns: int = field(60, range=(1, 5000), label="Витков"), wire: float = field(0.6, unit="мм", range=(0.05, 5.0), label="Провод"),
         paths: int = field(2, range=(1, 16), label="Параллельных ветвей"), cover: float = field(0.7, range=(0.1, 1.0), label="Доля под полюсом"),
         gear: float = field(36.0, range=(1.0, 1000.0), label="Передача"), voltage: float = field(11.1, unit="В", range=(-60.0, 60.0), label="Напряжение")):
    """The joint's main freedom gets a DC motor of this make: K and R by Lorentz, Faraday and Ohm (the file's text)."""
    d = design(gap, length, radius, turns, wire / 1000.0, paths, cover, gear)
    scene.ed.assembly_drive(joint=int(getattr(joint, "id", joint)), kind="dcmotor", constant=round(d["K"], 6), resistance=round(d["R"], 6),
                            voltage=float(voltage))
    stall = abs(voltage) / d["R"]
    return {**{k: round(v, 6) for k, v in d.items()}, "free_rad_s": round(abs(voltage) / d["K"], 3), "stall_A": round(stall, 2),
            "stall_N_m": round(d["K"] * stall, 3), "text": tr("K = {k:.4f} Н·м/А, R = {r:.4f} Ом").format(k=d["K"], r=d["R"])}


_books = {"stop": threading.Event()}


def _words(scene, id):
    return [line.split() for line in scene.ed.scene_get(id=id)["text"].splitlines()]


def _carrier(scene, joints):
    """The body the words hang over: a controller's target (a car's chassis), else the second link of a motor's joint."""
    for c in scene.ed.scene_list(kind="controller")["objects"]:
        for w in _words(scene, c["id"]):
            if w[:1] == ["action"] and "target" in w and int(w[w.index("target") + 1]):
                return int(w[w.index("target") + 1])
    for j in joints:
        link = next((w for w in _words(scene, j) if w[:1] == ["link"]), [])
        if "b" in link:
            return int(link[link.index("b") + 1])
    return 0


def _n(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def books_text(c, packs):
    """The words over the robot from sim_measure's channels {id: value}: each pack's charge, and its power as P = U·I."""
    lines = []
    for p in packs:
        v, i = c.get(f"power/{p} voltage", 0.0), c.get(f"power/{p} current", 0.0)
        lines.append(tr("{p}: {q} %").format(p=p, q=_n(c.get(f"power/{p} charge", 0.0), 1)))
        lines.append(tr("P = U·I = {v} В × {i} А = {w} Вт").format(v=_n(v, 2), i=_n(i), w=_n(v * i)))
    return "\n".join(lines)


def _show(scene, stop):
    packs = [b["name"] for b in scene.ed.scene_list(kind="battery")["objects"]]
    drivers = scene.ed.scene_list(kind="driver")["objects"]
    joints = [int(w[w.index("joint") + 1]) for d in drivers for w in _words(scene, d["id"]) if w[:1] == ["bridge"] and "joint" in w]
    body = _carrier(scene, joints)
    while not stop.wait(0.25):
        try:
            c = {x["id"]: x["value"] for x in scene.ed.sim_measure(scene=True)["channels"]}
            scene.ed.view_draw(key="motors/books", text=books_text(c, packs), at=[0, 0.35, 0], space="label", size=15,
                               **({"object": body, "leader": {"point": [0, 0, 0], "end": "dot", "place": "aside"}} if body else {}))
        except Exception as e:  # the run stopping under it ends it; anything else is said
            if stop.is_set():
                break
            print("motors:", e)


@on_play
def begin(scene):
    """▶: the books over the robot, when the scene has batteries or drivers."""
    _books["stop"].set()
    if scene.ed.scene_list(kind="battery")["objects"] or scene.ed.scene_list(kind="driver")["objects"]:
        _books["stop"] = threading.Event()
        threading.Thread(target=_show, args=(scene, _books["stop"]), daemon=True).start()


@on_stop
def end(scene):
    _books["stop"].set()
    scene.ed.view_draw(key="motors/books", clear=True)
