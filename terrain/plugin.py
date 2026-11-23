"""«Рельеф» (terrain): ground to drive on in one call - a static mesh of hills, its collider its own triangles (hollows
kept), flat about the middle; with `track`, a testing ground on it for a low car: a hump of 8° slopes, speed bumps of
3 cm, light crates (a pyramid of them) and balls.

The hills are smooth bumps placed by `seed` (the same seed, the same bytes), eased in from the flat middle, never below
the world's floor (y = 0), a hill's width at least 3.5 times its height, the highest hill where two meet (slopes under
12°); the mesh is written to the
user's data folder (terrain/) and the scene refers to it there. Run again, it replaces its own objects (by their names)."""
import math
import os
import random

from kinetica.plugin import command, field, tr

MINE = ("Рельеф", "Горка", "Лежачий полицейский", "Ящик", "Мяч")


def data_folder():
    """The user's data folder of Kinetica, as the editor's own: %APPDATA%, ~/Library/Application Support or XDG."""
    if os.environ.get("APPDATA"):
        return os.path.join(os.environ["APPDATA"], "Kinetica", "Kinetica")
    mac = os.path.expanduser("~/Library/Application Support")
    base = mac if os.path.isdir(mac) else os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "Kinetica", "Kinetica")


def hills_of(seed, count, side, height, flat):
    """Each hill (x, z, height, width) placed by `seed` outside the flat middle."""
    rnd = random.Random(seed)
    out = []
    for _ in range(1000):  # a small ground may hold fewer
        if len(out) == count:
            break
        x, z = rnd.uniform(-side / 2.5, side / 2.5), rnd.uniform(-side / 2.5, side / 2.5)
        h = height * rnd.uniform(0.5, 1.0)
        w = max(3.5 * h, rnd.uniform(2.0, 3.5))  # a bump's steepest slope is 0.6 h / w: under 0.17, 10°
        if math.hypot(x, z) > flat + 2.5 * w:  # only its foot reaches the flat middle
            out.append((x, z, h, w))
    return out


def height_at(x, z, hills, flat):
    """The highest of the hills there (not their sum: two overlapping are no steeper than one) and a low swell eased in
    from the flat middle."""
    t = min(1.0, max(0.0, (math.hypot(x, z) - flat) / 4.0))
    swell = t * t * (3 - 2 * t) * 0.05 * (1 + math.sin(0.45 * x) * math.cos(0.38 * z))
    return max([h * math.exp(-((x - hx) ** 2 + (z - hz) ** 2) / (2 * w * w)) for hx, hz, h, w in hills] + [0.0]) + swell


def write_obj(path, side, step, hills, flat):
    """The ground as an OBJ grid (two triangles a cell, facing up); its lowest and highest points."""
    n = int(round(side / step))
    rows, low, high = [], float("inf"), float("-inf")
    for i in range(n + 1):
        for k in range(n + 1):
            x, z = -side / 2 + i * step, -side / 2 + k * step
            y = height_at(x, z, hills, flat)
            low, high = min(low, y), max(high, y)
            rows.append(f"v {x:.3f} {y:.4f} {z:.3f}")
    at = lambda i, k: i * (n + 1) + k + 1
    for i in range(n):
        for k in range(n):
            rows.append(f"f {at(i, k)} {at(i, k + 1)} {at(i + 1, k + 1)}")
            rows.append(f"f {at(i, k)} {at(i + 1, k + 1)} {at(i + 1, k)}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Kinetica: the terrain plugin\n" + "\n".join(rows) + "\n")
    return low, high


def testing_ground(ed):
    """A testing ground for a low car (5 cm under its chassis, 0.7 m between its axles): a hump, bumps, crates, balls."""
    tilt, long, thick = 8.0, 1.2, 0.06
    cos, sin = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    ops = []
    for sign, foot in ((1, 2.5), (-1, 2.5 + 2 * long * cos + 0.8)):  # up from x 2.5, a 0.8 m top, down
        x, y = foot + sign * long / 2 * cos + sign * thick / 2 * sin, long / 2 * sin - thick / 2 * cos
        ops.append(("scene_add", {"shape": "box", "name": "Горка", "position": [x, y, 0], "size": [long, thick, 2.0], "role": "collider"}, sign * tilt))
    ops.append(("scene_add", {"shape": "box", "name": "Горка", "position": [2.5 + long * cos + 0.4, long * sin - thick / 2, 0], "size": [0.8, thick, 2.0], "role": "collider"}, 0))
    for x in (-2.2, -2.8, -3.4):
        ops.append(("scene_add", {"shape": "cylinder", "name": "Лежачий полицейский", "position": [x, 0.0, 0], "size": [0.06, 1.6, 0.06], "role": "collider"}, "x"))
    crates = [(1.5, 0.12, 1.6), (1.8, 0.12, -1.4), (-1.2, 0.12, 2.0), (0.4, 0.12, -2.2), (-1.8, 0.12, -1.6)]
    crates += [(7.5, 0.11 + row * 0.2, (k - (n - 1) / 2) * 0.22) for row, n in enumerate((3, 2, 1)) for k in range(n)]
    made = {"crates": 0, "balls": 0}
    for name, args, turn in ops:
        id = ed.scene_add(**args)["id"]
        if turn == "x":
            ed.scene_turn(ids=[id], axis="x", degrees=90)
        elif turn:
            ed.scene_turn(ids=[id], axis="z", degrees=turn)
    for p in crates:
        id = ed.scene_add(shape="box", name="Ящик", position=list(p), size=[0.2, 0.2, 0.2], role="rigid")["id"]
        ed.scene_set(id=id, line="rigid", key="density", values=[150])  # 1.2 kg: a car pushes it
        made["crates"] += 1
    for x, z in ((2.5, 2.5), (-2.5, 2.8), (0.0, 3.5)):
        id = ed.scene_add(shape="sphere", name="Мяч", position=[x, 0.3, z], size=[0.3, 0.3, 0.3], role="rigid")["id"]
        ed.scene_set(id=id, line="rigid", key="density", values=[60])
        made["balls"] += 1
    return made


@command(label="Рельеф")
def make(scene, size: float = field(30.0, unit="м", range=(4.0, 200.0), label="Сторона"),
         height: float = field(1.0, unit="м", range=(0.0, 10.0), label="Высота холмов"),
         hills: int = field(6, range=(0, 60), label="Холмов"), flat: float = field(3.0, unit="м", range=(0.0, 50.0), label="Ровно вокруг центра"),
         seed: int = field(1, range=(0, 1_000_000), label="Зерно"), track: bool = field(False, label="Полигон: горка, полицейские, ящики, мячи")):
    """Hilly static ground, flat about the middle - and, with track, a testing ground on it - in one call."""
    ed = scene.ed
    if ed.sim_state().get("mode") != "edit":
        ed.sim_stop()
    old = [o["id"] for o in ed.scene_list()["objects"] if o["name"] in MINE]
    if old:
        ed.scene_delete(ids=old)
    step = max(0.2, size / 100)  # at most a hundred cells a side
    placed = hills_of(seed, hills, size, height, flat)
    path = os.path.join(data_folder(), "terrain", f"terrain-{size:g}-{height:g}-{hills}-{flat:g}-{seed}.obj")
    low, high = write_obj(path, size, step, placed, flat)
    # A mesh is fitted to its size's largest edge about its box's centre: the centre put back where the grid has it.
    ground = ed.scene_put(block=f"""entity "Рельеф"
  shape mesh file "{path}" size {size} {max(high - low, 1e-3):.4f} {size} position 0 {(low + high) / 2:.4f} 0 rotation 0 0 0 color 0.46 0.56 0.36
  collider kind mesh
end""")["id"]
    out = {"id": ground, "file": path, "highest_m": round(high, 3)}
    if track:
        out.update(testing_ground(ed))
    return out
