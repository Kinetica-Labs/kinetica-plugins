"""The examples of the brain (plugin.py offers each as a button; the tests build them with no window): made by the
editor's operations, the same in the window and in a kinetica-core world."""
import hashlib
import math
import os
import random

import drive

# The race's track: its middle line through these points (x, z, metres), closed - a spline drawn through them. Its
# tightest turn is 2.4 m round: a car at 4 m/s takes it with 6.7 m/s^2 of its tyres' 9 (drive.GRIP) - a turn tighter
# than speed^2 / grip no car makes at that speed, and every one dies there (the first circuit had one of 1.34 m).
CIRCUIT = [(x * 1.3, z * 1.3) for x, z in ((0, 0), (6, -1.5), (11, 0), (13.5, 4), (11, 8), (6, 7.5), (2.5, 10), (-3, 11), (-8, 9), (-10, 4), (-6.5, -0.5))]


def balance(scene):
    """A cart and a metre-long pole on a free hinge: the brain keeps the pole up by the cart's thrust along x. It feels
    the pole's tilt by the hinge's encoder and how fast it falls by an IMU's gyroscope on its top (a balancing robot's
    way), and where the cart is by a lidar's one beam to the wall behind; a push on the pole at the start tips it.
    Returns the cart's id - the body the brain goes on."""
    ed = scene.ed
    cart = ed.scene_add(shape="box", name="Тележка", position=[0, 0.1, 0], size=[0.4, 0.2, 0.3], role="rigid")["id"]
    ed.scene_set(id=cart, line="rigid", key="mass", values=[2])
    ed.scene_set(id=cart, line="rigid", key="friction", values=[0.05])
    pole = ed.scene_add(shape="box", name="Шест", position=[0, 0.7, 0], size=[0.04, 1.0, 0.04], role="rigid")["id"]
    ed.scene_set(id=pole, line="rigid", key="mass", values=[0.3])
    joint = ed.assembly_joint(a=cart, b=pole, type="hinge", anchor=[0, 0.2, 0], axis=[0, 0, 1])["id"]
    ed.assembly_encoder(joint=joint, lines=1000, reference=0)
    ed.assembly_sensor(target=pole, kind="imu", face="+y")
    ed.scene_add(shape="box", name="Стена", position=[-2.5, 0.5, 0], size=[0.2, 1, 3], role="collider")
    ed.assembly_sensor(target=cart, kind="lidar", face="-x", model="TFmini-S")
    thrust = ed.scene_add_force(shape="directional", mode="constant", body=cart, strength=0)["id"]
    ed.scene_turn(ids=[thrust], axis=[0, 0, 1], degrees=-90)  # its +Y along the world's +x
    push = ed.scene_add_force(shape="directional", mode="impulse", body=pole, strength=0.06, at=[0, 1.1, 0])["id"]
    ed.scene_turn(ids=[push], axis=[0, 0, 1], degrees=-90)
    return cart


def spline(points, step=0.8):
    """A closed Catmull-Rom spline through the points: (x, z) about every `step` metres along it."""
    out, n = [], len(points)
    for k in range(n):
        p0, p1, p2, p3 = (points[(k + j) % n] for j in (-1, 0, 1, 2))
        pieces = max(1, math.ceil(math.dist(p1, p2) / step))
        for i in range(pieces):
            t = i / pieces
            out.append(tuple(0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t ** 3)
                             for a, b, c, d in zip(p0, p1, p2, p3)))
    return out


def _heading(a, b):
    """The turn about y that lays a shape's x from a to b, degrees."""
    return math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))


def walls(line, width, thickness=0.12, height=0.6):
    """The track's two walls as one model: each a closed band along the line - its face toward the track `width` / 2
    from the middle, `thickness` thick, `height` high (a car hitting a lower one at 13 m/s is thrown over it) - in triangles counter-clockwise from outside. (vertices, triangles)."""
    n, vertices, triangles = len(line), [], []
    for side in (1, -1):
        base = len(vertices)
        for k in range(n):
            a, b, c = line[k - 1], line[k], line[(k + 1) % n]
            tx, tz = c[0] - a[0], c[1] - a[1]
            length = math.hypot(tx, tz) or 1.0
            left = (-tz / length, tx / length)
            for d in (width / 2, width / 2 + thickness):  # the face toward the track, then the outer one
                x, z = b[0] + side * d * left[0], b[1] + side * d * left[1]
                vertices += [(x, 0.0, z), (x, height, z)]
        ring = 4  # a section: inner bottom, inner top, outer bottom, outer top
        for k in range(n):
            i, j = base + k * ring, base + ((k + 1) % n) * ring
            for p, q in ((0, 1), (1, 3), (3, 2), (2, 0)):  # the band's four faces between two sections
                triangles += [(i + p, j + p, j + q), (i + p, j + q, i + q)]
        mid = [((vertices[base + k * ring][0] + vertices[base + k * ring + 2][0]) / 2, height / 2,
                (vertices[base + k * ring][2] + vertices[base + k * ring + 2][2]) / 2) for k in range(n)]
        for t in range(len(triangles) - 8 * n, len(triangles)):  # each face turned out of the band
            v = [vertices[i] for i in triangles[t]]
            e1 = [v[1][x] - v[0][x] for x in range(3)]
            e2 = [v[2][x] - v[0][x] for x in range(3)]
            normal = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
            k = (triangles[t][0] - base) // ring
            centre = [sum(p[x] for p in v) / 3 - mid[k][x] for x in range(3)]
            if sum(normal[x] * centre[x] for x in range(3)) < 0:
                triangles[t] = (triangles[t][0], triangles[t][2], triangles[t][1])
    return vertices, triangles


def _model(vertices, triangles, folder=None):
    """The model's file: OBJ in the plugin's own folder of tracks (or `folder`), named by what it holds (the same
    track, the same file)."""
    text = "".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in vertices) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in triangles)
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracks")
    os.makedirs(folder, exist_ok=True)
    file = os.path.join(folder, "track-" + hashlib.sha1(text.encode()).hexdigest()[:12] + ".obj")
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as out:
            out.write(text)
    return file


def closest_return(line, width):
    """How near the line comes back to itself: the least distance between two of its points farther apart along it
    than 3 widths, m."""
    n, along = len(line), [0.0]
    for k in range(n):
        along.append(along[-1] + math.dist(line[k], line[(k + 1) % n]))
    lap, best = along[-1], float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            gap = along[j] - along[i]
            if min(gap, lap - gap) > 3 * width:
                best = min(best, math.dist(line[i], line[j]))
    return best


# A random track's kinds, by the training's turn: its control points (the more, the more bends) and how far each strays
# from the round (the more, the sharper - hairpins and S-bends).
KINDS = ((8, 0.2), (12, 0.35), (16, 0.45), (20, 0.4))


def random_track(seed, turns=12, wobble=0.35, width=1.8, slowest=1.0, tries=2000):
    """A drivable closed track drawn from `seed`: `turns` control points round a centre, each at its own angle and
    distance (the radius times 1 +- `wobble`), the spline through them - kept when its tightest turn is wide enough for
    the inner wall and for the slowest speed on the tyres' grip, and it never comes back near itself (no wall through
    another); else the next draw. The control points (race's `points`)."""
    rng, radius = random.Random(seed), 4.0 + 0.9 * turns  # more turns, more room for them
    for _ in range(tries):
        angles = sorted(2 * math.pi * (k + rng.uniform(-0.3, 0.3)) / turns for k in range(turns))
        points = [(radius * (1 + rng.uniform(-wobble, wobble)) * math.cos(a), radius * (1 + rng.uniform(-wobble, wobble)) * math.sin(a))
                  for a in angles]
        line = spline(points)
        if tightest(line) > max(width / 2 + 0.3, slowest ** 2 / drive.GRIP) and closest_return(line, width) > 2 * width + 0.5:
            return points
    raise ValueError(f"no drivable track of {turns} turns in {tries} draws")


def tightest(line):
    """The tightest turn of a closed line, its radius in metres (through each point and its two neighbours)."""
    n, out = len(line), float("inf")
    for k in range(n):
        a, b, c = line[k - 1], line[k], line[(k + 1) % n]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 > 1e-9:
            out = min(out, math.dist(a, b) * math.dist(b, c) * math.dist(c, a) / (2 * area2))
    return out


def race(scene, points=CIRCUIT, width=1.8, folder=None, joints=False):
    """A closed track - its middle line a spline through `points`, two walls `width` apart along it, one static body of
    triangles («Трасса», a mesh collider), the line kept in its «brain.track» (what the judge measures the way along) -
    and a car at its start: a block with a lidar on its roof - a fan of 21 rays over 200 degrees ahead - and an IMU - its
    brain steers it and its drive (drive.py) does the rest: tyres, motor, steering by the IMU's gyroscope; with `joints`
    the sports car of joints instead (joint_car). Returns the car's (its chassis') id."""
    ed = scene.ed
    line = spline(points)
    vertices, triangles = walls(line, width)
    lo = [min(v[x] for v in vertices) for x in range(3)]
    hi = [max(v[x] for v in vertices) for x in range(3)]
    middle, extent = [(lo[x] + hi[x]) / 2 for x in range(3)], [hi[x] - lo[x] for x in range(3)]
    floor = next(o["id"] for o in ed.scene_list(kind="entity")["objects"] if o["name"] == "Пол")
    ed.scene_set(id=floor, line="shape", key="size", values=[extent[0] + 6, 0.02, extent[2] + 6])
    ed.scene_set(id=floor, line="shape", key="position", values=[middle[0], 0.01, middle[2]])
    ed.world_set(line="world", key="size", values=[extent[0] + 6, 3, extent[2] + 6])
    ed.world_set(line="world", key="duration", values=[0])
    # blocks sliding on a floor: 4 a frame are plenty, 32 a waste; a car of joints - grams of knuckle on kilograms of chassis -
    # needs the 32 its joints are solved in (with 4 its chain is torn apart and it tumbles)
    ed.world_set(line="physics", key="substeps", values=[32 if joints else 4])
    word = lambda v: " ".join(f"{x:.4f}" for x in v)
    track = ed.scene_put(block=f'entity "Трасса"\n  object id 0 visible 1 locked 1\n'
                               f'  shape mesh file "{_model(vertices, triangles, folder)}" size {word(extent)} position {word(middle)} rotation 0 0 0 color 0.8 0.35 0.25\n'
                               f'  collider kind mesh fit 1 size 0.2 0.2 0.2 offset 0 0 0 rotation 0 0 0\nend')["id"]
    ed.component_set(id=track, type="brain.track", values={"points": [round(v, 3) for p in line for v in p], "width": width})
    start, ahead = line[0], line[1]
    if joints:  # the sports car of joints instead of the block: the same senses on its roof, its own wheels
        return joint_car(scene, start, _heading(start, ahead))
    car = ed.scene_add(shape="box", name="Машинка", position=[start[0], 0.1, start[1]], size=[0.5, 0.2, 0.3], role="rigid")["id"]
    ed.scene_set(id=car, line="rigid", key="mass", values=[1])
    ed.scene_set(id=car, line="rigid", key="friction", values=[0.2])
    lidar = ed.assembly_sensor(target=car, kind="lidar", face="+x", model="TFmini-S")["id"]
    ed.sensor_spec(sensor=lidar, spec={"rays_h": 21, "fov_h_deg": 200, "max_m": 8, "rate_hz": 30})  # 200 degrees: the walls beside it too; a ray each 10 degrees
    # on the roof's middle, as robots carry one: its 10 cm it cannot see (min_m) stay inside the car - on the nose they
    # were the road ahead, and a car at a wall read it as free
    ed.scene_set(id=lidar, line="pose", key="position", values=[0, 0.1, 0])
    ed.assembly_sensor(target=car, kind="imu", face="+y")
    ed.scene_turn(ids=[car], axis="y", degrees=_heading(start, ahead))
    return car


def joint_car(scene, at=(0.0, 0.0), heading=0.0):
    """A sports car of joints at `at` (x, z) facing `heading` degrees about the up (0: +x), for a brain: a low chassis, a
    cabin; each corner a hub on a shock absorber (the springs share the chassis), the front wheels on knuckles that a
    servo turns (robot.JointCar's steering), the rear ones on velocity motors (its drive) - and the race car's senses on
    the cabin's roof: the lidar of 21 rays over 200 degrees, then the IMU flat (the order the trained brain reads them in,
    27 numbers). Returns the chassis' id."""
    ed = scene.ed
    x0, z0, y = float(at[0]), float(at[1]), 0.3
    made = []

    def box(name, p, size):
        id = ed.scene_add(shape="box", name=name, position=[x0 + p[0], y + p[1], z0 + p[2]], size=list(size), role="rigid")["id"]
        made.append(id)
        return id

    chassis = box("Шасси", (0, 0, 0), (1.0, 0.05, 0.4))
    ed.scene_set(id=chassis, line="rigid", key="density", values=[250])
    cabin = box("Кабина", (-0.08, 0.075, 0), (0.4, 0.1, 0.3))
    ed.assembly_joint(a=cabin, b=chassis, type="weld", anchor=[x0 - 0.08, y + 0.075, z0])
    base, track = 0.7, 0.59
    for x in (base / 2, -base / 2):
        for side in (1, -1):
            hub = box("Ступица", (x, 0, side * 0.23), (0.06, 0.06, 0.06))
            ed.assembly_joint(a=hub, b=chassis, type="shock", anchor=[x0 + x, y, z0 + side * 0.2], axis=[0, 1, 0])
            carrier = hub
            if x > 0:
                carrier = box("Кулак", (x, 0, side * 0.275), (0.04, 0.04, 0.03))
                king = ed.assembly_joint(a=carrier, b=hub, type="hinge", anchor=[x0 + x, y, z0 + side * 0.26], axis=[0, 1, 0])["id"]
                ed.assembly_drive(joint=king, kind="servo", stiffness=40, damping=2, target=0)
            wheel = ed.scene_add(shape="cylinder", name="Колесо", position=[x0 + x, y, z0 + side * 0.32], size=[0.2, 0.06, 0.2], role="rigid")["id"]
            made.append(wheel)
            ed.scene_turn(ids=[wheel], axis="x", degrees=90)
            axle = ed.assembly_joint(a=wheel, b=carrier, type="hinge", anchor=[x0 + x, y, z0 + side * 0.29], axis=[0, 0, 1])["id"]
            if x < 0:
                ed.assembly_drive(joint=axle, kind="velocity", velocity=-1, force=4)  # its sign: forward (robot.JointCar)
    lidar = ed.assembly_sensor(target=chassis, kind="lidar", face="+y", model="TFmini-S")["id"]
    ed.sensor_spec(sensor=lidar, spec={"rays_h": 21, "fov_h_deg": 200, "max_m": 8, "rate_hz": 30})
    # over the cabin's roof, looking ahead (the race car's: its 10 cm it cannot see stay over the car), level
    ed.scene_set(id=lidar, line="pose", key="position", values=[0, 0.145, 0])
    ed.scene_set(id=lidar, line="pose", key="rotation", values=[-90, 0, 0])  # as the race car's on its nose: the fan level
    ed.assembly_sensor(target=chassis, kind="imu", face="+y")
    if heading:
        ed.scene_turn(ids=made, axis="y", degrees=heading, pivot=[x0, y, z0])
    return chassis
