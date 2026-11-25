"""showcase - one scene that shows what the editor can do, played by pressing ▶.

«Собрать сцену» builds it: on a chessboard the UR5e stands over the white queen; beside it a ball hangs on a pendulum
next to a tower of blocks; spotlights, a camera of the scene with its keys, formulas over the bodies. ▶ plays the story:
the hand takes the queen, twirls it by the wrist and sets it down on d8 - mate; a force of the scene sets the ball off
and it breaks the tower; the camera films six shots, each one slow move, cut to the next; words stand in the frame and
the bodies' own numbers beside them. Everything seen is the editor's own picture of its own scene: look through
«Камера фильма», or fly round the stage yourself while it plays. The scene can be saved and opened again: the story
finds its actors by their tags.

The plugins `textures` and `formula` of this repository give the scene its surfaces and its formulas (the latter needs
latex and dvisvgm); without them the scene is built bare and plays the same. The repository's film is this scene
recorded a frame at a time (tools/showcase_film.py of the editor).
"""
import json
import math
import os

import kinetica as ke
from kinetica.plugin import command, draw, on_play, on_stop, on_tick, show, tr, ui

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ke.__file__)))), "examples")
KEPT = os.path.join(os.path.expanduser("~"), "Kinetica", "showcase")  # the pieces' models and the titles' outlines

FPS = 60
LENGTH = 29.0  # seconds of the scene's time

# The story, in seconds of the scene's time: the hand over the queen, down on it, its fingers closed, up to where it
# twirls the queen, straight again, over d8, down, open, away - and the pendulum set off.
REACH, LOWER, GRIP, LIFT, TWIRL, STRAIGHT, CARRY, PLACE, OPEN, AWAY = 0.5, 3.3, 5.4, 6.6, 8.2, 12.2, 12.8, 14.4, 15.8, 16.4
STRIKE = 17.9

BOARD, SQUARE, BOARD_TOP = (0.0, -0.55), 0.05, 0.036  # the board's centre (x, z), a square's side, its top's height
GRASP = BOARD_TOP + 0.05                               # where the fingers take a piece: its waist
HAND = 0.074                                           # the arm's goal stands this much over what its fingers hold
HOLD = (0.03, 0.36, -0.52)                             # where the hand twirls the queen
DOWN = (0.0, 1.0, 0.0, 0.0)                            # the hand pointing down, a quaternion (w, x, y, z)

PIVOT = (1.05, 1.45, 0.75)  # where the pendulum hangs
ROD = 1.2                   # from the pivot to the ball's centre, m
TOWER = (1.53, 0.75)        # x, z of the tower: the ball's first step from rest

# The pieces, each a profile turned about its axis: (radius, height) from the foot up, metres.
PIECES = {
    "king": [(0.017, 0), (0.017, 0.004), (0.0145, 0.008), (0.0125, 0.013), (0.009, 0.022), (0.007, 0.038), (0.0062, 0.054), (0.0078, 0.062),
             (0.011, 0.065), (0.011, 0.068), (0.008, 0.07), (0.0095, 0.08), (0.0115, 0.084), (0.0075, 0.086), (0.003, 0.088), (0.003, 0.094),
             (0.0065, 0.0945), (0.0065, 0.0975), (0.003, 0.098), (0.003, 0.103), (0.0, 0.1035)],
    "queen": [(0.016, 0), (0.016, 0.004), (0.0135, 0.008), (0.012, 0.012), (0.0085, 0.02), (0.0065, 0.035), (0.0058, 0.05), (0.0072, 0.058),
              (0.0105, 0.061), (0.0105, 0.064), (0.0075, 0.066), (0.0085, 0.074), (0.0115, 0.08), (0.0075, 0.082), (0.004, 0.086), (0.0045, 0.089),
              (0.0, 0.0915)],
    "rook": [(0.016, 0), (0.016, 0.004), (0.0135, 0.008), (0.012, 0.012), (0.0095, 0.018), (0.0088, 0.038), (0.0105, 0.042), (0.013, 0.044),
             (0.013, 0.056), (0.0105, 0.056), (0.0105, 0.051), (0.0, 0.051)],
    "bishop": [(0.016, 0), (0.016, 0.004), (0.0135, 0.008), (0.0118, 0.012), (0.0082, 0.02), (0.0062, 0.034), (0.0056, 0.044), (0.0074, 0.05),
               (0.0098, 0.052), (0.0098, 0.055), (0.0068, 0.057), (0.0086, 0.064), (0.0074, 0.071), (0.0042, 0.077), (0.0022, 0.079), (0.0032, 0.081),
               (0.0, 0.083)],
    "pawn": [(0.0145, 0), (0.0145, 0.0035), (0.012, 0.007), (0.0105, 0.01), (0.0068, 0.017), (0.0052, 0.028), (0.007, 0.032), (0.009, 0.034),
             (0.006, 0.036), (0.0085, 0.04), (0.0098, 0.045), (0.0085, 0.05), (0.005, 0.0535), (0.0, 0.0545)],
}
# The position: white to move, the queen from d1 to d8 - mate on the back rank.
WHITE = {"g1": "king", "d1": "queen", "a1": "rook", "f2": "pawn", "g2": "pawn", "h2": "pawn"}
BLACK = {"g8": "king", "b7": "bishop", "a6": "pawn", "f7": "pawn", "g7": "pawn", "h7": "pawn"}


def square(name):
    """The centre of a square of the board, (x, z): the files along x, the first rank the robot's."""
    return BOARD[0] + ("abcdefgh".index(name[0]) - 3.5) * SQUARE, BOARD[1] + 0.175 - (int(name[1]) - 1) * SQUARE


def lathe(path, profile, segments=72):
    """A profile turned about the vertical axis, written as an OBJ model."""
    lines, rows = [], len(profile)
    for radius, height in profile:
        for k in range(segments):
            turn = 2 * math.pi * k / segments
            lines.append(f"v {radius * math.cos(turn):.5f} {height:.5f} {radius * math.sin(turn):.5f}")
    lines += [f"v 0 {profile[0][1]:.5f} 0", f"v 0 {profile[-1][1]:.5f} 0"]
    foot, tip = rows * segments + 1, rows * segments + 2
    for row in range(rows - 1):
        for k in range(segments):
            a, b = row * segments + k + 1, row * segments + (k + 1) % segments + 1
            lines += [f"f {a} {a + segments} {b + segments}", f"f {a} {b + segments} {b}"]
    for k in range(segments):
        lines += [f"f {foot} {k + 1} {(k + 1) % segments + 1}", f"f {tip} {(rows - 1) * segments + (k + 1) % segments + 1} {(rows - 1) * segments + k + 1}"]
    with open(path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")


def put(scene, thing, line, key, *values):
    scene.ed.scene_set(id=getattr(thing, "id", thing), line=line, key=key, values=list(values))


def look(eye, target):
    """A free camera's rotation (Euler degrees: pitch, yaw, roll) that looks from `eye` at `target` along its -z."""
    f = [t - e for e, t in zip(eye, target)]
    n = math.sqrt(sum(v * v for v in f))
    return [math.degrees(math.asin(f[1] / n)), math.degrees(math.atan2(-f[0], -f[2])), 0.0]


def turned(axis, degrees):
    half = math.radians(degrees) / 2
    return (math.cos(half), *[v * math.sin(half) for v in axis])


def times(a, b):
    """The turn b, then the turn a: quaternions (w, x, y, z)."""
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


# The film's shots: each one slow move of the camera, cut to the next. A move goes from its first eye and aim to its
# second, or round a point (`about`: its distance, its angle from the world's +x towards -z, its height - each from, to).
D1, D8 = square("d1"), square("d8")
SHOTS = [
    {"from": 0.0, "fov": 40, "about": (0.45, -0.1), "far": (4.5, 3.9), "angle": (57, 44), "high": (2.05, 1.45), "aim": ((0.45, 0.5, -0.1), (0.3, 0.42, -0.25))},
    {"from": 3.4, "fov": 30, "eye": ((-0.55, 0.21, -0.86), (-0.43, 0.14, -0.72)), "aim": ((D1[0], 0.1, D1[1]), (D1[0], 0.12, D1[1]))},
    {"from": 6.8, "fov": 36, "about": (HOLD[0], HOLD[2]), "far": (0.6, 0.48), "angle": (150, 68), "high": (0.3, 0.38), "aim": ((0.03, 0.27, -0.52), (0.03, 0.29, -0.52))},
    {"from": 12.6, "fov": 32, "eye": ((-0.56, 0.2, -1.06), (-0.4, 0.15, -0.96)), "aim": ((D8[0], 0.2, D8[1]), (D8[0], 0.09, D8[1]))},
    {"from": 17.4, "fov": 38, "eye": ((2.5, 0.5, -0.4), (2.9, 0.62, -0.05)), "aim": ((1.55, 0.38, 0.75), (1.95, 0.3, 0.75))},
    {"from": 20.4, "fov": 40, "about": (0.5, 0.0), "far": (3.3, 4.9), "angle": (40, 66), "high": (0.95, 2.4), "aim": ((0.6, 0.4, 0.2), (0.5, 0.4, 0.0))},
]


def shot_at(t):
    """The shot that runs at time t, and how far through it t is: 0 at its first frame, 1 at its last."""
    k = max(i for i, shot in enumerate(SHOTS) if shot["from"] <= t + 1e-9)
    since, until = SHOTS[k]["from"], SHOTS[k + 1]["from"] - 1 / FPS if k + 1 < len(SHOTS) else LENGTH
    return SHOTS[k], min(1.0, max(0.0, (t - since) / (until - since)))


def camera_at(t):
    """Where the film's camera stands at time t and what it looks at: its move a little slower at its two ends."""
    shot, s = shot_at(t)
    s = 0.45 * s + 0.55 * s * s * (3 - 2 * s)
    mix = lambda pair: [a + (b - a) * s for a, b in zip(*pair)] if isinstance(pair[0], tuple) else pair[0] + (pair[1] - pair[0]) * s  # noqa: E731
    if "about" in shot:
        far, angle = mix(shot["far"]), math.radians(mix(shot["angle"]))
        eye = [shot["about"][0] + far * math.cos(angle), mix(shot["high"]), shot["about"][1] - far * math.sin(angle)]
    else:
        eye = mix(shot["eye"])
    return eye, mix(shot["aim"])



def build_scene(scene):
    """The scene, and which of the two other plugins had a hand in it."""
    ed = scene.ed
    there = {op["name"] for op in ed.ops_list(families=["textures", "formula"])["ops"]}
    textures, formulas = "textures_apply" in there, "formula_outlines" in there
    scene.open(os.path.join(EXAMPLES, "ur5e.kscene"))
    for name in ("Кубик жёлтый", "Кубик зелёный", "Кубик пурпурный", "Площадка"):
        scene.objects[name].delete()
    for light in [o for o in scene.objects if o.kind == "light"]:
        light.delete()
    floor = scene.objects["Пол"]
    ed.world_set(line="world", key="size", values=[60, 20, 60])  # the stage's walls far out of the light
    ed.world_set(line="world", key="duration", values=[LENGTH])  # ▶ plays the story to its end and stops
    put(scene, floor, "shape", "size", 40, 0.02, 40)

    def texture(shapes, kind, repeats, relief=2.0, variant=0, colours=None):
        if not textures:
            return
        scene.editor.select(*shapes) if isinstance(shapes, list) else scene.editor.select(shapes)
        own = {"own": True, "base": list(colours[0]), "mark": list(colours[1])} if colours else {}
        ed.textures_apply(kind=kind, repeats=repeats, relief=relief, variant=variant, maps="нормали", **own)

    def formula(body, kind, **values):
        if formulas:
            ed.component_set(id=body.id, type=f"formula.{kind}", values=values)

    def tag(thing, word):
        ed.scene_tag(ids=[thing.id], tag=word, on=True)

    texture(floor, "плитка", 0.5, relief=1.5, colours=((0.21, 0.22, 0.27), (0.08, 0.085, 0.11)))  # dark slate, half a metre a tile

    # The board: its squares the built-in checker, a walnut rim round it; the pieces turned models with a box to be held by.
    rim = scene.add("box", name="Рама доски", position=(BOARD[0], 0.025, BOARD[1]), size=(0.47, 0.01, 0.47), role="rigid", fixed=True)
    texture(rim, "дерево", 6.0, relief=0.5, variant=2, colours=((0.36, 0.22, 0.13), (0.2, 0.11, 0.06)))
    board = scene.add("box", name="Доска", position=(BOARD[0], BOARD_TOP - 0.008, BOARD[1]), size=(0.4, 0.016, 0.4), role="rigid", fixed=True)
    put(scene, board, "texture", "kind", "checker"), put(scene, board, "texture", "repeats", 1 / (2 * SQUARE))
    put(scene, board, "shape", "color", 0.9, 0.88, 0.82)
    os.makedirs(KEPT, exist_ok=True)
    pieces = {}
    for side, colour in ((WHITE, (0.94, 0.91, 0.83)), (BLACK, (0.11, 0.11, 0.13))):
        for where, kind in side.items():
            model = os.path.join(KEPT, kind + ".obj")
            if not os.path.exists(model):
                lathe(model, PIECES[kind])
            tall, (x, z) = PIECES[kind][-1][1], square(where)
            wide = 2 * PIECES[kind][5][0] + 0.012  # a box round its waist: what stands on the board and what the fingers close on
            made = ed.scene_put(block=f'entity "{kind} {where}"\n  object id 0\n  shape mesh file "{model}" size {tall} {tall} {tall} position {x} {BOARD_TOP + tall / 2} {z} '
                                      f'rotation 0 0 0 color {colour[0]} {colour[1]} {colour[2]}\n  rigid density 1400 friction 0.9 restitution 0.1\n'
                                      f'  collider kind box fit 0 size {wide} {tall} {wide} offset 0 0 0 rotation 0 0 0\nend\n')
            pieces[where] = scene.objects[made["id"]]
    queen = pieces["d1"]

    # The pendulum: a frame of two posts and a beam, a rod on a hinge under the beam, a ball welded to its end.
    px, py, pz = PIVOT
    frame = [scene.add("box", name="Стойка", position=(px, (py + 0.1) / 2, pz + dz), size=(0.07, py + 0.1, 0.07), role="rigid", fixed=True) for dz in (-0.42, 0.42)]
    beam = scene.add("box", name="Балка", position=(px, py + 0.055, pz), size=(0.07, 0.07, 0.91), role="rigid", fixed=True)
    texture(frame + [beam], "металл", 2.0, relief=0.6)
    rod = scene.add("box", name="Штанга", position=(px, py - ROD / 2, pz), size=(0.03, ROD, 0.03), role="rigid")
    put(scene, rod, "shape", "color", 0.75, 0.77, 0.8)
    ball = scene.add("sphere", name="Шар", position=(px, py - ROD, pz), size=(0.26, 0.26, 0.26), role="rigid")
    put(scene, ball, "rigid", "density", 2600)
    texture(ball, "мрамор", 3.0, relief=0.5, variant=4, colours=((0.14, 0.15, 0.18), (0.75, 0.62, 0.32)))  # black marble, gold veins
    scene.add_joint(rod, kind="hinge", anchor=(px, py, pz), axis=(0, 0, 1))
    scene.add_joint(ball, rod, kind="weld", anchor=(px, py - ROD, pz))
    strike = scene.add_force("directional", "impulse", body=ball, strength=150.0)
    strike.turn("z", -90)  # its +y, the way it pushes, turned along the world's +x: at the tower
    put(scene, strike, "effect", "start", STRIKE)

    # The tower: three blocks a course, each course across the one under it.
    blocks, (tx, tz) = [], TOWER
    for course in range(7):
        for k in (-1, 0, 1):
            across = course % 2 == 1
            blocks.append(scene.add("box", name="Брусок", position=(tx + (0.0 if across else 0.082 * k), 0.02 + 0.04 + 0.08 * course, tz + (0.082 * k if across else 0.0)),
                                    size=(0.08, 0.08, 0.245) if not across else (0.245, 0.08, 0.08), role="rigid"))
    texture(blocks, "дерево", 4.0, relief=0.8, variant=3)

    # Formulas: what each thing is about. The plugin `formula` typesets them and the editor keeps them on their bodies.
    gold = [1.0, 0.86, 0.35]
    formula(queen, "label", latex=r"m\vec{g}", size=3.0, offset=[0, 0.085, 0], colour=gold)
    formula(ball, "label", latex=r"\vec{p} = m\vec{v}", size=3.6, offset=[0, 0.3, 0], colour=gold)
    formula(scene.objects["UR5e предплечье"], "label", latex=r"\vec{\tau} = J^{T}\vec{F}", size=3.2, offset=[0, 0.22, 0], colour=gold)
    formula(beam, "world", latex=r"T = 2\pi\sqrt{\frac{L}{g}}", size=0.11, offset=[0, 0.17, 0], turn=[0, 180, 0], colour=[0.72, 0.66, 1.0])

    # Light and dark: a strong warm key from the front left, a cool rim from behind that travels, a faint sun to lift the
    # shadows, an ember behind the tower and a lamp over the board. Each lamp fades to nothing a little past the stage:
    # what the story is about stands in the light, the rest is dark.
    key = scene.add_light("spot", position=(-2.0, 3.2, -2.6), target=(0.6, 0.2, 0.2))
    rim_light = scene.add_light("spot", position=(2.8, 2.6, 3.2), target=(0.6, 0.35, 0.2))
    fill = scene.add_light("sun", position=(1.0, 4.0, -2.0), target=(0.0, 0.0, 0.0))
    glow = scene.add_light("point", position=(2.1, 0.55, 1.5))
    lamp = scene.add_light("point", position=(0.25, 0.75, -0.95))
    for light, colour, intensity, shadows, cone, reach in ((key, (1.0, 0.86, 0.66), 4.6, True, 80, 11), (rim_light, (0.42, 0.62, 1.0), 3.4, True, 80, 12),
                                                          (fill, (0.8, 0.86, 1.0), 0.2, False, 0, 24), (glow, (1.0, 0.45, 0.2), 2.2, False, 0, 4),
                                                          (lamp, (1.0, 0.93, 0.8), 1.6, False, 0, 2.6)):
        put(scene, light, "pose", "color", *colour)
        light.intensity, light.shadows = intensity, shadows
        put(scene, light, "kind", "range", reach)
        if cone:
            put(scene, light, "kind", "cone", cone), put(scene, light, "kind", "softness", 25)

    # The camera of the film and the keys of what the sequence leads: the camera through its shots - a key every fifth of
    # a second, and the last frame of a shot beside the first of the next: a cut -, the rim light round the stage.
    camera = scene.add_camera(eye=camera_at(0)[0], target=camera_at(0)[1])
    put(scene, camera, "lens", "fov", SHOTS[0]["fov"])
    moments = sorted({round(k * 0.2, 4) for k in range(int(LENGTH * 5) + 1)} | {shot["from"] for shot in SHOTS} | {round(shot["from"] - 1 / FPS, 4) for shot in SHOTS[1:]})
    for t in moments:
        eye, aim = camera_at(t)
        put(scene, camera, "pose", "position", *eye)
        put(scene, camera, "pose", "rotation", *look(eye, aim))
        ed.sequence_key(ids=[camera.id], time=t)
    for t, place in ((0.0, (2.8, 2.6, 3.2)), (LENGTH / 2, (0.4, 2.9, 3.8)), (LENGTH, (-2.2, 2.6, 3.0))):
        put(scene, rim_light, "pose", "position", *place)
        ed.sequence_key(ids=[rim_light.id], time=t, interp="smooth")
    ed.sequence_set(play=True)
    ed.sequence_seek(time=0)

    # The actors by their tags: the story finds them in a scene saved and opened again.
    for thing, word in ((camera, "film-camera"), (queen, "film-queen"), (scene.objects["Цель «Схват»"], "film-hand"), (ball, "film-ball")):
        tag(thing, word)
    for joint in ("Шарнир 11", "Шарнир 13", "Шарнир 15"):
        tag(scene.objects[joint], "film-wrist")
    if formulas:  # the words of the frame, typeset once and kept: a tick has no time for LaTeX
        with open(os.path.join(KEPT, "titles.json"), "w", encoding="utf-8") as kept:
            json.dump({latex: ed.formula_outlines(latex=latex)["outlines"] for _, _, latex, _, _, _ in TITLES}, kept)
    scene.editor.select()
    return textures, formulas


def act(scene, parts, t, done):
    """What the arm is told at the scene's time t: each move once, when its moment comes - and while it twirls the queen,
    the turn of its hand every frame: a swing about the vertical there and back, with a lean that goes round."""
    goal, low = parts["goal"], GRASP + HAND
    moves = ((REACH, lambda: goal.pose(position=(D1[0], low + 0.12, D1[1]))), (LOWER, lambda: goal.pose(position=(D1[0], low, D1[1]))),
             (GRIP, lambda: goal.grip(force=6)), (LIFT, lambda: goal.pose(position=HOLD)), (STRAIGHT, lambda: goal.pose(position=HOLD, quaternion=DOWN)),
             (CARRY, lambda: goal.pose(position=(D8[0], 0.34, D8[1]))), (PLACE, lambda: goal.pose(position=(D8[0], low + 0.004, D8[1]))),
             (OPEN, lambda: goal.grip(close=False)), (AWAY, lambda: goal.pose(position=(D8[0] + 0.05, 0.38, D8[1] + 0.16))))
    for k, (moment, move) in enumerate(moves):
        if t >= moment and k not in done:
            done.add(k)
            move()
    if TWIRL <= t < STRAIGHT:
        s = (t - TWIRL) / (STRAIGHT - TWIRL)
        eased = s * s * (3 - 2 * s)
        swing, lean, round_ = 135 * math.sin(2 * math.pi * eased), 38 * math.sin(math.pi * s), 3 * math.pi * eased
        goal.pose(position=HOLD, quaternion=times(turned((math.cos(round_), 0, math.sin(round_)), lean), times(turned((0, 1, 0), swing), DOWN)))


# The words in the frame: (from, to, LaTeX, where, size, colour). Where: a place in the frame (x, y), or a point of the
# world (x, y, z) the words stand over.
WHITE_, PALE = (1.0, 1.0, 1.0), (0.74, 0.82, 1.0)
TITLES = [(0.4, 3.3, r"\text{Kinetica}", (0.5, 0.74), 0.115, WHITE_),
          (0.9, 3.3, r"\text{robotics} \cdot \text{physics} \cdot \text{plugins}", (0.5, 0.87), 0.04, PALE),
          (3.7, 6.6, r"\text{inverse kinematics}", (0.5, 0.91), 0.044, WHITE_),
          (7.2, 12.4, r"\text{six joints, one wrist}", (0.5, 0.91), 0.044, WHITE_),
          (16.0, 17.35, r"\text{Qd8\#}", (D8[0], 0.2, D8[1]), 0.05, (1.0, 0.86, 0.35)),
          (17.6, 20.3, r"\text{rigid-body physics}", (0.5, 0.91), 0.044, WHITE_),
          (21.2, 99.0, r"\text{Kinetica Editor}", (0.5, 0.76), 0.085, WHITE_),
          (21.7, 99.0, r"\text{github.com/Kinetica-Labs}", (0.5, 0.87), 0.032, PALE)]




@command(label="Собрать сцену")
def build(scene):
    """Сцена-витрина: рука над шахматной доской, маятник у башни, свет, камера по ключам, формулы. Потом ▶ - и история играет сама."""
    textures, formulas = build_scene(scene)
    missing = [name for name, there in (("textures", textures), ("formula", formulas)) if not there]
    words = tr("сцена собрана: ▶ играет историю") + (tr("; без плагинов {missing} - без текстур и формул").format(missing=", ".join(missing)) if missing else "")
    show(state=words)
    return {"note": words}


_story = {}  # the run's own: its time, what is done, what is drawn, the actors


def _tagged(scene, word):
    return [scene.objects[o["id"]] for o in scene.ed.scene_list(tag=word)["objects"]]


@on_play
def begin(scene):
    """The story from its start - when the scene is the showcase's: its actors are there by their tags."""
    _story.clear()
    hand, queen, ball, camera = (_tagged(scene, word) for word in ("film-hand", "film-queen", "film-ball", "film-camera"))
    if not (hand and queen and ball and camera):
        return
    titles = {}
    if os.path.exists(os.path.join(KEPT, "titles.json")):
        with open(os.path.join(KEPT, "titles.json"), encoding="utf-8") as kept:
            titles = json.load(kept)
    _story.update(t=0.0, done=set(), shown=set(), lens=SHOTS[0]["fov"], titles=titles,
                  parts={"goal": hand[0], "queen": queen[0], "ball": ball[0], "camera": camera[0], "wrist": _tagged(scene, "film-wrist")})
    put(scene, camera[0], "lens", "fov", SHOTS[0]["fov"])
    scene.ed.view_through(camera=camera[0].id)  # the film as its camera sees it; «Смотрю через» gives the editor's own view back


@on_tick
def play(scene, dt, sensors):
    """One frame of the story: the arm's move if its moment has come, the lens of the shot, the words and the numbers."""
    if not _story:
        return {}
    t, parts = _story["t"], _story["parts"]
    act(scene, parts, t, _story["done"])
    if shot_at(t)[0]["fov"] != _story["lens"]:  # a cut: the next shot's lens
        _story["lens"] = shot_at(t)[0]["fov"]
        put(scene, parts["camera"], "lens", "fov", _story["lens"])
    shown = _story["shown"]
    for k, (since, until, latex, where, size, colour) in enumerate(TITLES):
        on = since <= t < until and latex in _story["titles"]
        if on and k not in shown:
            shown.add(k)
            place = {"space": "screen", "place": where} if len(where) == 2 else {"space": "label", "at": where}
            draw.fill(f"title/{k}", _story["titles"][latex], scale=size, colour=colour, stroke=(0.02, 0.03, 0.05), stroke_width=0.05, **place)
        elif not on and k in shown:
            shown.discard(k)
            draw.clear(f"title/{k}")
    drawing(t, shown)
    numbers(scene, parts, t, shown)
    _story["t"] = t + dt
    return {}


@on_stop
def end(scene):
    if _story:
        scene.ed.view_through()
    _story.clear()


def drawing(t, shown):
    """The first shot's drawing round the arm, as a draughtsman lays a figure out: the circle its hand can reach, on the
    floor and upright, and the square it stands in - gone at the first cut."""
    on = 0.25 <= t < SHOTS[1]["from"] - 1 / FPS
    if on and "reach" not in shown:
        shown.add("reach")
        reach, shoulder, gold = 0.85, 0.163, (0.95, 0.8, 0.42)
        turn = math.radians(50)  # the upright figures face the first shot's camera
        across, to_camera = (math.sin(turn), 0.0, math.cos(turn)), (math.cos(turn), 0.0, -math.sin(turn))
        draw.circle("reach/floor", (0, 0.024, 0), reach, colour=gold)
        draw.circle("reach/ring", (0, shoulder, 0), reach, axis=to_camera, colour=gold)
        side = shoulder + reach
        corners = [(-side / 2, 0.024), (side / 2, 0.024), (side / 2, side), (-side / 2, side), (-side / 2, 0.024)]
        draw.line("reach/square", [(across[0] * u, v, across[2] * u) for u, v in corners], colour=gold)
    elif not on and "reach" in shown:
        shown.discard("reach")
        draw.clear("reach/")


def numbers(scene, parts, t, shown):
    """What the bodies measure, written beside them while it matters: the wrist's three angles as the hand twirls the queen,
    the ball's momentum and speed as it goes for the tower."""
    ball, wrist, sky = parts["ball"], parts["wrist"], (0.62, 0.9, 1.0)
    if TWIRL - 0.8 <= t < STRAIGHT + 0.5 and wrist:
        seen = scene.measure(joints=wrist)
        angles = "   ".join(f"{math.degrees(seen[f'{joint.id}/rx coordinate']):+.0f}°" for joint in wrist)
        draw.label("number", (HOLD[0], HOLD[1] + 0.2, HOLD[2]), "wrist   " + angles, colour=sky, size=21)
        shown.add("number")
    elif STRIKE - 0.5 <= t < SHOTS[5]["from"] - 1 / FPS:
        seen = scene.measure(objects=[ball])
        at = [seen[f"{ball.id}/centre {axis}"] for axis in "xyz"]
        draw.label("number", (at[0], at[1] + 0.2, at[2]), f"{seen[f'{ball.id}/momentum']:.0f} kg m/s    {seen[f'{ball.id}/speed']:.1f} m/s", colour=sky, size=21)
        shown.add("number")
    elif "number" in shown:
        shown.discard("number")
        draw.clear("number")


ui.panel("Витрина", ui.label("Одна сцена, в которой занято всё: рука и схват, суставы и твёрдые тела, сила, свет с тенями, камера по ключам, формулы и текстуры."),
         ui.button(build, "Собрать сцену"), ui.label("Потом ▶: история играет сама, вид - через «Камеру фильма»."), ui.value("state", "Состояние", "—"))
