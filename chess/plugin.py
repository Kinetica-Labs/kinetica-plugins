"""chess - «Шахматы»: a board before the arm, and a planner that moves a piece at a word - the agent's or a person's.

«Доска» (`chess_board`) puts the UR5e on the floor if the scene has no arm with an IK goal, and before it a chessboard of
5 cm squares with the pieces in their starting places - turned models with a box at the waist for the fingers to close on.
«Ход» (`chess_move {move: "e2e4"}`) plans at once, before anything moves: over the piece, down to its waist, the fingers
closed, up, over the square it goes to, down, open, up - a capture first carries the taken piece off the board beside
it. Every point is asked of inverse kinematics first (assembly_can_reach, the hand pointing down): a move the arm cannot
make is refused with the square it cannot reach, nothing moved. The answer is the plan - the piece's id, its points and
how long it takes;
the plan's path is drawn over the board. While the scene plays the plugin walks it in the scene's own time (on_tick): each
point given to the arm's goal when the last is reached, the hand led there by the editor (RobotWorld: 0.4 m/s), the
fingers closed with 6 N and the plugin waiting until they hold. Plans queue: several moves said at once are made in turn.
`chess_state` answers where the pieces stand and what the hand is doing.
"""
import math
import os

from kinetica.plugin import command, draw, field, on_stop, on_tick, show, tr, ui

SQUARE, BOARD_TOP, AHEAD = 0.05, 0.036, 0.55   # a square's side; the board's top over the floor; its centre before the arm's base, m
WAIST, HAND, CLEAR = 0.05, 0.074, 0.12         # a piece is held this high over the board; the goal stands HAND over what it holds; lifted CLEAR
DOWN = (0.0, 1.0, 0.0, 0.0)                    # the hand pointing down, a quaternion (w, x, y, z)
LEAD, SETTLE, HOLD = 0.4, 0.45, 0.7            # the editor's lead of a hand, m/s; s still at each point; s for the fingers to close or open
FORCE, PATH = 6.0, (0.35, 0.8, 1.0)            # N the fingers press with; the plan's colour
KEPT = os.path.join(os.path.expanduser("~"), "Kinetica", "chess")  # the pieces' models

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
    "knight": [(0.016, 0), (0.016, 0.004), (0.0135, 0.008), (0.012, 0.012), (0.009, 0.02), (0.0075, 0.03), (0.0095, 0.045), (0.0105, 0.058),
               (0.0085, 0.066), (0.005, 0.071), (0.0, 0.072)],
    "pawn": [(0.0145, 0), (0.0145, 0.0035), (0.012, 0.007), (0.0105, 0.01), (0.0068, 0.017), (0.0052, 0.028), (0.007, 0.032), (0.009, 0.034),
             (0.006, 0.036), (0.0085, 0.04), (0.0098, 0.045), (0.0085, 0.05), (0.005, 0.0535), (0.0, 0.0545)],
}
BACK = ("rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook")
NAMES = {"king": "король", "queen": "ферзь", "rook": "ладья", "bishop": "слон", "knight": "конь", "pawn": "пешка"}

_board = {"centre": None, "goal": None, "at": {}}  # the board's centre (x, z); the arm's goal; square -> the piece's id
_plan = []                                          # what the arm is still to do: (what, data, words), in turn
_walk = {"t": 0.0, "until": 0.0, "from": None}      # the scene's time; when the step under way is done; where the hand was sent last


def square(name, centre):
    """The centre of a square, (x, z): the files a-h along x, the first rank the arm's - nearest to it."""
    return centre[0] + ("abcdefgh".index(name[0]) - 3.5) * SQUARE, centre[1] + 3.5 * SQUARE - (int(name[1]) - 1) * SQUARE


def lathe(path, profile, segments=48):
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


def world(scene, thing):
    """Where an object stands in the world: its place and its parents' (the groups an assembly is put in are not turned)."""
    at, up = list(thing.position), getattr(thing, "parent", None)
    while up is not None:
        at = [a + b for a, b in zip(at, up.position)]
        up = getattr(up, "parent", None)
    return at


def find_goal(scene):
    """The scene's arm: (its IK goal's id, the effector's id), or None."""
    for o in scene.objects:
        if o.kind != "entity":
            continue
        lines = scene.ed.scene_get(id=o.id)["lines"]
        goal = next((line for line in lines if line[:2] == ["goal", "effector"]), None)
        if goal:
            return o.id, int(goal[2])
    return None


def plan(move, centre, at):
    """The points of a move - [(what, data, words)] - and its pieces' new places; a capture first carries the taken piece
    off the board, to the right of it. Raises ValueError in words when the move is not one."""
    move = str(move).strip().lower().replace("-", "").replace(" ", "")
    if len(move) != 4 or move[0] not in "abcdefgh" or move[2] not in "abcdefgh" or move[1] not in "12345678" or move[3] not in "12345678":
        raise ValueError(tr("ход - два поля, откуда и куда: e2e4"))
    a, b = move[:2], move[2:]
    if a not in at:
        raise ValueError(tr("на {square} нет фигуры").format(square=a))
    if a == b:
        raise ValueError(tr("фигура уже на {square}").format(square=a))
    low, high = BOARD_TOP + WAIST + HAND, BOARD_TOP + WAIST + HAND + CLEAR
    steps, after = [], dict(at)

    def carry(start, end, end_xz, name, floor=False):
        (sx, sz), (ex, ez) = square(start, centre), end_xz
        put = low + 0.004 - (BOARD_TOP if floor else 0)  # set down on the board, or off it on the floor
        steps.extend([("go", (sx, high, sz), tr("над {square}").format(square=start)), ("go", (sx, low, sz), tr("к фигуре")),
                      ("grip", True, tr("взять {name}").format(name=name)), ("go", (sx, high, sz), tr("поднять")),
                      ("go", (ex, high, ez), tr("над {square}").format(square=end)), ("go", (ex, put, ez), tr("опустить")),
                      ("grip", False, tr("отпустить")), ("go", (ex, high, ez), tr("отойти"))])

    if b in after:  # a capture: the taken piece off the board first
        taken = sum(1 for k in after if k.startswith("x"))
        off = (centre[0] + 5.2 * SQUARE, centre[1] + (3.5 - taken % 8) * SQUARE)
        carry(b, tr("за доску"), off, tr("взятую"), floor=True)
        after[f"x{taken}"] = after.pop(b)
    carry(a, b, square(b, centre), tr("фигуру"))
    after[b] = after.pop(a)
    return steps, after


def seconds(steps, start):
    """How long the arm takes over the steps: each point at the editor's lead speed and a settle, each grip HOLD."""
    total, here = 0.0, start
    for what, data, _ in steps:
        if what == "go":
            total += (math.dist(here, data) / LEAD if here else 1.0) + SETTLE
            here = data
        else:
            total += HOLD
    return total


@command(label="Доска")
def board(scene):
    """Шахматная доска перед рукой и фигуры в начальной позиции; рука UR5e, если в сцене нет руки с целью."""
    arm = find_goal(scene)
    if not arm:
        scene.ed.assembly_add(name="Рука UR5e", position=[0, 0, 0])
        arm = find_goal(scene)
    base = next((o for o in scene.objects if o.name == "UR5e основание"), None)
    bx, _, bz = world(scene, base) if base else world(scene, scene.objects[arm[1]])
    centre = (bx, bz - AHEAD)
    rim = scene.add("box", name="Рама доски", position=(centre[0], 0.025, centre[1]), size=(0.47, 0.01, 0.47), role="rigid", fixed=True)
    scene.ed.scene_set(id=rim.id, line="shape", key="color", values=[0.33, 0.2, 0.12])
    deck = scene.add("box", name="Доска", position=(centre[0], BOARD_TOP - 0.008, centre[1]), size=(0.4, 0.016, 0.4), role="rigid", fixed=True)
    scene.ed.scene_set(id=deck.id, line="texture", key="kind", values=["checker"])
    scene.ed.scene_set(id=deck.id, line="texture", key="repeats", values=[1 / (2 * SQUARE)])
    scene.ed.scene_set(id=deck.id, line="shape", key="color", values=[0.9, 0.88, 0.82])
    os.makedirs(KEPT, exist_ok=True)
    at = {}
    for rank, kinds, colour in (("1", BACK, (0.94, 0.91, 0.83)), ("2", ("pawn",) * 8, (0.94, 0.91, 0.83)),
                                ("7", ("pawn",) * 8, (0.11, 0.11, 0.13)), ("8", BACK, (0.11, 0.11, 0.13))):
        for file, kind in zip("abcdefgh", kinds):
            model = os.path.join(KEPT, kind + ".obj")
            if not os.path.exists(model):
                lathe(model, PIECES[kind])
            tall, (x, z), where = PIECES[kind][-1][1], square(file + rank, centre), file + rank
            wide = 2 * PIECES[kind][5][0] + 0.012  # a box round its waist: what stands on the board and what the fingers close on
            made = scene.ed.scene_put(block=f'entity "{NAMES[kind]} {where}"\n  object id 0\n  shape mesh file "{model}" size {tall} {tall} {tall} '
                                            f'position {x} {BOARD_TOP + tall / 2} {z} rotation 0 0 0 color {colour[0]} {colour[1]} {colour[2]}\n'
                                            f'  rigid density 1400 friction 0.9 restitution 0.1\n'
                                            f'  collider kind box fit 0 size {wide} {tall} {wide} offset 0 0 0 rotation 0 0 0\nend\n')
            at[where] = made["id"]
    _board.update(centre=centre, goal=arm[0], at=at)
    _plan.clear()
    show(state=tr("доска готова: ход - «e2e4»"))
    return {"board": [round(centre[0], 3), round(centre[1], 3)], "goal": arm[0], "pieces": len(at)}


@command(label="Ход")
def move(scene, move: str = field("e2e4", label="Ход", help="Два поля, откуда и куда: e2e4; взятие уносит побитую фигуру за доску")):
    """Ход фигурой: план сразу - точки руки, каждая проверена обратной кинематикой; рука делает его в «Пуске»."""
    if not _board["centre"]:
        raise ValueError(tr("сначала доска - «Доска» (chess_board)"))
    steps, after = plan(move, _board["centre"], _board["at"])
    goal = scene.objects[_board["goal"]]
    effector = int(next(line for line in scene.ed.scene_get(id=goal.id)["lines"] if line[:2] == ["goal", "effector"])[2])
    for what, data, words in steps:
        if what == "go":
            asked = scene.ed.assembly_can_reach(effector=effector, position=list(data), quaternion=list(DOWN))
            if not asked.get("reached"):
                raise ValueError(tr("рука не достаёт: {words} ({error} мм мимо)").format(words=words, error=round(1000 * asked.get("position_error", 0), 1)))
    start = _walk["from"] or (tuple(world(scene, goal)) if not _plan else None)
    running = scene.ed.sim_state().get("mode") != "edit"
    if not running:
        _board["before"] = dict(_board["at"])  # Stop puts the pieces back here
    _plan.extend(steps)
    _board["at"] = after
    points = [list(data) for what, data, _ in _plan if what == "go"]
    draw.line("plan", points, colour=PATH)
    if not running:
        scene.ed.sim_play()
    return {"piece": _board["at"][str(move).strip().lower().replace("-", "").replace(" ", "")[2:]],
            "steps": [{"do": words, **({"at": [round(v, 3) for v in data]} if what == "go" else {})} for what, data, words in steps],
            "seconds": round(seconds(steps, start), 1), "queued": len(_plan)}


@command(label="Где фигуры")
def state(scene):
    """Где стоят фигуры (поле - имя фигуры) и что делает рука."""
    names = {o.id: o.name for o in scene.objects}
    goal = scene.objects[_board["goal"]] if _board["goal"] else None
    return {"pieces": {k: names.get(v, v) for k, v in sorted(_board["at"].items())}, "steps_left": len(_plan),
            "hand": goal.gripped if goal else None}


@on_tick
def walk(scene, dt, sensors):
    """The plan's next point when the arm is done with the last: in the scene's own time, so the same on any computer."""
    _walk["t"] += dt
    if not _plan or _walk["t"] < _walk["until"]:
        return {}
    goal = scene.objects[_board["goal"]]
    if _plan[0][0] == "grip" and _plan[0][1] and _walk.get("gripping"):  # closing: wait until they hold, a second at most
        if goal.gripped == "closing" and _walk["t"] < _walk["gripping"] + 1.0:
            return {}
        _walk.pop("gripping")
        _plan.pop(0)
        return {}
    what, data, words = _plan[0]
    show(state=words)
    if what == "go":
        goal.pose(position=data, quaternion=DOWN)
        _walk["until"] = _walk["t"] + (math.dist(_walk["from"], data) / LEAD if _walk["from"] else 1.0) + SETTLE
        _walk["from"] = data
        _plan.pop(0)
    elif data:
        goal.grip(force=FORCE)
        _walk["gripping"], _walk["until"] = _walk["t"], _walk["t"] + HOLD
    else:
        goal.grip(close=False)
        _walk["until"] = _walk["t"] + HOLD
        _plan.pop(0)
    if not _plan:
        draw.clear("plan")
    return {}


@on_stop
def stopped(scene):
    """Stop puts the scene back as it was before ▶: the pieces on their squares again, nothing planned."""
    _plan.clear()
    if "before" in _board:
        _board["at"] = _board.pop("before")
    _walk.update(t=0.0, until=0.0, **{"from": None})
    _walk.pop("gripping", None)
    draw.clear("plan")


ui.panel("Шахматы", ui.section("Партия", ui.button(board, "Доска"), ui.text("move", "e2e4", label="Ход"),
                               ui.row(ui.button(move, "Сделать ход"), ui.button(state, "Где фигуры"))),
         ui.label("Ход - два поля: e2e4. План строится сразу, рука делает его в «Пуске»."),
         ui.value("state", "Рука", "—"))
