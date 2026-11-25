"""«Пульт» (teleop, as ROS' teleop_twist_joy): a person's hand into the scene's controllers.

- A game controller is a device of the scene's controllers (the editor reads only the keys and the mouse): while the
  scene runs, each controller's gamepad bindings (`bind <action> gamepad left_x`, scene_get shows them) are read off the
  pad (gamepad.py, Linux) and sent by controller_input where a value changed - an arm's IK goal, a joint, a car's
  `value` actions steer and throttle (which the brain plugin's stabiliser holds).
- The depz orientation board held like a wheel (`teleop_board`): its tilt from the pose it had when taken - left/right
  turns, forward/back drives - straight to the brain plugin's `brain_car_drive`.

Only the standard library."""
import math
import threading
import time

from kinetica.plugin import command, field, on_play, on_stop, show, tr, ui

import gamepad

PERIOD = 0.05     # s between two commands
DEAD = 4.0        # ° of tilt that do nothing
_run = {"thread": None, "stop": threading.Event()}


def relative(q, zero):
    """The board's turn from its zero pose: zero⁻¹ · q, quaternions (w x y z), unit."""
    w0, x0, y0, z0 = zero[0], -zero[1], -zero[2], -zero[3]
    w, x, y, z = q
    return (w0 * w - x0 * x - y0 * y - z0 * z, w0 * x + x0 * w + y0 * z - z0 * y,
            w0 * y - x0 * z + y0 * w + z0 * x, w0 * z + x0 * y - y0 * x + z0 * w)


def tilt(q):
    """Roll and pitch of a turn (w x y z), degrees (about x and y, z-y-x order)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


def shaped(degrees, full):
    """A tilt as a command -1..1: nothing in the dead zone, then linear to `full` degrees."""
    if abs(degrees) <= DEAD:
        return 0.0
    return max(-1.0, min(1.0, math.copysign(abs(degrees) - DEAD, degrees) / max(full - DEAD, 1.0)))


def command_of(stick_x, stick_y, turn_max, speed_max):
    """Two axes -1..1 (x right, y forward) as the stabiliser's (turn rad/s - positive to the left -, speed m/s); backwards
    at most a third of forward."""
    return -stick_x * turn_max, (stick_y * speed_max if stick_y >= 0 else stick_y * speed_max / 3)


def _drive(scene, chassis, read, turn_max, speed_max):
    stop = _run["stop"]
    while not stop.is_set():
        try:
            x, y = read()
            turn, speed = command_of(x, y, turn_max, speed_max)
            scene.ed.brain_car_drive(body=chassis, turn=round(turn, 3), speed=round(speed, 3))
            show(turn=round(turn, 2), speed=round(speed, 2))
        except Exception as e:  # the board unplugged, the run stopped: say it, keep trying
            show(state=tr("пульт: {e}").format(e=e))
        stop.wait(PERIOD)


def _start(scene, chassis, read, turn_max, speed_max, what):
    _release()
    if scene.ed.sim_state().get("mode") != "playing":  # the stabiliser holds the car in a running world
        scene.ed.sim_play()
    _run["stop"] = threading.Event()
    _run["thread"] = threading.Thread(target=_drive, args=(scene, chassis, read, turn_max, speed_max), daemon=True)
    _run["thread"].start()
    show(state=what)
    return {"driving": True, "source": what}


def _release():
    _run["stop"].set()
    if _run["thread"]:
        _run["thread"].join(timeout=1)
    _run["thread"] = None


@command(label="Пульт: датчик ориентации")
def board(scene, body=field(ref="body", label="Шасси"), full: float = field(30.0, unit="°", range=(10.0, 80.0), label="Полный наклон"),
          turn: float = field(1.0, unit="рад/с", range=(0.1, 4.0), label="Поворот до"),
          speed: float = field(2.0, unit="м/с", range=(0.2, 8.0), label="Скорость до")):
    """The car steered by the depz board: its pose now is the zero; tilt it left/right to turn, forward/back to drive."""
    zero = scene.ed.depz_imu_read().get("quat", {}).get("values")
    if not zero:
        return {"error": tr("нет показаний датчика: подключите его («Подключить датчик ориентации»)")}

    def read():
        q = scene.ed.depz_imu_read()["quat"]["values"]
        roll, pitch = tilt(relative(q, zero))
        return shaped(roll, full), shaped(pitch, full)

    return _start(scene, int(getattr(body, "id", body)), read, turn, speed, tr("датчик ориентации"))


@command(label="Отпустить пульт")
def release(scene):
    """The hand off: the car keeps its last command until the run stops."""
    _release()
    show(state=tr("свободен"))
    return {"driving": False}


def panels(scene):
    """The scene's controllers with gamepad bindings: id -> ({action: its kind}, [(action, controls, invert, toggle)]),
    from their blocks («action <name> <kind> drive ...», «bind <action> gamepad <control>... [invert] [toggle]»)."""
    out = {}
    for c in scene.ed.scene_list(kind="controller")["objects"]:
        kinds, binds = {}, []
        for w in (line.split() for line in scene.ed.scene_get(id=c["id"])["text"].splitlines()):
            if w[:1] == ["action"] and len(w) > 2:
                kinds[w[1]] = w[2]
            elif w[:1] == ["bind"] and len(w) > 3 and w[2] == "gamepad":
                binds.append((w[1], [x for x in w[3:] if x not in ("invert", "toggle")], "invert" in w, "toggle" in w))
        if binds:
            out[c["id"]] = (kinds, binds)
    return out


class Panel:
    """One controller's actions from a pad, as the editor makes them from keys: the bindings of an action summed, a
    toggle flipped by a press; only the changed values sent."""

    def __init__(self, kinds, binds):
        self.kinds, self.binds, self.flipped, self.pressed, self.sent = kinds, binds, {}, {}, {}

    def values(self, pad):
        out = {}
        for action, controls, invert, toggle in self.binds:
            if action not in self.kinds or not controls:
                continue
            v = pad.value(controls[0])
            x, y = v if isinstance(v, tuple) else (v, 0.0)
            if toggle:
                down = x > 0.5
                if down and not self.pressed.get(action):
                    self.flipped[action] = not self.flipped.get(action)
                self.pressed[action], x, y = down, float(self.flipped.get(action, False)), 0.0
            sign = -1.0 if invert else 1.0
            sx, sy = out.get(action, (0.0, 0.0))
            out[action] = (sx + sign * x, sy + sign * y)
        return {a: ([round(max(-1.0, min(1.0, x)), 3), round(max(-1.0, min(1.0, y)), 3)] if self.kinds[a] == "vector2"
                    else round(max(-1.0, min(1.0, x)), 3)) for a, (x, y) in out.items()}

    def changed(self, pad):
        """The values that differ from the last sent; nothing for a still pad at the start."""
        out = {}
        for a, v in self.values(pad).items():
            if self.sent.get(a, [0.0, 0.0] if isinstance(v, list) else 0.0) != v:
                out[a] = self.sent[a] = v
        return out


_pad = {"pad": None, "stop": threading.Event()}


def _serve(scene, found, stop):
    while not stop.wait(0.02):  # 50 Hz, as a pad's report
        pad = _pad["pad"]
        if pad is None or pad.lost:
            path = gamepad.find()
            _pad["pad"] = pad = gamepad.Pad.open(path) if path else None
            show(pad=pad.name if pad else tr("не подключён"))
            if pad is None:
                stop.wait(1.0)
                continue
        for id, panel in found.items():
            values = panel.changed(pad)
            if values:
                try:
                    scene.ed.controller_input(id=id, values=values)
                except Exception as e:  # the run stopping: said, the next frame tries again
                    show(pad=tr("пульт: {e}").format(e=e))


@on_play
def begin(scene):
    """▶: the pad serves the scene's controllers that have gamepad bindings."""
    _pad["stop"].set()
    found = {id: Panel(*p) for id, p in panels(scene).items()}
    if found:
        _pad["stop"] = threading.Event()
        threading.Thread(target=_serve, args=(scene, found, _pad["stop"]), daemon=True).start()


@command(label="Геймпад")
def pad_state(scene):
    """The game controller's state now - its name and each control's value: a check that it is read."""
    pad = _pad["pad"]
    own = pad is None or pad.lost  # not running: opened for this look
    if own:
        path = gamepad.find()
        if not path:
            return {"error": tr("геймпад не найден: подключите его (Linux: /dev/input/js*)")}
        pad = gamepad.Pad.open(path)
        time.sleep(0.1)  # the device says its state on opening
        pad.close()
    return {"name": pad.name, "path": pad.path, "controls": {k: round(v, 3) for k, v in sorted(pad.state.items())}}


@on_stop
def end(scene):
    _release()
    _pad["stop"].set()
    if _pad["pad"]:
        _pad["pad"].close()
    _pad["pad"] = None


ui.panel("Пульт",
         ui.section("Геймпад", ui.value("pad", "Геймпад", "ждёт ▶"), ui.button(pad_state, "Показания")),
         ui.section("Датчик ориентации как руль", ui.value("state", "Ведёт", "свободен"), ui.value("turn", "Поворот, рад/с"),
                    ui.value("speed", "Скорость, м/с"), ui.row(ui.button(board, "Взять"), ui.button(release, "Отпустить"))))
