"""«Пульт» (teleop, as ROS' teleop_twist_joy): a car of joints driven by hand - by the depz orientation board held like a
wheel, or by a joystick. Each 50 ms a source gives a turn and a speed, and the brain plugin's stabiliser holds them
(`brain_car_drive`: the knuckles by the gyroscope, the driven wheels by an electronic differential).

- The board: its tilt from the pose it had when taken (the zero - hold it as you like) - left/right turns, forward/back
  drives or brakes; a dead zone of a few degrees, full at the tilt of the card's «Полный наклон».
- The joystick: Linux' /dev/input/js0, read by the standard library (8-byte events: time, value, type, number) - the left
  stick's x turns, its y drives (up: forward).

Only the standard library; the turn and speed are the stabiliser's - this plugin only reads a hand."""
import math
import struct
import threading
import time

from kinetica.plugin import command, field, on_stop, show, tr, ui

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


def js_events(data):
    """Linux joystick events in raw bytes (struct js_event: u32 time, s16 value, u8 type, u8 number) as (type, number,
    value/32767); the init flag 0x80 folded away."""
    out = []
    for k in range(0, len(data) - 7, 8):
        _, value, kind, number = struct.unpack("<IhBB", data[k:k + 8])
        out.append((kind & 0x7F, number, value / 32767.0))
    return out


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


@command(label="Пульт: джойстик")
def joystick(scene, body=field(ref="body", label="Шасси"), device: str = field("/dev/input/js0", label="Устройство"),
             turn: float = field(1.0, unit="рад/с", range=(0.1, 4.0), label="Поворот до"),
             speed: float = field(2.0, unit="м/с", range=(0.2, 8.0), label="Скорость до")):
    """The car steered by a joystick (Linux): the left stick's x turns, its y drives."""
    try:
        stream = open(device, "rb", buffering=0)
    except OSError as e:
        return {"error": tr("джойстик не найден ({device}): {e}").format(device=device, e=e)}
    axes = {0: 0.0, 1: 0.0}

    def listen():
        while not _run["stop"].is_set():
            data = stream.read(8)
            if not data:
                break
            for kind, number, value in js_events(data):
                if kind == 2 and number in axes:
                    axes[number] = value

    threading.Thread(target=listen, daemon=True).start()
    return _start(scene, int(getattr(body, "id", body)), lambda: (axes[0], -axes[1]), turn, speed, tr("джойстик"))


@command(label="Отпустить пульт")
def release(scene):
    """The hand off: the car keeps its last command until the run stops."""
    _release()
    show(state=tr("свободен"))
    return {"driving": False}


@on_stop
def end(scene):
    _release()


ui.panel("Пульт",
         ui.section("Вести машинку рукой", ui.value("state", "Ведёт", "свободен"), ui.value("turn", "Поворот, рад/с"),
                    ui.value("speed", "Скорость, м/с"),
                    ui.row(ui.button(board, "Датчик ориентации"), ui.button(joystick, "Джойстик")), ui.button(release, "Отпустить")))
