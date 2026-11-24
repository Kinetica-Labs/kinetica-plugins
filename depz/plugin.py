"""depz.ai devices in the editor (https://depz.ai). For now the orientation board «IMU Sensor BNO086 USB» (fusion up to
400 Hz, accelerometer ±8 g, gyroscope ±2000 °/s): «Подключить» (the plugin's tab of the inspector) finds it, starts the
orientation, the acceleration and the rotation rate, keeps their last samples with the board's own clock in µs, and
gives them to the plots («Графики»);
depz_imu_read gives them to whoever asks - the console, a script, an agent, another plugin. The board's protocol is
bno086.py beside this file, its tests test_depz.py.

«Рука за датчиком» (`depz_arm`) lets the board lead the arm while a clutch is held - R the wrist, F the base, the
shoulder and the elbow: the board's turn since the press, split into its three angles, each added to its own joint
(hand.py); Space closes the hand's fingers and opens them again;
the rest of the arm stands, no inverse kinematics of the editor, no goal. «Записать движение» keeps the board's orientation in a file of JSON lines and
«Проиграть запись» plays it back as if the board moved: a take filmed again, the arm tried without the board."""
import collections
import json
import math
import os
import threading
import time

from kinetica.plugin import command, field, on_quit, on_tick, show, tr, ui

import bno086
import hand

KEEP = 4000  # samples kept per stream: 10 s at 400 Hz
TAKES = os.path.join(os.path.expanduser("~"), "Kinetica", "depz")  # recorded motions, a file of JSON lines each
ORIENTATION = {"игровая": 0x08, "с компасом": 0x05}  # game rotation vector (no heading drift jumps) or the absolute one
X = {"t": "время с подключения, с"}

_lock = threading.Lock()
_imu = {"board": None, "port": "", "device": "", "streams": {}}
_lead = {}  # what the board leads, by clutch: "wrist", "arm" - joints, rest, the board's axes, zero, angles (none: nothing)
_grip = {}  # the hand's fingers: (joint, open, closed) each, and the panel whose Space closes them
SIGN = 1.0  # a joint's coordinate grows as body a turns about the axis this way round
_playing = {"stop": threading.Event()}


def _sample(stream, t_us, values, accuracy):
    with _lock:
        _imu["streams"].setdefault(stream, collections.deque(maxlen=KEEP)).append((t_us, values, accuracy))


def _rate(samples):
    """A stream's rate over its kept samples, Hz."""
    return (len(samples) - 1) * 1e6 / (samples[-1][0] - samples[0][0]) if len(samples) > 1 and samples[-1][0] > samples[0][0] else 0.0


def _status(board):
    words = {"quat": tr("ориентация"), "accel": tr("ускорение"), "gyro": tr("угл. скорость")}
    with _lock:
        rates = " · ".join(f"{words[s]} {_rate(d):.0f} {tr('Гц')}" for s, d in _imu["streams"].items())
        t = max((d[-1][0] for d in _imu["streams"].values()), default=0)
    return tr("{device} на {port} · {rates} · потеряно пакетов {lost} · часы платы {clock:.1f} с").format(
        device=_imu["device"], port=_imu["port"], rates=rates, lost=board.lost, clock=t / 1e6)


def _chart(board, hz=30):
    """The plots' lines: the newest sample of each stream, `hz` times a second, on the board's clock; the words
    once a second. Ends with the board; unplugged, says so."""
    shown, tick, t0 = None, 0, None
    while not board.stopped.wait(1 / hz):
        with _lock:
            latest = {s: d[-1] for s, d in _imu["streams"].items() if d}
        t = max((v[0] for v in latest.values()), default=None)
        if t is None or t == shown:
            continue
        shown, tick, t0 = t, tick + 1, t if t0 is None else t0
        values = {"t": (t - t0) / 1e6}
        if "quat" in latest:
            q = latest["quat"][1]
            values.update(zip(("qw", "qx", "qy", "qz"), q))
            values.update(zip(("roll", "pitch", "yaw"), map(math.degrees, bno086.euler(q))))
        if "accel" in latest:
            values.update(zip(("ax", "ay", "az"), latest["accel"][1]))
        if "gyro" in latest:
            values.update(zip(("gx", "gy", "gz"), map(math.degrees, latest["gyro"][1])))
        if tick % hz == 1:
            values["status"] = _status(board)
        show(**values)
    if _imu["board"] is board:
        show(status=tr("связь с платой на {port} потеряна").format(port=_imu["port"]))


@command(label="Подключить датчик ориентации")
def imu_connect(scene, port: str = field("", label="Порт", help="пусто - найти плату самому; или /dev/ttyACM0, COM5"),
                rate: int = field(100, unit="Гц", range=(1, 400), label="Частота"),
                orientation: str = field("игровая", choices=tuple(ORIENTATION), label="Ориентация",
                                         help="игровая - без компаса, курс плавно уходит; с компасом - курс на север, железо рядом его сбивает")):
    """Connect the BNO086 board and start its orientation, acceleration and rotation rate at `rate` Hz."""
    imu_disconnect(scene)
    path = port or bno086.find_port()
    if not path:
        raise RuntimeError(tr("плата BNO086 не найдена: подключите её по USB или впишите порт"))
    board = bno086.Board(path, _sample)
    try:  # the port is open: whatever fails before the board is kept closes it
        device = board.text(bno086.GET_DEVICE_NAME)
        taken = [r for r in (ORIENTATION[orientation], 0x01, 0x02) if board.feature(r, rate)]
        if not taken:
            raise RuntimeError(tr("{path}: плата не отвечает на команды").format(path=path) + ("" if device else tr(" (это точно BNO086?)")))
        serial = board.text(bno086.GET_SERIAL)
    except BaseException:
        board.close()
        raise
    with _lock:
        _imu.update(board=board, port=os.path.realpath(path), device=f"{device} {serial}".strip(), streams={}, taken=taken)
    threading.Thread(target=_chart, args=(board,), name="depz-chart", daemon=True).start()
    show(status=tr("{device} на {port}: жду отсчётов…").format(device=_imu["device"], port=_imu["port"]))
    return {"port": _imu["port"], "device": _imu["device"], "streams": len(taken)}


def _release():
    """The board's reports stopped and its port closed: a board left streaming into a port no one reads is not left."""
    board, _imu["board"] = _imu["board"], None
    if not board:
        return False
    if not board.stopped.is_set():
        for r in _imu.get("taken", ()):
            board.feature(r, 0)
    board.close()
    return True


@command(label="Отключить датчик")
def imu_disconnect(scene):
    """Stop the board's reports and close its port; the kept samples stay for depz_imu_read."""
    if _release():
        show(status=tr("не подключён"))
    return {"connected": False}


@on_quit
def _quitting(scene):
    """The editor closes or restarts the plugin: the board stopped before the process ends."""
    _playing["stop"].set()
    _release()


@command(label="Показания датчика")
def imu_read(scene, last: int = field(0, range=(0, KEEP), label="Последних отсчётов")):
    """The newest sample of each stream - quat (w x y z), accel (m/s²), gyro (rad/s), each with the board's time t_us
    and its accuracy 0..3 - the angles roll, pitch, yaw in degrees, the rates; with `last`, that many samples of each."""
    board = _imu["board"]
    with _lock:
        streams = {s: list(d) for s, d in _imu["streams"].items()}
    out = {"connected": bool(board and not board.stopped.is_set()), "port": _imu["port"], "device": _imu["device"],
           "lost": board.lost if board else 0}
    for s, d in streams.items():
        if d:
            t, v, a = d[-1]
            out[s] = {"t_us": t, "values": v, "accuracy": a, "rate_hz": round(_rate(d), 1)}
            if last:
                out[s]["last"] = [[t, *v, a] for t, v, a in d[-last:]]
    if "quat" in out:
        out["euler_deg"] = dict(zip(("roll", "pitch", "yaw"), map(math.degrees, bno086.euler(out["quat"]["values"]))))
    return out


def _latest_quat():
    with _lock:
        d = _imu["streams"].get("quat")
        return d[-1][1] if d else None


def _servos(scene):
    """The arm's servo hinges from its base out (joint ids in the scene's order) and the coordinates they hold now."""
    joints, rest = [], []
    for o in sorted((o for o in scene.objects if o.kind == "joint"), key=lambda o: o.id):
        lines = scene.ed.scene_get(id=o.id)["lines"]
        drive = next((line for line in lines if line[:2] == ["drive", "rx"]), None)
        if drive:  # a servo turning about its rx: a hinge of the arm (a limited one too), not a finger's slider
            joints.append(o.id)
            rest.append(float(drive[drive.index("target") + 1]) if "target" in drive else 0.0)
    return joints, rest


def _fingers(scene):
    """The hand's fingers - the servo sliders (drive tx) - and how far each closes: (joint, open, closed) - its limits."""
    out = []
    for o in sorted((o for o in scene.objects if o.kind == "joint"), key=lambda o: o.id):
        lines = scene.ed.scene_get(id=o.id)["lines"]
        limit = next((line for line in lines if line[:2] == ["limit", "tx"]), None)
        if any(line[:2] == ["drive", "tx"] for line in lines) and limit:
            out.append((o.id, float(limit[limit.index("lower") + 1]), float(limit[limit.index("upper") + 1])))
    return out


def _world_turn(scene, id):
    """An object's turn in the world as the scene file has it: its own rotation under its parents' (the arm at rest)."""
    q = (1.0, 0.0, 0.0, 0.0)
    while id:
        lines = scene.ed.scene_get(id=id)["lines"]
        pose = next((line for line in lines if line[0] in ("shape", "pose") and "rotation" in line), None)
        if pose:
            k = pose.index("rotation")
            q = hand.mul(hand.euler([float(v) for v in pose[k + 1:k + 4]]), q)
        head = next(line for line in lines if line[0] == "object")
        id = int(head[head.index("parent") + 1]) if "parent" in head else 0
    return q


def _axis(scene, joint):
    """A hinge's axis in the world, the arm at rest: its frame in body a (frame_a), that body as the scene has it."""
    link = next(line for line in scene.ed.scene_get(id=joint)["lines"] if line[0] == "link")
    a, k = int(link[link.index("a") + 1]), link.index("frame_a")
    return hand.rotate(hand.mul(_world_turn(scene, a), hand.euler([float(v) for v in link[k + 1:k + 4]])), (1.0, 0.0, 0.0))


CLUTCH = "Сцепление depz"  # the scene's panel: R lets the board lead the wrist while held, F the big joints, Space the fingers
GROUPS = (("wrist", "R"), ("arm", "F"))  # the panel's buttons: which joints they let the board lead, and their keys
BUTTONS = GROUPS + (("grip", "Space toggle"),)  # and the fingers' button: pressed - closed, pressed again - open


def _clutch(scene):
    """The panel with the two clutches' buttons - R the wrist, F the big joints - the scene's own, made once (one undo step)."""
    found = next((o for o in scene.objects if o.kind == "controller" and o.name == CLUTCH), None)
    if found and all(["action", name] == line[:2] for name, _ in BUTTONS
                     for line in [next((l for l in scene.ed.scene_get(id=found.id)["lines"] if l[:2] == ["action", name]), [])]):
        return found.id
    if found:  # an older panel, with other buttons: made anew
        scene.ed.scene_delete(ids=[found.id])
    block = "".join(f"  action {name} button drive value\n  bind {name} keys {key}\n" for name, key in BUTTONS)
    return scene.ed.scene_put(block=f'controller "{CLUTCH}"\n  object id 0\n{block}end\n')["id"]


@command(label="Рука за датчиком")
def arm(scene, on: bool = field(True, label="Вести руку"),
        turn: int = field(0, choices=(0, 90, 180, 270), unit="°", label="Плата повёрнута",
                          help="Как держите плату: 0 - ось X вправо, Y к экрану, чип вверх; иначе - на столько повёрнута вокруг Z")):
    """The board leads the arm while a clutch is held, as a surgical robot's pedal: R - the wrist's three joints (the
    hand turns), F - the big three, the base, the shoulder, the elbow (the hand moves). At the press the board's turn and
    the joints' angles are caught; then each of the board's angles since the press turns its own joint (each board axis
    leads the joint it lies nearest in this view, held X to the right, Y toward the screen, chip up); let go - the arm
    stands and the board can be taken anew. ▶ starts if the scene is still."""
    if not on:
        _lead.clear(), _grip.clear()
        show(status=tr("рука отпущена"))
        return {"leading": False}
    if _latest_quat() is None:
        raise RuntimeError(tr("нет показаний датчика: подключите его или проиграйте запись"))
    joints, rest = _servos(scene)
    if len(joints) < 6:
        raise RuntimeError(tr("в сцене нет руки с запястьем: нужны шесть шарниров с сервоприводами"))
    view = scene.ed.view_get()
    mount = hand.held([t - e for e, t in zip(view["eye"], view["target"])], turn)
    clutch = _clutch(scene)
    _lead.clear()
    for (name, _), part in zip(GROUPS, (slice(3, 6), slice(0, 3))):
        ids = joints[part]
        _lead[name] = {"joints": ids, "rest": rest[part], "led": hand.match(mount, [_axis(scene, j) for j in ids]), "zero": None,
                       "base": [0.0] * 3, "split": [0.0] * 3, "angles": [0.0] * 3, "shown": list(rest[part]), "clutch": clutch}
    _grip.update(fingers=_fingers(scene), clutch=clutch)
    if scene.ed.sim_state().get("mode") == "edit":
        scene.ed.sim_play()
    show(status=tr("зажмите R - запястье, F - плечи и локоть; Пробел - схват"))
    return {"leading": True, "clutch": clutch, "joints": {name: g["joints"] for name, g in _lead.items()},
            "fingers": [f[0] for f in _grip["fingers"]]}


def _held(sensors, name, clutch):
    """Whether a clutch's button is held now: its panel's values, as the run gives them to a controller."""
    panel = sensors.get(clutch) or sensors.get(str(clutch)) or {}
    return float((panel.get("values") or {}).get(name, 0) or 0) > 0.5


@on_tick
def lead(scene, dt, sensors):
    """This frame's targets of the arm's servos: for each clutch held, each joint's angle at the press plus the board's
    turn since the press about the board's axis that leads it; smoothed."""
    q = _latest_quat() if _lead else None
    if q is None:
        return {}
    out = {}
    for name, g in _lead.items():
        if _held(sensors, name, g["clutch"]):
            if g["zero"] is None:  # the press: the board's turn and the joints' angles caught here
                g["zero"], g["base"], g["split"] = q, list(g["angles"]), [0.0] * 3
            # the board's turn since the press (zero⁻¹ · q), split into turns about its three axes, each added to its joint
            g["split"] = hand.solve(hand.mul(hand.conj(g["zero"]), q), g["led"], g["split"])
            g["angles"] = [b + a for b, a in zip(g["base"], g["split"])]
        else:
            g["zero"] = None
        g["shown"] = hand.smooth(g["shown"], [r + SIGN * a for r, a in zip(g["rest"], g["angles"])], dt)
        out.update({j: {"target": round(v, 5)} for j, v in zip(g["joints"], g["shown"])})
    closed = _held(sensors, "grip", _grip.get("clutch"))  # Space: pressed - the fingers close to their limit (they stop on what they hold)
    out.update({j: {"target": shut if closed else open_} for j, open_, shut in _grip.get("fingers", ())})
    return out


@command(label="Записать движение")
def record(scene, seconds: float = field(10.0, unit="с", range=(1.0, 120.0), label="Сколько"),
           name: str = field("движение", label="Имя записи")):
    """The board's orientation for `seconds`, kept in ~/Kinetica/depz/<name>.jsonl - a sample a line, its time on the board's clock."""
    if _latest_quat() is None:
        raise RuntimeError(tr("нет показаний датчика: подключите его"))
    os.makedirs(TAKES, exist_ok=True)
    path = os.path.join(TAKES, f"{name}.jsonl")

    def keep():
        seen, end = None, time.monotonic() + seconds
        with open(path, "w", encoding="utf-8") as out:
            while time.monotonic() < end:
                with _lock:
                    d = _imu["streams"].get("quat")
                    fresh = [x for x in d if seen is None or x[0] > seen] if d else []
                for t, q, a in fresh:
                    out.write(json.dumps({"t_us": t, "q": list(q)}) + "\n")
                    seen = t
                time.sleep(0.05)
        show(status=tr("запись «{name}» готова").format(name=name))

    threading.Thread(target=keep, name="depz-record", daemon=True).start()
    return {"file": path, "seconds": seconds}


@command(label="Проиграть запись")
def play(scene, name: str = field("движение", label="Имя записи или путь к файлу"), loop: bool = field(False, label="По кругу")):
    """A recorded motion played back as if the board moved: its samples into the orientation stream at their own pace."""
    path = name if os.path.isabs(name) else os.path.join(TAKES, f"{name}.jsonl")
    with open(path, encoding="utf-8") as kept:
        samples = [json.loads(line) for line in kept if line.strip()]
    if not samples:
        raise ValueError(tr("в записи нет отсчётов"))
    _playing["stop"].set()
    stop = _playing["stop"] = threading.Event()

    def feed():
        while not stop.is_set():
            start, first = time.monotonic(), samples[0]["t_us"]
            for k, sample in enumerate(samples):
                wait = start + (sample["t_us"] - first) / 1e6 - time.monotonic()
                if stop.wait(max(0.0, wait)):
                    return
                _sample("quat", sample["t_us"], tuple(sample["q"]), 3)
            if not loop:
                return

    with _lock:
        _imu["streams"]["quat"] = collections.deque(maxlen=KEEP)
    _sample("quat", samples[0]["t_us"], tuple(samples[0]["q"]), 3)
    threading.Thread(target=feed, name="depz-play", daemon=True).start()
    return {"samples": len(samples), "seconds": round((samples[-1]["t_us"] - samples[0]["t_us"]) / 1e6, 2)}


ui.panel("depz",
         ui.section("Датчик ориентации BNO086",
                    ui.text("port", "", "Порт"), ui.number("rate", 100, "Частота", "Гц", (1, 400)),
                    ui.choice("orientation", tuple(ORIENTATION), label="Ориентация"),
                    ui.row(ui.button(imu_connect, "Подключить"), ui.button(imu_disconnect, "Отключить")),
                    ui.value("status", "Датчик", "не подключён")),
         ui.section("Рука за датчиком", ui.label("Плата: X вправо, Y к экрану, чип вверх. «Взять руку», потом зажмите R - запястье, F - плечи и локоть; отпустите - рука стоит. Пробел - сжать или разжать схват."),
                    ui.row(ui.button(arm, "Взять руку"), ui.button(record, "Записать"), ui.button(play, "Проиграть"))),
         ui.chart("Углы", {"roll": "крен", "pitch": "тангаж", "yaw": "рыскание"}, unit="°", x=X),
         ui.chart("Ускорение", {"ax": "x", "ay": "y", "az": "z"}, unit="м/с²", x=X),
         ui.chart("Угловая скорость", {"gx": "x", "gy": "y", "gz": "z"}, unit="°/с", x=X),
         ui.chart("Кватернион", {"qw": "w", "qx": "x", "qy": "y", "qz": "z"}, x=X))
