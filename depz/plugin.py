"""depz.ai devices in the editor (https://depz.ai). For now the orientation board «IMU Sensor BNO086 USB» (fusion up to
400 Hz, accelerometer ±8 g, gyroscope ±2000 °/s): «Подключить» (the plugin's tab of the inspector) finds it, starts the
orientation, the acceleration and the rotation rate, keeps their last samples with the board's own clock in µs, and
gives them to the plots («Графики»);
depz_imu_read gives them to whoever asks - the console, a script, an agent, another plugin. The board's protocol is
bno086.py beside this file, its tests test_depz.py.

«Рука за датчиком» (`depz_arm`) makes the board an arm's puppet strings while the scene plays: its turn from the pose it
had when taken - heading to the base, tilt to the shoulder and the elbow, twist to the wrist (hand.py) - led to the
arm's servos each frame, smoothed. «Записать движение» keeps the board's orientation in a file of JSON lines and
«Проиграть запись» plays it back as if the board moved: a take filmed again, the arm tried without the board."""
import collections
import json
import math
import os
import threading
import time

from kinetica.plugin import command, field, on_tick, show, tr, ui

import bno086
import hand

KEEP = 4000  # samples kept per stream: 10 s at 400 Hz
TAKES = os.path.join(os.path.expanduser("~"), "Kinetica", "depz")  # recorded motions, a file of JSON lines each
ORIENTATION = {"игровая": 0x08, "с компасом": 0x05}  # game rotation vector (no heading drift jumps) or the absolute one
X = {"t": "время с подключения, с"}

_lock = threading.Lock()
_imu = {"board": None, "port": "", "device": "", "streams": {}}
_arm = {"joints": [], "rest": [], "zero": None, "shown": None, "flip": ()}  # the arm the board leads (none: it leads nothing)
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


@command(label="Отключить датчик")
def imu_disconnect(scene):
    """Stop the board's reports and close its port; the kept samples stay for depz_imu_read."""
    board = _imu["board"]
    if not board:
        return {"connected": False}
    _imu["board"] = None
    if not board.stopped.is_set():
        for r in _imu.get("taken", ()):
            board.feature(r, 0)
    board.close()
    show(status=tr("не подключён"))
    return {"connected": False}


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


@command(label="Рука за датчиком")
def arm(scene, on: bool = field(True, label="Вести руку"),
        flip: str = field("", label="Обратить", help="углы, что идут наоборот, через запятую: yaw (поворот), pitch (наклон), roll (вращение)")):
    """The board leads the arm while the scene plays: turn - the base, tilt - the shoulder and the elbow, twist - the
    wrist. The pose the board has now is the arm's pose now; ▶ starts if the scene is not running."""
    if not on:
        _arm.update(joints=[], zero=None)
        show(status=tr("рука отпущена"))
        return {"leading": False}
    q = _latest_quat()
    if q is None:
        raise RuntimeError(tr("нет показаний датчика: подключите его или проиграйте запись"))
    joints, rest = _servos(scene)
    if len(joints) < 2:
        raise RuntimeError(tr("в сцене нет руки: нужны шарниры с сервоприводами"))
    flips = tuple(w.strip() for w in flip.split(",") if w.strip())
    if set(flips) - {"yaw", "pitch", "roll"}:
        raise ValueError(tr("обратить можно yaw, pitch, roll"))
    _arm.update(joints=joints, rest=rest, zero=q, shown=list(rest), flip=flips)
    if scene.ed.sim_state().get("mode") == "edit":
        scene.ed.sim_play()
    show(status=tr("датчик ведёт руку: {n} суставов").format(n=len(joints)))
    return {"leading": True, "joints": joints}


@on_tick
def lead(scene, dt, sensors):
    """This frame's targets of the arm's servos from the board's turn, smoothed."""
    q = _latest_quat() if _arm["joints"] else None
    if q is None:
        return {}
    wanted = hand.targets(hand.angles(hand.relative(q, _arm["zero"])), _arm["rest"], _arm["flip"])
    _arm["shown"] = hand.smooth(_arm["shown"], wanted, dt)
    return {j: {"target": round(v, 5)} for j, v in zip(_arm["joints"], _arm["shown"])}


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
         ui.section("Рука за датчиком", ui.text("flip", "", "Обратить"),
                    ui.row(ui.button(arm, "Взять руку"), ui.button(record, "Записать"), ui.button(play, "Проиграть"))),
         ui.chart("Углы", {"roll": "крен", "pitch": "тангаж", "yaw": "рыскание"}, unit="°", x=X),
         ui.chart("Ускорение", {"ax": "x", "ay": "y", "az": "z"}, unit="м/с²", x=X),
         ui.chart("Угловая скорость", {"gx": "x", "gy": "y", "gz": "z"}, unit="°/с", x=X),
         ui.chart("Кватернион", {"qw": "w", "qx": "x", "qy": "y", "qz": "z"}, x=X))
