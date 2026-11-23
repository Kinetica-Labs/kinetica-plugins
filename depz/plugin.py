"""depz.ai devices in the editor (https://depz.ai). For now the orientation board «IMU Sensor BNO086 USB» (fusion up to
400 Hz, accelerometer ±8 g, gyroscope ±2000 °/s): «Подключить» (the plugin's tab of the inspector) finds it, starts the
orientation, the acceleration and the rotation rate, keeps their last samples with the board's own clock in µs, and
gives them to the plots («Графики»);
depz_imu_read gives them to whoever asks - the console, a script, an agent, another plugin. The board's protocol is
bno086.py beside this file, its tests test_depz.py."""
import collections
import math
import os
import threading

from kinetica.plugin import command, field, show, tr, ui

import bno086

KEEP = 4000  # samples kept per stream: 10 s at 400 Hz
ORIENTATION = {"игровая": 0x08, "с компасом": 0x05}  # game rotation vector (no heading drift jumps) or the absolute one
X = {"t": "время с подключения, с"}

_lock = threading.Lock()
_imu = {"board": None, "port": "", "device": "", "streams": {}}


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
    device = board.text(bno086.GET_DEVICE_NAME)
    taken = [r for r in (ORIENTATION[orientation], 0x01, 0x02) if board.feature(r, rate)]
    if not taken:
        board.close()
        raise RuntimeError(tr("{path}: плата не отвечает на команды").format(path=path) + ("" if device else tr(" (это точно BNO086?)")))
    with _lock:
        _imu.update(board=board, port=os.path.realpath(path), device=f"{device} {board.text(bno086.GET_SERIAL)}".strip(), streams={}, taken=taken)
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


ui.panel("depz",
         ui.section("Датчик ориентации BNO086",
                    ui.text("port", "", "Порт"), ui.number("rate", 100, "Частота", "Гц", (1, 400)),
                    ui.choice("orientation", tuple(ORIENTATION), label="Ориентация"),
                    ui.row(ui.button(imu_connect, "Подключить"), ui.button(imu_disconnect, "Отключить")),
                    ui.value("status", "Датчик", "не подключён")),
         ui.chart("Углы", {"roll": "крен", "pitch": "тангаж", "yaw": "рыскание"}, unit="°", x=X),
         ui.chart("Ускорение", {"ax": "x", "ay": "y", "az": "z"}, unit="м/с²", x=X),
         ui.chart("Угловая скорость", {"gx": "x", "gy": "y", "gz": "z"}, unit="°/с", x=X),
         ui.chart("Кватернион", {"qw": "w", "qx": "x", "qy": "y", "qz": "z"}, x=X))
