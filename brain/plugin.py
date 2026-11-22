"""«Мозг»: a neural network as the brain of a robot in the scene - its sensors in, its thrusts out - trained by natural
selection in worlds with no window, then put on the body and played with ▶.

The parts: network.py - the net (layers of any width, a memory of one step if asked); robot.py - what the brain sees
and drives (the body, what hangs on it by joints, their sensors and thrusts); evolution.py - the training (a pool of
kinetica-core worlds, one a CPU core; also run by hand: python3 evolution.py scene.kscene); scenes.py - the examples.

    select a body → «Мозг на тело» → «Обучить» (the panel shows each generation) → «Поставить лучшего» → ▶

The brain reads only its sensors - noise, counts and delays as the devices have them; the judge of the training reads
the physics' truth to score a run, as a laboratory would, and nothing of it reaches the brain."""
import os
import tempfile
import threading

from kinetica.plugin import command, component, field, on_play, on_stop, on_tick, show, ui

import evolution
import network
import scenes
from network import hidden_of, weights_of, weights_text
from robot import Wiring

TYPE = "brain.net"
BARS = "▁▂▃▄▅▆▇█"


@component(label="Мозг", help="Нейросеть: датчики тела - на входе, его тяги - на выходе. Обучается кнопкой «Обучить» панели «Мозг»")
class Net:
    layers = field("8", label="Скрытые слои", help="ширины через запятую: 8 или 16, 8; 0 - без скрытых")
    activation = field("tanh", choices=("tanh", "relu", "sigmoid"), label="Нелинейность")
    recurrent = field(False, label="Память на шаг", help="скрытые слои видят свои значения шага назад: сеть узнаёт скорость по двум положениям")
    rate_hz = field(30.0, unit="Гц", range=(1.0, 60.0), label="Решений в секунду")
    power = field(15.0, unit="Н", range=(0.0, 10000.0), label="Тяга при выходе 1")
    judge = field("upright", choices=sorted(evolution.JUDGES), label="Судья", help="upright - шест стоит; track - путь по трассе, м")
    seconds = field(8.0, unit="с", range=(0.5, 600.0), label="Прогон")
    weights = field("", label="Веса", help="числа сети через запятую; пусто - мозг ещё не обучен")
    score = field(0.0, label="Счёт обучения")


@component(label="Трасса", help="Средняя линия трассы (x, z по точкам): по ней судья track меряет путь машинки")
class Track:
    points = field("", label="Точки")


_training = {"thread": None, "stop": None, "best": None, "body": None, "score": None}
_live = {}  # body -> (wiring, net, weights, decide every n frames, frames since the run began)


def _body(scene):
    """The selected body, else the only one with a brain."""
    chosen = [o for o in scene.editor.selection if o.kind == "entity"]
    if chosen:
        return chosen[-1].id
    rows = scene.ed.component_list(type=TYPE)["components"]
    if len(rows) == 1:
        return rows[0]["id"]
    raise ValueError("выберите тело" + (" с мозгом" if rows else ""))


def _brain(scene, body):
    _, values = evolution.brain_of(scene, body)
    return values


@command(label="Мозг на тело")
def add(scene, layers: str = field("8", label="Скрытые слои"), recurrent: bool = field(False, label="Память на шаг"),
        power: float = field(15.0, unit="Н", label="Тяга при выходе 1"), judge: str = field("upright", choices=sorted(evolution.JUDGES), label="Судья"),
        seconds: float = field(8.0, unit="с", range=(0.5, 600.0), label="Прогон")):
    """The selected body gets a brain: its sensors (and those of all that hangs on it) in, its thrusts out; the
    thrusts set to 0 - from now on the brain sets them."""
    body = _body(scene)
    wiring = Wiring(scene, body)
    if not wiring.outputs:
        raise ValueError("у тела нет тяги: добавьте «Сила ▾ → Тяга» на это тело или на то, что на нём держится")
    for id, _ in wiring.outputs:
        scene.ed.scene_set(id=id, line="effect", key="strength", values=[0])
    scene.ed.component_set(id=body, type=TYPE, values={"layers": str(layers), "activation": "tanh", "recurrent": bool(recurrent),
                                                         "rate_hz": 30.0, "power": float(power), "judge": str(judge), "seconds": float(seconds),
                                                         "weights": "", "score": 0.0})
    size = network.Net(wiring.size, hidden_of(layers), len(wiring.outputs), "tanh", recurrent).count()
    show(status=f"Мозг на теле: {wiring.said()}; весов {size}")
    return {"body": body, "inputs": wiring.size, "outputs": len(wiring.outputs), "weights": size, "wiring": wiring.said()}


@command(label="Обучить")
def train(scene, population: int = field(100, range=(4, 2000), label="Особей"), generations: int = field(30, range=(1, 10000), label="Поколений"),
          mutation: float = field(0.2, range=(0.0, 5.0), label="Мутация"), elite: int = field(2, range=(0, 100), label="Лучших без изменений"),
          workers: int = field(0, range=(0, 256), label="Потоков (0 - ядра)"), seed: int = field(1, range=(0, 1000000), label="Зерно"),
          cars: int = field(0, range=(0, 1000), label="Роботов в мире (0 - сам)")):
    """Natural selection in worlds with no window, one a CPU core: the panel shows each generation; «Поставить
    лучшего» puts the best on the body. It goes on in the background - the editor stays free."""
    if _training["thread"] and _training["thread"].is_alive():
        raise ValueError("обучение уже идёт: «Остановить»")
    body = _body(scene)
    brain = _brain(scene, body)
    seconds, judge = float(brain["seconds"]), str(brain["judge"])  # the brain's own task
    file = os.path.join(tempfile.gettempdir(), f"kinetica-brain-{os.getpid()}.kscene")
    with open(file, "w", encoding="utf-8") as out:
        out.write(scene.ed.scene_get()["text"])  # the scene as it is now, unsaved edits too; the person's file untouched
    stop, scores = threading.Event(), []

    def report(g, best, mean):
        scores.append(best)
        top = max(scores) or 1.0
        curve = "".join(BARS[min(len(BARS) - 1, int(len(BARS) * max(s, 0.0) / top))] for s in scores[-40:])
        show(status=f"поколение {g + 1} из {generations}: лучший {best:.2f}, средний {mean:.2f}", curve=curve)

    def run():
        try:
            weights, score, _ = evolution.train(file, body, brain, population, generations, seconds, mutation, elite, workers, seed, judge,
                                                report, stop, first=weights_of(brain.get("weights", "")), cars=cars)
            _training.update(best=weights, body=body, score=score)
            show(status=f"готово: лучший {score:.2f} - «Поставить лучшего», затем ▶")
        except Exception as e:  # the panel says it; the editor goes on
            show(status=f"обучение прервано: {e}")
        finally:
            os.remove(file) if os.path.exists(file) else None

    _training.update(stop=stop, thread=threading.Thread(target=run, name="brain-train", daemon=True), best=None)
    _training["thread"].start()
    show(status="обучение началось")
    return {"body": body, "started": True}


@command(label="Остановить")
def halt(scene):
    """The training ends after the generation it is in; the best so far is kept."""
    if _training["stop"]:
        _training["stop"].set()
    return {"stopping": bool(_training["thread"] and _training["thread"].is_alive())}


@command(label="Поставить лучшего")
def apply(scene):
    """The training's best weights into the body's brain - one undo step."""
    if not _training["best"]:
        raise ValueError("лучшего ещё нет: «Обучить» и дождитесь конца (или «Остановить»)")
    body = _training["body"]
    values = {**_brain(scene, body), "weights": weights_text(_training["best"]), "score": round(float(_training["score"]), 4)}
    scene.ed.component_set(id=body, type=TYPE, values=values)
    show(status=f"мозг обучен (счёт {_training['score']:.2f}): ▶ - смотреть")
    return {"body": body, "weights": len(_training["best"])}


@command(label="Пример: балансир")
def balance(scene):
    """A new scene: a cart and a pole on a hinge, with an encoder, an IMU and a lidar, and a brain on the cart."""
    scene.ed.scene_new()
    cart = scenes.balance(scene)
    scene.editor.select(scene.objects[cart])
    return add(scene, "8", False, 15.0, "upright", 8.0)


@command(label="Пример: гонка")
def race(scene):
    """A new scene: a closed track - a spline through its points, low walls along it - and a car at its start: a
    sliding block with two thrusts at its back (their difference steers), a lidar's fan of 9 rays ahead, an IMU, and a
    brain judged by the metres it makes along the track in 20 s."""
    scene.ed.scene_new()
    car = scenes.race(scene)
    scene.editor.select(scene.objects[car])
    return add(scene, "12", False, 6.0, "track", 20.0)


@on_play
def start(scene):
    """Each trained brain wired to its robot for the run."""
    _live.clear()
    for row in scene.ed.component_list(type=TYPE)["components"]:
        v = {"layers": "8", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 15.0, "weights": "", **row["values"]}
        weights = weights_of(v["weights"])
        if not weights:
            continue
        wiring = Wiring(scene, row["id"])
        net = network.Net(wiring.size, hidden_of(v["layers"]), len(wiring.outputs), v["activation"], v["recurrent"] in (True, 1, "1"))
        if net.count() != len(weights):
            show(status=f"веса мозга не по сети ({len(weights)} вместо {net.count()}): обучите заново")
            continue
        _live[row["id"]] = [wiring, net, weights, max(1, round(60.0 / float(v["rate_hz"]))), 0, float(v["power"])]


@on_tick
def think(scene, dt, sensors):
    """A frame of the run: every 1 / rate_hz each brain reads its sensors and sets its thrusts (held in between)."""
    commands = {}
    for body, brain in _live.items():
        wiring, net, weights, every, frame, power = brain
        if frame % every == 0:
            commands.update(wiring.commands(net.forward(weights, wiring.numbers(sensors)), power))
        brain[4] = frame + 1
    return commands


@on_stop
def end(scene):
    _live.clear()


ui.panel("Мозг",
         ui.section("Робот", ui.button(add, "Мозг на тело"), ui.row(ui.button(balance, "Пример: балансир"), ui.button(race, "Пример: гонка"))),
         ui.section("Обучение", ui.value("status", "Ход", "выберите тело с тягой и датчиками"), ui.value("curve", "Лучший по поколениям"),
                    ui.row(ui.button(train, "Обучить"), ui.button(halt, "Остановить")), ui.button(apply, "Поставить лучшего")))
