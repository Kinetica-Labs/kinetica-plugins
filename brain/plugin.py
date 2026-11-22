"""«Мозг»: a neural network as the brain of a robot in the scene - its sensors in, its thrusts or a car's drive out -
trained by natural selection in worlds with no window, then put on the body and played with ▶.

The parts: network.py - the net (layers of any width, a memory of one step, an echo of its own command; its picture
and its .brain.json file); robot.py - what the brain sees and drives; drive.py - a car's drive (tyres, motor, the
steering by the gyroscope, the reflex that turns from a wall); evolution.py - the training (worker processes, one a
CPU core; random tracks; also by hand: python3 evolution.py scene.kscene --tracks 4 --out best.brain.json); scenes.py -
the examples and random tracks.

    select a body → «Мозг на тело» → «Обучить» (the charts by generation, each one's best 20 driving here) →
    «Поставить лучшего» → ▶

The brain reads only its sensors - noise, counts and delays as the devices have them; the judge of the training reads
the physics' truth to score a run, as a laboratory would, and nothing of it reaches the brain."""
import colorsys
import json
import os
import shutil
import tempfile
import threading

from kinetica.plugin import command, component, draw, field, on_play, on_stop, on_tick, show, ui

import drive
import evolution
import network
import scenes
from network import hidden_of, weights_of, weights_text
from robot import Wiring, read_world

TYPE = "brain.net"


@component(label="Мозг", help="Нейросеть: датчики тела - на входе, его тяги - на выходе. Обучается кнопкой «Обучить» панели «Мозг»")
class Net:
    layers = field("8", label="Скрытые слои", help="ширины через запятую: 8 или 16, 8; 0 - без скрытых")
    activation = field("tanh", choices=("tanh", "relu", "sigmoid"), label="Нелинейность")
    recurrent = field(False, label="Память на шаг", help="скрытые слои видят свои значения шага назад: сеть узнаёт скорость по двум положениям")
    echo = field(False, label="Слышит свою команду", help="на входе сети - её выходы шага назад: машина знает скорость, которую задала")
    reflex = field(0.0, range=(0.0, 50.0), label="Рефлекс отворота", help="машина сама отворачивает от ближних стенок (по лидару); "
                   "руль мозга - поправка поверх. 0 - без рефлекса")
    rate_hz = field(30.0, unit="Гц", range=(1.0, 60.0), label="Решений в секунду")
    power = field(15.0, unit="Н", range=(0.0, 10000.0), label="Тяга при выходе 1")
    drive = field("thrusts", choices=("thrusts", "car"), label="Привод", help="thrusts - выходы задают силы тяг тела; car - один выход, руль: "
                  "шины, мотор и руль по гироскопу IMU держат курс и скорость (drive.py)")
    speed = field(4.0, unit="м/с", range=(0.0, 50.0), label="Скорость наибольшая")
    speed_min = field(1.0, unit="м/с", range=(0.0, 50.0), label="Скорость наименьшая")
    turn = field(3.0, unit="рад/с", range=(0.1, 20.0), label="Поворот при выходе 1")
    judge = field("upright", choices=sorted(evolution.JUDGES), label="Судья", help="upright - шест стоит; track - путь по трассе, м")
    seconds = field(8.0, unit="с", range=(0.5, 600.0), label="Прогон")
    weights = field("", label="Веса", help="числа сети через запятую; пусто - мозг ещё не обучен")
    score = field(0.0, label="Счёт обучения")


@component(label="Трасса", help="Средняя линия трассы (x, z по точкам): по ней судья track меряет путь машинки")
class Track:
    points = field("", label="Точки")
    width = field(1.8, unit="м", label="Ширина")


_training = {"thread": None, "stop": None, "best": None, "body": None, "score": None}
_live = {}  # body -> (wiring, net, weights, decide every n frames, frames since the run began)
PARADE, PARADE_SECONDS = 20, 120.0  # the training's best shown side by side, and the longest run of theirs watched


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
        seconds: float = field(8.0, unit="с", range=(0.5, 600.0), label="Прогон"), mode: str = field("thrusts", choices=("thrusts", "car"), label="Привод"),
        speed: float = field(4.0, unit="м/с", range=(0.0, 50.0), label="Скорость машины")):
    """The selected body gets a brain: its sensors (and those of all that hangs on it) in, its thrusts out; the
    thrusts set to 0 - from now on the brain sets them."""
    body = _body(scene)
    wiring = Wiring(scene, body, drive=mode)
    if wiring.car and wiring.gyro is None:
        raise ValueError("машина рулит по гироскопу: поставьте на неё IMU («Датчик ▾ → IMU»)")
    if not wiring.count:
        raise ValueError("у тела нет тяги: добавьте «Сила ▾ → Тяга» на это тело или на то, что на нём держится")
    for id, _ in ([] if wiring.car else wiring.outputs):
        scene.ed.scene_set(id=id, line="effect", key="strength", values=[0])
    scene.ed.component_set(id=body, type=TYPE, values={"layers": str(layers), "activation": "tanh", "recurrent": bool(recurrent),
                                                         "rate_hz": 30.0, "power": float(power), "judge": str(judge), "seconds": float(seconds),
                                                         "drive": str(mode), "speed": float(speed), "speed_min": 1.0, "turn": 3.0, "echo": wiring.car,
                                                         "reflex": 6.0 if wiring.car else 0.0,
                                                         "weights": "", "score": 0.0})
    size = network.Net(wiring.size, hidden_of(layers), wiring.count, "tanh", recurrent, wiring.car).count()  # a car hears its command
    show(status=f"Мозг на теле: {wiring.said()}; весов {size}")
    return {"body": body, "inputs": wiring.size, "outputs": wiring.count, "weights": size, "wiring": wiring.said()}


@command(label="Обучить")
def train(scene, population: int = field(100, range=(4, 2000), label="Особей"), generations: int = field(30, range=(1, 10000), label="Поколений"),
          mutation: float = field(0.2, range=(0.0, 5.0), label="Мутация"), elite: int = field(2, range=(0, 100), label="Лучших без изменений"),
          workers: int = field(0, range=(0, 256), label="Потоков (0 - ядра)"), seed: int = field(1, range=(0, 1000000), label="Зерно"),
          cars: int = field(0, range=(0, 1000), label="Роботов в мире (0 - сам)"),
          tracks: int = field(0, range=(0, 16), label="Ещё случайных трасс (0 - только эта)"),
          watch: bool = field(True, label="Лучший едет на виду")):
    """Natural selection in worlds with no window, one a CPU core: the panel shows each generation; «Поставить
    лучшего» puts the best on the body. It goes on in the background - the editor stays free. `tracks`: random tracks
    (scenes.random_track, easy to very hard) the example's car drives too - the score its worst track: a driver, not
    this track learnt by heart. `watch`: each generation's best 20 drive here, side by side, while
    the next generation is bred - the training seen (its copies: «Убрать стаю»; the card keeps its brain: «Поставить
    лучшего» puts one on)."""
    if _training["thread"] and _training["thread"].is_alive():
        raise ValueError("обучение уже идёт: «Остановить»")
    body = _body(scene)
    brain = _brain(scene, body)
    seconds, judge = float(brain["seconds"]), str(brain["judge"])  # the brain's own task
    folder = tempfile.mkdtemp(prefix="kinetica-brain-")  # this training's scenes and tracks, gone with it
    file = os.path.join(folder, "scene.kscene")
    with open(file, "w", encoding="utf-8") as out:
        out.write(scene.ed.scene_get()["text"])  # the scene as it is now, unsaved edits too; the person's file untouched
    stop, scores = threading.Event(), []

    def report(g, best, mean, clean):
        scores.append(best)
        show(status=f"поколение {g + 1} из {generations}: лучший {best:.2f}, средний {mean:.2f}, чисто {clean:.0%}",
             generation=g + 1, best=round(best, 3), mean=round(mean, 3), clean=round(100 * clean, 1))  # the charts' points

    def keep(g, ranked):  # the generation's best: «Поставить лучшего» puts it on while the training goes on
        _training.update(best=list(ranked[0]), body=body, score=scores[-1])
        if watch:  # and its best drive here, the next time the parade begins a run
            _flock["next"] = (g + 1, ranked)

    def run():
        try:
            def grown(s):
                show(status=f"лучший прожил весь прогон: дальше прогоны по {s:.0f} с")
                _flock["seconds"] = min(s, PARADE_SECONDS)  # the parade's runs grow too, as far as one is watched
            if tracks:
                show(status=f"строю {tracks} случайных трасс…")
            courses = evolution.courses(brain, tracks, seed, folder) if tracks else ()
            weights, score, _ = evolution.train(file, body, brain, population, generations, seconds, mutation, elite, workers, seed, judge,
                                                report, stop, first=weights_of(brain.get("weights", "")), cars=cars, report_horizon=grown,
                                                keep=keep, tracks=courses)
            _training.update(best=weights, body=body, score=score)
            show(status=f"готово: лучший {score:.2f} - «Поставить лучшего», затем ▶")
        except Exception as e:  # the panel says it; the editor goes on
            show(status=f"обучение прервано: {e}")
        finally:
            done.set()
            shutil.rmtree(folder, ignore_errors=True)

    done = threading.Event()
    if watch:
        if _flock.get("thread") and _flock["thread"].is_alive():
            raise ValueError("на трассе уже стая: «Убрать стаю»")
        flock, wirings, shape = _cars(scene, body, brain, PARADE)
        _flock.clear()
        _flock.update(cars=flock, wirings=wirings, nets=[network.Net(*shape) for _ in flock], genomes=None, hues=[0.55] * len(flock), marks=[""] * len(flock),
                      aims=[(0.0, 0.0)] * len(flock), seconds=float(brain["seconds"]), brain=brain, body=body, stop=done, generations=int(generations))
        _flock["thread"] = threading.Thread(target=_parade, args=(scene, done), name="brain-parade", daemon=True)
        _flock["thread"].start()
    _training.update(stop=stop, thread=threading.Thread(target=run, name="brain-train", daemon=True), best=None)
    _training["thread"].start()
    show(status="обучение началось")
    return {"body": body, "started": True}


@command(label="Остановить")
def halt(scene):
    """The training ends after the generation it is in; the best so far is kept."""
    if _training["stop"]:
        _training["stop"].set()
    if _flock.get("stop"):
        _flock["stop"].set()
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


@command(label="Сохранить мозг")
def save(scene, file: str = field("", label="Файл мозга (.brain.json)")):
    """The body's trained brain into a *.brain.json file (network.brain_file): its net, wiring, card and weights - to
    keep it, or to put it on another robot of the same sensors. No file given: <the body's name>.brain.json at home."""
    body = _body(scene)
    card = _brain(scene, body)
    weights = weights_of(card.get("weights", ""))
    if not weights:
        raise ValueError("мозг ещё не обучен: «Обучить», затем «Поставить лучшего»")
    wiring = Wiring(scene, body, drive=card["drive"])
    net = network.Net(wiring.size, hidden_of(card["layers"]), wiring.count, card["activation"], card["recurrent"], card["echo"])
    if net.count() != len(weights):
        raise ValueError(f"веса мозга не по сети ({len(weights)} вместо {net.count()}): обучите заново")
    name = next((o["name"] for o in scene.ed.scene_list(kind="entity")["objects"] if o["id"] == body), f"тело {body}")
    path = os.path.expanduser(file) if file else os.path.join(os.path.expanduser("~"), f"{name}.brain.json")
    path = path if path.endswith(".brain.json") else path + ".brain.json"
    doc = network.brain_file(net, weights, {k: v for k, v in card.items() if k not in ("weights", "score")}, wiring.said(),
                             {"score": float(card.get("score", 0) or 0)})
    with open(path, "w", encoding="utf-8") as out:
        json.dump(doc, out, ensure_ascii=False, indent=1)
    show(status=f"мозг сохранён: {path}")
    return {"file": path, "weights": len(weights)}


@command(label="Загрузить мозг")
def load(scene, file: str = field("", label="Файл мозга (.brain.json)")):
    """A *.brain.json brain onto the selected body - one undo step; its sensors must give the net what it was trained on."""
    if not file:
        raise ValueError("укажите файл мозга (.brain.json)")
    body = _body(scene)
    with open(os.path.expanduser(file), encoding="utf-8") as f:
        doc = network.read_brain_file(f.read())
    card, net = doc["card"], doc["net"]
    wiring = Wiring(scene, body, drive=card.get("drive", "thrusts"))
    if (wiring.size, wiring.count) != (int(net["inputs"]), int(net["outputs"])):
        raise ValueError(f"мозг обучен на {net['inputs']} входов и {net['outputs']} выходов ({doc.get('wiring', '')}), а у тела {wiring.said()}")
    scene.ed.component_set(id=body, type=TYPE, values={**card, "layers": ", ".join(str(n) for n in net["layers"]),
                                                        "weights": weights_text(doc["weights"]), "score": doc.get("trained", {}).get("score", 0)})
    show(status=f"мозг загружен: {os.path.basename(file)} - ▶ смотреть")
    return {"body": body, "weights": len(doc["weights"])}


@command(label="Экзамен: 100 кругов")
def exam(scene, laps: int = field(100, range=(1, 10000), label="Кругов")):
    """The brain on the body alone in a world with no window, driven until it has made `laps` laps or its first error:
    the panel says how many it made. In the background - the editor stays free."""
    body = _body(scene)
    brain = _brain(scene, body)
    if not weights_of(brain.get("weights", "")):
        raise ValueError("мозг ещё не обучен: «Обучить», затем «Поставить лучшего»")
    file = os.path.join(tempfile.gettempdir(), f"kinetica-brain-exam-{os.getpid()}.kscene")
    with open(file, "w", encoding="utf-8") as out:
        out.write(scene.ed.scene_get()["text"])

    def run():
        try:
            made, seconds, finished = evolution.exam(file, body, brain, brain["weights"], laps, brain.get("judge", "track"))
            show(status=(f"экзамен сдан: {laps} кругов без ошибки за {seconds:.0f} с мира" if finished
                         else f"экзамен: {made:.1f} круга до первой ошибки ({seconds:.0f} с мира)"))
        except Exception as e:
            show(status=f"экзамен прерван: {e}")
        finally:
            os.remove(file) if os.path.exists(file) else None

    threading.Thread(target=run, name="brain-exam", daemon=True).start()
    show(status=f"экзамен: {laps} кругов, идёт без окна…")
    return {"body": body, "laps": laps}


@command(label="Пример: балансир")
def balance(scene):
    """A new scene: a cart and a pole on a hinge, with an encoder, an IMU and a lidar, and a brain on the cart."""
    scene.ed.scene_new()
    cart = scenes.balance(scene)
    scene.editor.select(scene.objects[cart])
    return add(scene, "8", False, 15.0, "upright", 8.0, "thrusts", 4.0)


@command(label="Пример: гонка")
def race(scene):
    """A new scene: a closed track - a spline through its points, low walls along it - and a car at its start: a block
    with a lidar on its roof and an IMU, its brain planning a turn and a speed that its drive holds (drive.py). The brain
    comes trained (examples/race.brain.json: five tracks at once, eight it never saw driven clean): ▶ and it drives."""
    scene.ed.scene_new()
    car = scenes.race(scene)
    scene.editor.select(scene.objects[car])
    wired = add(scene, "16, 8", False, 6.0, "track", 30.0, "car", 6.0)
    load(scene, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "race.brain.json"))
    return wired


# ---------------------------------------------------------------------------
# The net drawn: a flat picture in the panel (network.picture)
# ---------------------------------------------------------------------------
def _show_net(net, weights, wiring, card, title):
    """The net in the panel's picture, its neurons lit as they are now; its inputs named, its outputs with their
    values - a car's as the turn and the speed its brain asks for."""
    out = net.seen[-1] if getattr(net, "seen", None) else [0.0] * net.sizes[-1]
    if wiring.car:
        turn, speed = drive.plan(out, card)
        words = [f"поворот {turn:+.1f} рад/с".replace("-", "−"), f"скорость {speed:.1f} м/с"]
        echo = ["эхо: поворот", "эхо: скорость"]
    else:
        words = [f"{name} {v:+.2f}".replace("-", "−") for (_, name), v in zip(wiring.outputs, out)]
        echo = ["эхо: " + name for _, name in wiring.outputs]
    show(net=network.picture(net, weights, wiring.names() + (echo if net.echo else []), words), net_title=title)



# ---------------------------------------------------------------------------
# Training in sight: the flock on the track in the editor's own world
# ---------------------------------------------------------------------------
_flock = {}  # the parade: its cars, their brains, colours and judges, and the run's clock


def _colour(hue):
    return colorsys.hsv_to_rgb(hue % 1.0, 0.9, 1.0)


def _cars(scene, body, brain, n):
    """The body and n - 1 copies of it where it stands, for a flock: each touching the world and not the others, each
    hearing the noise the original does. (cars, their wirings, their nets' shape)."""
    ed = scene.ed
    ed.sim_stop()
    ed.scene_tag(ids=[body], tag=evolution.TAG, on=True)  # on: no `on` switches a tag the scene has off
    ed.scene_set(id=body, line="ignores", values=[evolution.TAG])
    for id, _, kind, _ in Wiring(scene, body).inputs:  # every car hears the noise the original does
        if kind in ("imu", "sonar", "wrench", "lidar"):
            ed.scene_set(id=id, line=kind, key="seed", values=[id])
    cars = [body] + [ed.scene_duplicate(ids=[body])["ids"][0] for _ in range(n - 1)]
    def height(id):
        words = next(w for w in ed.scene_get(id=id)["lines"] if w and w[0] == "shape")
        return float(words[words.index("position") + 2])
    for c in cars[1:]:  # a copy stands above its original: put back where it stands
        ed.scene_move(ids=[c], by=[0, height(body) - height(c), 0])
    world = read_world(scene)
    wirings = [Wiring(scene, c, world, brain.get("drive", "thrusts")) for c in cars]
    return cars, wirings, (wirings[0].size, hidden_of(brain["layers"]), wirings[0].count, brain["activation"], brain["recurrent"], brain["echo"])


def _play(scene):
    """One generation of the flock played in the editor's world: built anew, stepped half a second at a time (no
    faster than the frames are ticked, faster than the clock) until its run is over or it is stopped. Its scores."""
    f, ed = _flock, scene.ed
    f.update(t=0.0, frame=0, look=0.0, judges=None, alive=set(range(len(f["cars"]))), over=threading.Event())
    for net in f["nets"]:
        net.reset()
    draw.clear("flock/")  # the last generation's marks
    ed.sim_stop()
    ed.sim_step(frames=0, state=False)  # built: the play hook makes the judges
    while not f["over"].is_set() and not f["stop"].is_set():
        ed.sim_step(frames=30, state=False)
    scores = [j.score() for j in f["judges"]] if f["judges"] else [0.0] * len(f["cars"])
    ed.sim_stop()
    return scores


def _parade(scene, done):
    """The training's generations in sight: its best of each, side by side on this track - the winner gold, the others
    by their place, a red card grey -, its latest generation played again until the next one comes."""
    f = _flock
    try:
        while not done.is_set():
            if f.get("next"):
                g, ranked = f.pop("next")
                n = len(f["cars"])
                f.update(genomes=[list(w) for w in ranked[:n]], generation=g, title=f"поколение {g}: ", marks=["★"] + [""] * (n - 1),
                         hues=[0.13] + [0.55 + 0.4 * k / n for k in range(1, n)])
            if f.get("genomes"):
                _play(scene)
            else:
                done.wait(1.0)
    except Exception as e:
        show(status=f"парад поколений прерван: {e}")
    finally:
        draw.clear("flock/")


@command(label="Убрать стаю")
def disband(scene):
    """The parade's copies gone - the original car stays; the parade stops, the training goes on."""
    if _flock.get("stop"):
        _flock["stop"].set()
    if _flock.get("thread"):
        _flock["thread"].join(timeout=10)
    cars = _flock.get("cars", [])[1:]
    if cars:
        scene.ed.sim_stop()
        scene.ed.scene_delete(ids=cars)
    _flock.clear()
    draw.clear("flock/")
    return {"removed": len(cars)}


@on_tick(world=True)
def referee(scene, dt, bodies):
    """The physics' side of the brains: every car's tyres and motor (drive.py - mechanics, which read the physics as a
    tyre does); and the flock's judge and colours: every half second each car's way is judged, one ruled out goes
    grey and stops, the marks are drawn over the cars. The physics' truth - the tyres' and the judge's, never a brain's."""
    pushes = {}
    for body, brain in _live.items():
        if brain[0].car and body in bodies:
            car = brain[6]
            car.setdefault("moment", drive.inertia(bodies[body]["mass"], brain[0].footprint))
            pushes[str(body)] = {"force": drive.tyres(bodies[body], car["aim"][1])}
    f = _flock
    if not f.get("cars") or f.get("over") is None or f["over"].is_set():
        return pushes
    for i, w in enumerate(f["wirings"]):
        if w.car and w.body in bodies:
            f.setdefault("moments", {}).setdefault(w.body, drive.inertia(bodies[w.body]["mass"], w.footprint))
            pushes[str(w.body)] = {"force": drive.tyres(bodies[w.body], f["aims"][i][1] if i in f["alive"] else 0.0)}
    if f["judges"] is None:
        f["judges"] = [evolution.JUDGES[f["brain"]["judge"]](scene, w) for w in f["wirings"]]
    f["t"] += dt
    f["look"] += dt
    f["shown"] = f.get("shown", 0.0) + dt
    if f["look"] >= type(f["judges"][0]).every - 1e-9:
        c = {}
        for car in f["cars"]:
            p = bodies.get(car, {}).get("position", [0, 0, 0])
            c[f"{car}/centre x"], c[f"{car}/centre z"] = p[0], p[2]
            c[f"{car}/rotation"] = bodies.get(car, {}).get("rotation", [0, 0, 0])
        for i in sorted(f["alive"]):
            if not f["judges"][i].step(c, f["look"]):
                f["alive"].discard(i)
        f["look"] = 0.0
    if f["shown"] >= 0.5:
        f["shown"] = 0.0
        if f["alive"]:  # the leader: the farthest along of those still in it, its metres and laps over it
            lead = max(f["alive"], key=lambda i: f["judges"][i].score())
            j = f["judges"][lead]
            p = bodies.get(f["cars"][lead], {}).get("position", [0, 0, 0])
            laps = f" · {j.score() / j.length:.1f} круга" if hasattr(j, "length") else ""
            draw.label("flock/leader", [p[0], 1.25, p[2]], f"{f.get('title', '')}♛ №{lead + 1}: {j.score():.0f} м{laps}", colour=_colour(f["hues"][lead]), size=18)
            _show_net(f["nets"][lead], f["genomes"][lead], f["wirings"][lead], f["brain"], f"лидер №{lead + 1}")
    if f["frame"] % 3 == 0:
        for i, car in enumerate(f["cars"]):
            p = bodies.get(car, {}).get("position")
            if not p:
                continue
            alive = i in f["alive"]
            draw.points(f"flock/{i}", [[p[0], 0.55, p[2]]], colour=_colour(f["hues"][i]) if alive else (0.3, 0.3, 0.3), size=16 if alive else 9)
            mark = f["marks"][i] if alive else "✕"
            if mark:
                draw.label(f"flock/mark/{i}", [p[0], 0.85, p[2]], mark, colour=_colour(f["hues"][i]) if alive else (0.5, 0.5, 0.5))
    f["frame"] += 1
    if f["t"] >= f["seconds"] or not f["alive"]:
        f["over"].set()
    return pushes


@on_play
def start(scene):
    """Each trained brain wired to its robot for the run."""
    _live.clear()
    for row in scene.ed.component_list(type=TYPE)["components"]:
        v = {"layers": "8", "activation": "tanh", "recurrent": False, "rate_hz": 30.0, "power": 15.0, "weights": "", "echo": False, **row["values"]}
        weights = weights_of(v["weights"])
        if not weights:
            continue
        wiring = Wiring(scene, row["id"], drive=v.get("drive", "thrusts"))
        net = network.Net(wiring.size, hidden_of(v["layers"]), wiring.count, v["activation"], v["recurrent"] in (True, 1, "1"), v["echo"] in (True, 1, "1"))
        if net.count() != len(weights):
            show(status=f"веса мозга не по сети ({len(weights)} вместо {net.count()}): обучите заново")
            continue
        _live[row["id"]] = [wiring, net, weights, max(1, round(60.0 / float(v["rate_hz"]))), 0, float(v["power"]),
                            {"aim": (0.0, 0.0), "card": v}]


@on_tick
def think(scene, dt, sensors):
    """A frame of the run: every 1 / rate_hz each brain reads its sensors and sets its thrusts (held in between)."""
    commands = {}
    f = _flock
    if f.get("cars") and f.get("over") is not None and not f["over"].is_set():
        decide = f["frame"] % max(1, round(60.0 / float(f["brain"]["rate_hz"]))) == 0
        for i, w in enumerate(f["wirings"]):
            going = i in f["alive"]
            if w.car:
                if decide:
                    x = w.numbers(sensors)
                    f["aims"][i] = drive.plan(f["nets"][i].forward(f["genomes"][i], x), f["brain"], w.away(x)) if going else (0.0, 0.0)
                commands[str(w.body)] = {"torque": _steer(w, sensors, f["aims"][i][0], f.get("moments", {}).get(w.body))}
            elif decide:
                outputs = f["nets"][i].forward(f["genomes"][i], w.numbers(sensors)) if going else [0.0] * w.count
                commands.update(w.commands(outputs, f["brain"]["power"]))
        return commands
    for body, brain in _live.items():
        wiring, net, weights, every, frame, power, car = brain
        if frame % every == 0:
            x = wiring.numbers(sensors)
            outputs = net.forward(weights, x)
            if wiring.car:
                car["aim"] = drive.plan(outputs, car["card"], wiring.away(x))
            else:
                commands.update(wiring.commands(outputs, power))
            if frame % (every * 6) == 0 and body == min(_live):
                _show_net(net, weights, wiring, car["card"] if wiring.car else {}, "мозг на теле")
        if wiring.car:
            commands[str(body)] = {"torque": _steer(wiring, sensors, car["aim"][0], car.get("moment"))}
        brain[4] = frame + 1
    return commands


def _steer(wiring, sensors, turn, moment):
    """A car's steering torque by its gyroscope (drive.py); its moment known from the physics hook's first frame."""
    gyro = (sensors.get(wiring.gyro) or {}).get("gyro", [0.0, 0.0, 0.0])[2]
    return drive.steering(gyro, turn, moment or 0.0)


@on_stop
def end(scene):
    _live.clear()


ui.panel("Мозг",
         ui.section("Робот", ui.button(add, "Мозг на тело"), ui.row(ui.button(balance, "Пример: балансир"), ui.button(race, "Пример: гонка"))),
         ui.section("Обучение", ui.value("status", "Ход", "выберите тело с тягой и датчиками"),
                    ui.chart("Эволюция", {"best": "лучший", "mean": "средний"}, x={"generation": "поколение"}),
                    ui.chart("Чистые заезды", {"clean": "доехали весь прогон"}, unit="%", x={"generation": "поколение"}),
                    ui.row(ui.button(train, "Обучить"), ui.button(halt, "Остановить")),
                    ui.row(ui.button(apply, "Поставить лучшего"), ui.button(exam, "Экзамен: 100 кругов"), ui.button(disband, "Убрать стаю"))),
         ui.section("Сеть", ui.value("net_title", "Чья"), ui.svg("net", height=600), ui.text("file", "", "Файл мозга (.brain.json)"),
                    ui.row(ui.button(save, "Сохранить мозг"), ui.button(load, "Загрузить мозг"))))
