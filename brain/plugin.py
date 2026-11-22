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

from kinetica.plugin import command, component, draw, field, on_play, on_stop, on_tick, show, ui

import evolution
import network
import scenes
from network import hidden_of, weights_of, weights_text
from robot import Wiring, read_world

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
    width = field(1.8, unit="м", label="Ширина")


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


# ---------------------------------------------------------------------------
# Training in sight: the flock on the track in the editor's own world
# ---------------------------------------------------------------------------
_flock = {}  # while it trains in sight: its cars, their brains, colours and judges, and the generation's clock


def _mix(a, b, rng, radiation):
    """A child's colour: its parents' mixed, nudged by as much as its genes were (radiation's mark)."""
    return tuple(max(0.05, min(1.0, (x + y) / 2 + rng.gauss(0.0, 0.35 * radiation))) for x, y in zip(a, b))


@command(label="Обучать на виду")
def live(scene, population: int = field(30, range=(2, 200), label="Машинок на трассе"), generations: int = field(50, range=(1, 10000), label="Поколений"),
         mutation: float = field(0.2, range=(0.0, 5.0), label="Радиация мутации"), elite: int = field(3, range=(0, 50), label="Лучших без изменений"),
         seed: int = field(1, range=(0, 1000000), label="Зерно")):
    """Natural selection you watch: a flock of copies of the selected robot in the editor's own world, each with its
    brain, driven every frame and judged twice a second; one ruled out goes grey and stops; a generation over, the best
    live on unchanged, the rest are born of two parents picked by tournament - a child's ring the mix of its parents'
    colours, nudged by its mutation. Ctrl+Z after it - the flock gone."""
    if _flock.get("thread") and _flock["thread"].is_alive():
        raise ValueError("стая уже учится: «Остановить»")
    body = _body(scene)
    brain = _brain(scene, body)
    ed = scene.ed
    ed.sim_stop()
    ed.scene_tag(ids=[body], tag=evolution.TAG)
    ed.scene_set(id=body, line="ignores", values=[evolution.TAG])
    for id, _, kind, _ in Wiring(scene, body).inputs:  # every car hears the noise the original does
        if kind in ("imu", "sonar", "wrench", "lidar"):
            ed.scene_set(id=id, line=kind, key="seed", values=[id])
    cars = [body] + [ed.scene_duplicate(ids=[body])["ids"][0] for _ in range(int(population) - 1)]
    def height(id):
        words = next(w for w in ed.scene_get(id=id)["lines"] if w and w[0] == "shape")
        return float(words[words.index("position") + 2])
    for c in cars[1:]:  # a copy stands above its original: put back where it stands
        ed.scene_move(ids=[c], by=[0, height(body) - height(c), 0])
    world = read_world(scene)
    wirings = [Wiring(scene, c, world) for c in cars]
    shape = (wirings[0].size, hidden_of(brain["layers"]), len(wirings[0].outputs), brain["activation"], brain["recurrent"])
    rng = network.new_rng(seed)
    genomes = [network.Net(*shape).random_weights(rng) for _ in cars]
    first = weights_of(brain.get("weights", ""))
    if len(first) == network.Net(*shape).count():
        genomes[0] = first
    colours = [(rng.random(), rng.random(), rng.random()) for _ in cars]
    _flock.clear()
    _flock.update(cars=cars, wirings=wirings, nets=[network.Net(*shape) for _ in cars], genomes=genomes, colours=colours, rng=rng,
                  brain=brain, body=body, generations=int(generations), mutation=float(mutation), elite=int(elite), stop=threading.Event())
    _flock["thread"] = threading.Thread(target=_flock_run, args=(scene,), name="brain-flock", daemon=True)
    _flock["thread"].start()
    show(status=f"стая из {len(cars)} машинок учится на виду", radiation=f"σ {float(mutation):.2f}")
    return {"cars": cars}


def _flock_run(scene):
    """The generations, played one after another in the editor's world: built paused, stepped half a second at a time
    (no faster than the frames are ticked, faster than the clock), the judges' scores read, the next generation bred."""
    f, ed = _flock, scene.ed
    seconds, scores_seen = float(f["brain"]["seconds"]), []
    try:
        for g in range(f["generations"]):
            f.update(t=0.0, frame=0, look=0.0, judges=None, alive=set(range(len(f["cars"]))), generation=g, over=threading.Event())
            for net in f["nets"]:
                net.reset()
            ed.sim_stop()
            ed.sim_step(frames=0, state=False)  # built: the play hook makes the judges
            while not f["over"].is_set() and not f["stop"].is_set():
                ed.sim_step(frames=30, state=False)
            scores = [j.score() for j in f["judges"]] if f["judges"] else [0.0] * len(f["cars"])
            ed.sim_stop()
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            best = order[0]
            if not _training.get("score") or scores[best] > _training["score"]:
                _training.update(best=list(f["genomes"][best]), body=f["body"], score=scores[best])
            scores_seen.append(scores[best])
            if f["stop"].is_set():
                break
            rng, m = f["rng"], f["mutation"]
            genomes, colours, born = [list(f["genomes"][i]) for i in order[:f["elite"]]], [f["colours"][i] for i in order[:f["elite"]]], {}
            while len(genomes) < len(f["cars"]):
                a, b = evolution._pick(rng, scores), evolution._pick(rng, scores)
                genomes.append(evolution._child(rng, f["genomes"][a], f["genomes"][b], m))
                colours.append(_mix(f["colours"][a], f["colours"][b], rng, m))
                for p in {a, b}:
                    born[p] = born.get(p, 0) + 1
            f.update(genomes=genomes, colours=colours)
            top = max(scores_seen) or 1.0
            parents = sorted(born.items(), key=lambda kv: -kv[1])[:3]
            show(status=f"поколение {g + 1} из {f['generations']}: лучший {scores[best]:.1f} м, средний {sum(scores) / len(scores):.1f} м, "
                        f"выжило до конца {sum(1 for j in f['judges'] if j.idle < 3.0)} из {len(scores)}",
                 curve="".join(BARS[min(len(BARS) - 1, int(len(BARS) * max(s, 0.0) / top))] for s in scores_seen[-40:]),
                 family="; ".join(f"№{p + 1} ({scores[p]:.0f} м) - {n} детей" for p, n in parents), radiation=f"σ {m:.2f}")
        show(status=f"стая выучилась: лучший {_training.get('score', 0):.1f} м - «Поставить лучшего», затем «Убрать стаю»")
    except Exception as e:
        show(status=f"обучение на виду прервано: {e}")
    finally:
        draw.clear("flock/")


@command(label="Убрать стаю")
def disband(scene):
    """The flock's copies gone - the original car stays; the training stopped first."""
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
    """The flock's judge and colours: every half second each car's way is judged; one ruled out goes grey; every third
    frame the rings are drawn round the cars. The physics' truth - the judge's, never a brain's."""
    f = _flock
    if not f.get("cars") or f.get("over") is None or f["over"].is_set():
        return {}
    if f["judges"] is None:
        f["judges"] = [evolution.JUDGES[f["brain"]["judge"]](scene, w) for w in f["wirings"]]
    f["t"] += dt
    f["look"] += dt
    if f["look"] >= 0.5:
        c = {}
        for car in f["cars"]:
            p = bodies.get(car, {}).get("position", [0, 0, 0])
            c[f"{car}/centre x"], c[f"{car}/centre z"] = p[0], p[2]
        for i in sorted(f["alive"]):
            if not f["judges"][i].step(c, f["look"]):
                f["alive"].discard(i)
        f["look"] = 0.0
    if f["frame"] % 3 == 0:
        for i, car in enumerate(f["cars"]):
            p = bodies.get(car, {}).get("position")
            if p:
                draw.circle(f"flock/{i}", centre=[p[0], 0.02, p[2]], radius=0.4, colour=f["colours"][i] if i in f["alive"] else (0.35, 0.35, 0.35))
    f["frame"] += 1
    if f["t"] >= float(f["brain"]["seconds"]) or not f["alive"]:
        f["over"].set()
    return {}


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
    f = _flock
    if f.get("cars") and f.get("over") is not None and not f["over"].is_set():
        if f["frame"] % max(1, round(60.0 / float(f["brain"]["rate_hz"]))) == 0:
            for i, w in enumerate(f["wirings"]):
                outputs = f["nets"][i].forward(f["genomes"][i], w.numbers(sensors)) if i in f["alive"] else [0.0] * len(w.outputs)
                commands.update(w.commands(outputs, f["brain"]["power"]))
        return commands
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
                    ui.value("family", "Кто дал потомство"), ui.value("radiation", "Радиация мутации"),
                    ui.row(ui.button(train, "Обучить"), ui.button(live, "Обучать на виду"), ui.button(halt, "Остановить")),
                    ui.row(ui.button(apply, "Поставить лучшего"), ui.button(disband, "Убрать стаю"))))
