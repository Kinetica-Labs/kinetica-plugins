"""replay - «Kinetica Replay»: a session of the editor's journal played back as a film of its building.

The journal (the editor's ops-journal.jsonl, or a film's: examples/films/<name>.jsonl of the editor) holds every edit a
session made, by a person, a script or an agent. `replay_play` takes the session's scene back to its first snapshot
(journal_restore) and does its edits again one by one with a pause - the core's edits, as the restore replays them, so
the same ids come out and each step builds on the last - each with a card of the «Рассказ» plugin saying the step in
words, the part it made lit and the camera flown to it; where the session pressed Play, the film plays the scene as long
as it ran, then stops. Every step is an edit of this window: one undo each. `replay_stop` ends a film half way.
Games call this a replay: the commands recorded, done again, the same world comes out. Only the standard library."""
import threading
import time
from datetime import datetime

from kinetica.client import OperationError
from kinetica.plugin import command, field

SHAPES = {"box": "брусок", "sphere": "шар", "cylinder": "цилиндр", "cone": "конус", "plane": "плоскость", "mesh": "модель"}
JOINTS = {"hinge": "Шарнир", "weld": "Сварка", "slider": "Ползун", "ball": "Шаровой сустав", "cylinder": "Цилиндрический сустав",
          "universal": "Кардан", "planar": "Плоский сустав", "shock": "Амортизатор"}
KEYS = {"rotation": "поворот", "position": "место", "size": "размер", "color": "цвет", "density": "плотность", "mass": "масса"}
_run = {"stop": threading.Event(), "thread": None}


def seconds(t):
    """A journal line's time (ISO, UTC) in seconds."""
    return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def plan(lines, edits):
    """The film of a session's lines: (start, steps, run) - the line of its first snapshot, the steps after it as
    (line, op, args) of the core's edits that went well (`edits`: their names) and Play where the session pressed it
    ("sim_play", its seconds), as the restore replays them: what was done while the scene played is left out."""
    start = next((k for k, l in enumerate(lines) if l.get("op") == "journal_snapshot"), None)
    if start is None:
        return None, [], 0.0
    steps, playing, began = [], False, None
    for k in range(start + 1, len(lines)):
        l = lines[k]
        op = l.get("op", "")
        if op == "sim_play" and l.get("ok"):
            playing, began = True, l.get("t")
            continue
        if op == "sim_stop" and playing:
            playing = False
            steps.append((k, "sim_play", {"seconds": round(seconds(l["t"]) - seconds(began), 1) if began else 0.0}))
            continue
        if not playing and l.get("ok") and op in edits and l.get("src") != "journal":
            steps.append((k, op, l.get("args", {})))
    return start, steps, sum(s[2]["seconds"] for s in steps if s[1] == "sim_play")


def words(op, args, names):
    """A step said for the viewer: (title, text, the object it is about - 0: none). names: id -> name in the scene."""
    named = lambda key: names.get(int(args.get(key) or 0), "")  # noqa: E731
    if op == "scene_add":
        name = args.get("name") or SHAPES.get(args.get("shape"), args.get("shape", ""))
        return name, f"Ставлю {SHAPES.get(args.get('shape'), 'фигуру')} «{name}»", 0
    if op == "scene_set":
        values = " ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in args.get("values", []))
        return named("id"), f"«{named('id')}»: {KEYS.get(args.get('key'), args.get('key', ''))} {values}".strip(), int(args.get("id") or 0)
    if op == "assembly_joint":
        kind = JOINTS.get(args.get("type", "hinge"), "Сустав")
        return kind, f"{kind}: «{named('a')}» к «{named('b')}»", int(args.get("a") or 0)
    if op == "assembly_add":
        return args.get("name", ""), f"Готовая сборка «{args.get('name', '')}»", 0
    if op == "scene_delete":
        return "Убираю", "Убираю " + ", ".join(f"«{names.get(int(i), i)}»" for i in args.get("ids", [args.get("id")]) if i), 0
    if op == "sim_play":
        return "Пуск", "Смотрим, что получилось", 0
    target = int(args.get("id") or args.get("target") or 0)
    return names.get(target, ""), op.replace("_", " "), target


def _say(scene, title, text, target):
    try:
        scene.ed.narration_say(text=text, title=title, target=target)
    except OperationError:  # no «Рассказ»: the film goes on without cards
        print(f"{title}: {text}")


def _film(scene, file, session, steps, pause, run, stop):
    try:
        for k, op, args in steps:
            if stop.is_set():
                return
            names = {o["id"]: o.get("name", "") for o in scene.ed.scene_list()["objects"]}
            title, text, target = words(op, args, names)
            if op == "sim_play":
                if run <= 0:
                    continue
                _say(scene, title, text, 0)
                scene.ed.sim_play()
                stop.wait(min(run, args["seconds"] or run))
                scene.ed.sim_stop()
                continue
            answer = getattr(scene.ed, op)(**args)
            made = int(answer.get("id") or 0) if op == "scene_add" else 0
            _say(scene, title, text, target or made)
            stop.wait(pause)
        _say(scene, "Готово", f"{len(steps)} шагов сеанса - повтор окончен", 0)
    except OperationError as e:
        _say(scene, "Повтор прерван", str(e), 0)


@command(label="Повтор сеанса")
def play(scene, file: str = field("pendulum-tower.jsonl", label="Фильм", help="Журнал фильма из examples/films редактора; пусто - журнал этого окна"),
         session: str = field("", label="Сеанс", help="Номер сеанса (journal_sessions); пусто - последний, где что-то строили"),
         pause: float = field(2.0, unit="с", range=(0.2, 10.0), label="Пауза", help="Сколько стоит каждый шаг: камера успевает подлететь"),
         run: float = field(6.0, unit="с", range=(0.0, 60.0), label="Пуск", help="Сколько секунд играть сцену, где сеанс нажимал «Пуск»; 0 - не играть")):
    """A session of the journal played back step by step as a film: its first scene, then each edit with a card."""
    where = {"file": file} if file else {}
    if not session:
        current = scene.ed.journal_sessions().get("current") if not file else None
        built = [s for s in scene.ed.journal_sessions(**where)["sessions"] if s["edits"] > 0 and s["session"] != current]
        if not built:
            return {"playing": False, "why": "в журнале нет сеанса, где что-то строили"}
        session = built[-1]["session"]
    lines = scene.ed.journal_lines(session=session, **where)["lines"]
    edits = {o["name"] for o in scene.ed.ops_list()["ops"] if o.get("kind") == "edit"}
    start, steps, played = plan(lines, edits)
    if start is None:
        return {"playing": False, "why": f"у сеанса {session} нет снимка сцены - начать не с чего"}
    stop()
    scene.ed.journal_restore(session=session, line=start, **where)
    _run["stop"] = threading.Event()
    _run["thread"] = threading.Thread(target=_film, args=(scene, file, session, steps, pause, run, _run["stop"]), daemon=True)
    _run["thread"].start()
    return {"playing": True, "session": session, "steps": len(steps), "seconds": round(len(steps) * pause + min(run, played), 1)}


@command(label="Стоп повтора")
def stop(scene=None):
    """The film ends where it is: the scene stays as far as it got."""
    _run["stop"].set()
    if _run["thread"] and _run["thread"] is not threading.current_thread():
        _run["thread"].join(timeout=3)
    return {"stopped": True}
