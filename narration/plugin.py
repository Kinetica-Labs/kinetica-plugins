"""narration - «Рассказ»: what an agent is building, written in the view as it builds it.

`narration_say {text, target}` puts a line at the bottom of the view, typed letter by letter on a dark plate as
subtitles are (WebVTT's cue box): the last phrases stay whole, as many as four lines hold, older ones go up and out. `target` - the part the line
is about: it lights up (a halo round its silhouette, draw.glow) and stays lit, softer, while the line is the newest;
the next line's part takes the light over. An agent filmed while it builds says before each step what it does - the
viewer reads the plan, sees the part, then the formula (the «Формулы» plugin's callouts). `narration_clear` takes it
all off. Drawings, not scene edits: no undo, nothing saved.

The camera directs itself, as a film of a thing being made is shot: each line with a part flies the view to it - eased,
1.6 s, a three-quarter view from a little above, each shot from a new side - and then circles it slowly, a turntable
(view_look glide and orbit: the editor moves the camera, one call a shot); a line about nothing pulls back to the whole.
While the scene runs the view is the run's (a chase camera follows a car); a press in the view gives it to the person.
`narration_camera {on}` switches the directing off and on. «Свет для съёмки» (`narration_stage`) lights the scene as a
studio does: a warm key and a cool rim from behind, both with shadows, a faint fill, the floor dark.
"""
import math
import textwrap
import threading
import time

from kinetica.plugin import command, draw, field, on_play, on_stop

PHRASES, MOST, WIDE = 3, 4, 58  # phrases kept, lines shown at most (whole phrases, the newest always), letters a line
TYPE = 45.0                  # letters a second: quick enough to keep up, slow enough to be read as typed
RISE, SETTLE, HOLD, FADE = 0.25, 0.9, 0.55, 0.5  # the part's light: up in RISE s, down to HOLD by SETTLE s; out in FADE s
WORDS, PLATE, LIGHT = (0.93, 0.95, 1.0), (0.04, 0.06, 0.09, 0.74), (0.35, 0.8, 1.0)
PLACE, SIZE = (0.025, 0.955), 19

GLIDE, ORBIT, LIFT, TURN = 1.6, 6.0, 22.0, 35.0  # a shot: s to fly, °/s to circle, ° from above, ° to the next side

_said = []                   # [(phrase, born)]: what was said, the newest last
_camera = {"on": True, "playing": False, "shot": 0, "last": None}
_lit = {}                    # part -> (lit at, put out at or None)
_lock = threading.Lock()
_animating = {"thread": None}


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def light(since, out=None, now=None):
    """A part's light `since` seconds after it was lit, put out `out` seconds ago (None: still the newest's)."""
    up = _ease(since / RISE) if since < RISE else 1 - (1 - HOLD) * _ease((since - RISE) / (SETTLE - RISE))
    return up * (1 - _ease(out / FADE)) if out is not None else up


def frame(now):
    """The plate's text now and whether anything still moves - typing, a light rising or going out."""
    with _lock:
        blocks, moving = [], False
        for k, (phrase, born) in enumerate(_said):
            wrapped = textwrap.wrap(phrase, WIDE)
            if k == len(_said) - 1:  # the newest, typed letter by letter across its lines
                left, typed = max(int((now - born) * TYPE), 0), []
                for line in wrapped:
                    if left <= 0:
                        break
                    typed.append(line[:left])
                    left -= len(line)
                if sum(map(len, typed)) < sum(map(len, wrapped)):
                    typed = (typed or [""])[:-1] + [(typed or [""])[-1] + "▍"]
                    moving = True
                wrapped = typed
            blocks.append(wrapped)
        lines = blocks.pop() if blocks else []
        while blocks and len(lines) + len(blocks[-1]) <= MOST:  # whole phrases above it, as many as fit
            lines = blocks.pop() + lines
        lights = {}
        for part, (lit, out) in list(_lit.items()):
            since, gone = now - lit, None if out is None else now - out
            if gone is not None and gone >= FADE:
                lights[part] = 0.0
                del _lit[part]
                continue
            lights[part] = light(since, gone)
            moving = moving or since < SETTLE or gone is not None
        return "\n".join(lines), lights, moving


def _show(now):
    text, lights, moving = frame(now)
    if text:
        draw.label("lines", (0, 0, 0), text, colour=WORDS, size=SIZE, space="screen", place=PLACE, background=PLATE)
    for part, strength in lights.items():
        if strength > 0:
            draw.glow(f"part/{part}", part, strength, colour=LIGHT)
        else:
            draw.clear(f"part/{part}")
    return moving


def _animate():
    try:
        while _show(time.monotonic()):
            time.sleep(1 / 30)
    finally:
        _animating["thread"] = None


def _wake():
    if _animating["thread"] is None:
        _animating["thread"] = threading.Thread(target=_animate, daemon=True)
        _animating["thread"].start()


def centre(scene, id):
    """Where a part is (its parents' places added, as the editor nests them) and how big - a joint: its moving body's."""
    part = scene.objects[int(id)]
    if part.kind == "joint":
        link = next(line for line in scene.ed.scene_get(id=int(id))["lines"] if line[0] == "link")
        part = scene.objects[int(link[link.index("a") + 1])]
    at, size, up = list(part.position), max(part.size) if part.kind == "entity" else 0.3, getattr(part, "parent", None)
    while up is not None:
        at = [a + b for a, b in zip(at, up.position)] if hasattr(up, "position") and up.position else at
        up = getattr(up, "parent", None)
    return at, size


def shot(eye, target, size, k):
    """The k-th shot's eye: the side the view looks from now, turned TURN° on for each shot, LIFT° from above, far enough to
    see the part and what it stands on."""
    side = math.atan2(eye[2] - target[2], eye[0] - target[0]) + math.radians(TURN) * (1 if k % 2 == 0 else -0.6)
    far, lift = min(max(0.9 + 1.8 * size, 1.0), 3.2), math.radians(LIFT)
    return [target[0] + far * math.cos(lift) * math.cos(side), target[1] + far * math.sin(lift), target[2] + far * math.cos(lift) * math.sin(side)]


def direct(scene, target):
    """The view to the part (or back to the whole after a line about none), then circling it."""
    if not _camera["on"] or _camera["playing"]:
        return
    view = scene.ed.view_get()
    if target:
        aim, size = centre(scene, target)
        _camera["last"] = (aim, size)
    elif _camera["last"]:
        aim, size = _camera["last"][0], _camera["last"][1] * 2.5 + 0.4
    else:
        return
    _camera["shot"] += 1
    scene.ed.view_look(eye=shot(view["eye"], aim, size, _camera["shot"]), target=aim, glide=GLIDE, orbit=ORBIT)


@on_play
def _running(scene):
    _camera["playing"] = True


@on_stop
def _stopped(scene):
    _camera["playing"] = False


@command(label="Сказать")
def say(scene, text: str = field("", label="Что сейчас строю", help="Одна короткая фраза для зрителя: что и зачем"),
        target: int = field(0, label="Деталь", help="Номер объекта, о котором строка: он светится, пока строка последняя (0 - никакой)")):
    """A line at the bottom of the view, typed letter by letter; the part it is about lights up."""
    text = " ".join(str(text).split())
    if not text:
        return {"said": False, "why": "text: пустая строка - нечего сказать"}
    with _lock:
        _said.append((text, time.monotonic()))
        del _said[:-PHRASES]
        for part, (lit, out) in list(_lit.items()):
            if out is None:
                _lit[part] = (lit, time.monotonic())
        if target:
            _lit[int(target)] = (time.monotonic(), None)
    _wake()
    if scene is not None:
        direct(scene, int(target))
    return {"said": True, "lines": len(_said)}


@command(label="Камера ведёт")
def camera(scene, on: bool = field(True, label="Вести камеру", help="Строка о детали подводит к ней камеру и облетает её; щелчок в виде отдаёт камеру вам")):
    """The camera directs itself with the lines, or not."""
    _camera["on"] = bool(on)
    return {"camera": _camera["on"]}


@command(label="Свет для съёмки")
def stage(scene):
    """A studio's light: a warm key from the front left and a cool rim from behind, both with shadows, a faint fill; the floor
    dark - what is built stands in the light."""
    lights = []
    for kind, position, colour, intensity, shadows in (("spot", (-1.8, 2.6, 1.9), (1.0, 0.88, 0.7), 3.2, True),
                                                       ("spot", (1.6, 2.2, -2.2), (0.45, 0.65, 1.0), 2.6, True),
                                                       ("sun", (1.0, 4.0, 2.0), (0.8, 0.86, 1.0), 0.25, False)):
        light = scene.add_light(kind, position=position, target=(0.0, 0.3, 0.0))
        scene.ed.scene_set(id=light.id, line="pose", key="color", values=list(colour))
        light.intensity, light.shadows = intensity, shadows
        lights.append(light.id)
    floor = next((o for o in scene.objects if o.name == "Пол"), None)
    if floor is not None:
        scene.ed.scene_set(id=floor.id, line="shape", key="color", values=[0.16, 0.17, 0.21])
    return {"lights": lights}


@command(label="Убрать рассказ")
def clear(scene):
    """The plate and every light off."""
    with _lock:
        _said.clear()
        _lit.clear()
    draw.clear()
    return {"cleared": True}
