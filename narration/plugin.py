"""narration - «Рассказ»: what an agent is building, written in the view as it builds it.

`narration_say {text, title, target}` shows a card at the bottom left of the view, as motion design's lower thirds are:
a dark glass card with a thin light edge and an accent bar, over the words a small capitals label «ШАГ 3 · ПЛЕЧО»; it
slides in from the left eased out (0.35 s), the words typed letter by letter, the card before it gone - one card at a
time, what is happening now. `target` - the part the card is about: it lights up in the card's accent (a halo round its
silhouette, draw.glow) and stays lit, softer, while the card is the newest; the next card's part takes the light over.
An agent filmed while it builds says before each step what it does - the viewer reads the step, sees the part, then the
formula (the «Формулы» plugin's callouts). `narration_clear` takes it all off. Drawings, not scene edits: no undo.

The camera directs itself, as a film of a thing being made is shot: each card with a part flies the view to it - eased,
1.6 s, a three-quarter view from a little above, each shot from a new side - and then circles it slowly, a turntable
(view_look glide and orbit: the editor moves the camera, one call a shot); a card about nothing pulls back to the whole.
While the view follows a body (a chase camera behind a car) it is left alone; a press in the view gives it to the person.
`narration_camera {on}` switches the directing off and on. «Свет для съёмки» (`narration_stage`) lights the scene as a
studio does: a warm key and a cool rim from behind, both with shadows, a faint fill, the floor dark.
"""
import math
import textwrap
import threading
import time

from kinetica.plugin import command, draw, field, tr

WIDE, TYPE = 52, 45.0        # letters a line; letters a second - quick enough to keep up, slow enough to be read as typed
SLIDE, IN, WORDS_AT = 0.35, 0.03, 0.2  # the card slides in over SLIDE s from IN of the view's width to the left; words from WORDS_AT s
RISE, SETTLE, HOLD, FADE = 0.25, 0.9, 0.55, 0.5  # the part's light: up in RISE s, down to HOLD by SETTLE s; out in FADE s
WORDS, GLASS, LIGHT = (0.94, 0.96, 1.0), (0.05, 0.07, 0.1, 0.68), (0.35, 0.8, 1.0)  # the light: the card's accent and the part's
PLACE, SIZE = (0.03, 0.95), 19

GLIDE, ORBIT, LIFT, TURN = 1.6, 6.0, 22.0, 35.0  # a shot: s to fly, °/s to circle, ° from above, ° to the next side

_card = {}                   # the card now: text, title, step, born
_camera = {"on": True, "shot": 0, "last": None}
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
    """The card now - its words as typed so far, its label, where it has slid, how much it shows - the parts' lights, and
    whether anything still moves."""
    with _lock:
        card, moving = None, False
        if _card:
            t = now - _card["born"]
            wrapped = textwrap.wrap(_card["text"], WIDE)
            left, typed = max(int((t - WORDS_AT) * TYPE), 0), []
            for line in wrapped:
                if left <= 0:
                    break
                typed.append(line[:left])
                left -= len(line)
            typing = sum(map(len, typed)) < sum(map(len, wrapped))
            if typing:
                typed = (typed or [""])[:-1] + [(typed or [""])[-1] + "▍"]
            slid = _ease(t / SLIDE)
            label = f"{tr('Шаг')} {_card['step']}" + (f" · {_card['title']}" if _card["title"] else "")
            card = {"text": "\n".join(typed) or "▍", "title": label, "place": (PLACE[0] - IN * (1 - slid), PLACE[1]), "opacity": slid}
            moving = typing or t < SLIDE
        lights = {}
        for part, (lit, out) in list(_lit.items()):
            since, gone = now - lit, None if out is None else now - out
            if gone is not None and gone >= FADE:
                lights[part] = 0.0
                del _lit[part]
                continue
            lights[part] = light(since, gone)
            moving = moving or since < SETTLE or gone is not None
        return card, lights, moving


def _show(now):
    card, lights, moving = frame(now)
    if card:
        draw.label("card", (0, 0, 0), card["text"], title=card["title"], colour=WORDS, stroke=LIGHT, size=SIZE, space="screen",
                   place=card["place"], background=GLASS, opacity=card["opacity"])
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
    """The view to the part (or back to the whole after a line about none), then circling it - not while the view follows
    a body (a chase camera behind a car)."""
    if not _camera["on"]:
        return
    view = scene.ed.view_get()
    if view.get("follow_body"):
        return
    if target:
        aim, size = centre(scene, target)
        _camera["last"] = (aim, size)
    elif _camera["last"]:
        aim, size = _camera["last"][0], _camera["last"][1] * 2.5 + 0.4
    else:
        return
    _camera["shot"] += 1
    scene.ed.view_look(eye=shot(view["eye"], aim, size, _camera["shot"]), target=aim, glide=GLIDE, orbit=ORBIT)


@command(label="Сказать")
def say(scene, text: str = field("", label="Что сейчас строю", help="Одна короткая фраза для зрителя: что и зачем"),
        title: str = field("", label="Шаг", help="Одно-два слова над фразой: что за шаг (Основание, Плечо)"),
        target: int = field(0, label="Деталь", help="Номер объекта, о котором карточка: он светится, пока карточка последняя (0 - никакой)")):
    """A card at the bottom left of the view - the step's label, the words typed; the part it is about lights up."""
    text, title = " ".join(str(text).split()), " ".join(str(title).split())
    if not text:
        return {"said": False, "why": "text: пустая строка - нечего сказать"}
    with _lock:
        if _card.get("text") != text or _card.get("title") != title:  # said again at once: the same card
            _card.update(text=text, title=title, step=_card.get("step", 0) + 1, born=time.monotonic())
        for part, (lit, out) in list(_lit.items()):
            if out is None and part != int(target):
                _lit[part] = (lit, time.monotonic())
        if target and (int(target) not in _lit or _lit[int(target)][1] is not None):
            _lit[int(target)] = (time.monotonic(), None)
    _wake()
    if scene is not None:
        direct(scene, int(target))
    return {"said": True, "step": _card["step"]}


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
    """The card and every light off; the steps counted from one again."""
    with _lock:
        _card.clear()
        _lit.clear()
    draw.clear()
    return {"cleared": True}
