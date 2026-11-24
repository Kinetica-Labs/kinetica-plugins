"""narration - «Рассказ»: what an agent is building, written in the view as it builds it.

`narration_say {text, target}` puts a line at the bottom of the view, typed letter by letter on a dark plate as
subtitles are (WebVTT's cue box): the last three lines stay, older ones go up and out. `target` - the part the line
is about: it lights up (a halo round its silhouette, draw.glow) and stays lit, softer, while the line is the newest;
the next line's part takes the light over. An agent filmed while it builds says before each step what it does - the
viewer reads the plan, sees the part, then the formula (the «Формулы» plugin's callouts). `narration_clear` takes it
all off. Drawings, not scene edits: no undo, nothing saved.
"""
import textwrap
import threading
import time

from kinetica.plugin import command, draw, field

LINES, WIDE = 3, 58          # lines kept on the plate, letters a line
TYPE = 45.0                  # letters a second: quick enough to keep up, slow enough to be read as typed
RISE, SETTLE, HOLD, FADE = 0.25, 0.9, 0.55, 0.5  # the part's light: up in RISE s, down to HOLD by SETTLE s; out in FADE s
WORDS, PLATE, LIGHT = (0.93, 0.95, 1.0), (0.04, 0.06, 0.09, 0.74), (0.35, 0.8, 1.0)
PLACE, SIZE = (0.025, 0.955), 19

_said = []                   # [(line, born)]: the lines on the plate, the newest last
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
        lines, moving = [], False
        for k, (line, born) in enumerate(_said):
            shown = int((now - born) * TYPE) if k == len(_said) - 1 else len(line)
            typing = shown < len(line)
            lines.append(line[:max(shown, 0)] + ("▍" if typing else ""))
            moving = moving or typing
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


@command(label="Сказать")
def say(scene, text: str = field("", label="Что сейчас строю", help="Одна короткая фраза для зрителя: что и зачем"),
        target: int = field(0, label="Деталь", help="Номер объекта, о котором строка: он светится, пока строка последняя (0 - никакой)")):
    """A line at the bottom of the view, typed letter by letter; the part it is about lights up."""
    text = " ".join(str(text).split())
    if not text:
        return {"said": False, "why": "text: пустая строка - нечего сказать"}
    now = time.monotonic()
    with _lock:
        for line in textwrap.wrap(text, WIDE):
            _said.append((line, now))
            now += len(line) / TYPE  # a wrapped line starts when the one before it is typed
        del _said[:-LINES]
        for part, (lit, out) in list(_lit.items()):
            if out is None:
                _lit[part] = (lit, time.monotonic())
        if target:
            _lit[int(target)] = (time.monotonic(), None)
    _wake()
    return {"said": True, "lines": len(_said)}


@command(label="Убрать рассказ")
def clear(scene):
    """The plate and every light off."""
    with _lock:
        _said.clear()
        _lit.clear()
    draw.clear()
    return {"cleared": True}
