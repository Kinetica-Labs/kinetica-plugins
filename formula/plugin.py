"""formula - LaTeX formulas in the scene.

A formula belongs to a shape, as its component. «Формула над объектом» is drawn on the screen over its shape, facing
the viewer, and goes where the shape goes - in the edit and while the scene plays; «Формула в мире» lies in the world
on the shape; «Титр» stands in the frame; «Выноска» stands aside with a leader line to a point of the shape, as a drawing
labels a part (ГОСТ 2.316, ISO 128-22) - what an agent explaining what it builds needs. Select a shape and press a button of the «Формулы» tab (or «+ Добавить
компонент» in the inspector), then type LaTeX in the component's card: the formula is typeset anew on Enter. With
nothing selected a button makes the formula a small hidden shape of its own, «Формула N»: click the formula to select it.

The source is typeset by the system's own `latex` and `dvisvgm` (TeX Live, MiKTeX) into filled outlines when the computer
has them, else by the built-in ziamath (third_party/: pure-Python submodules, MIT, with the STIX Two Math font, SIL OFL) -
nothing to install; the editor paints them (draw.fill). No other package is used than those and the standard library.
"""
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

from kinetica.plugin import command, component, draw, field, on_change, on_stop, show, tr, ui

_LATEX = field("F = ma", label="LaTeX", help="Формула, как между $ и $: T = 2\\pi\\sqrt{\\frac{L}{g}}")
_COLOUR = dict(range=(0.0, 1.0), label="Цвет: красный, зелёный, синий")
_EDGE = field(True, label="Обводка", help="Тёмный контур вокруг знаков: формула читается на любом фоне")
EDGE = {"stroke": (0.02, 0.03, 0.05), "stroke_width": 0.06}  # a dark edge, six hundredths of an em


@component(label="Формула над объектом")
class Label:
    """LaTeX над фигурой: рисуется на экране лицом к зрителю и едет вместе с фигурой."""
    latex = _LATEX
    size = field(5.0, unit="% вида", range=(0.5, 50.0), label="Размер", help="Высота буквы в долях высоты вида")
    offset = field((0.0, 0.35, 0.0), unit="м", label="Сдвиг от фигуры", help="По осям мира: формула остаётся над телом, как бы оно ни кувыркалось")
    colour = field((1.0, 0.86, 0.3), **_COLOUR)
    outline = _EDGE


@component(label="Формула в мире")
class World:
    """LaTeX в пространстве, на своей плоскости: лежит на фигуре, видна с перспективой."""
    latex = _LATEX
    size = field(0.2, unit="м", range=(0.01, 10.0), label="Размер", help="Высота буквы в метрах")
    offset = field((0.0, 0.5, 0.0), unit="м", label="Сдвиг на фигуре", help="В осях фигуры")
    turn = field((0.0, 0.0, 0.0), unit="°", label="Поворот на фигуре")
    colour = field((0.72, 0.62, 0.97), **_COLOUR)
    outline = _EDGE


@component(label="Титр")
class Screen:
    """LaTeX в кадре, как титр: камера его не двигает."""
    latex = _LATEX
    size = field(6.0, unit="% вида", range=(0.5, 50.0), label="Размер", help="Высота буквы в долях высоты вида")
    left = field(50.0, unit="%", range=(0.0, 100.0), label="Слева")
    top = field(12.0, unit="%", range=(0.0, 100.0), label="Сверху")
    colour = field((0.96, 0.96, 0.98), **_COLOUR)
    outline = _EDGE


# The leader's end, as a drawing marks it (ГОСТ 2.316, ISO 128-22): a dot on the part, an arrow on its outline, or none.
_ENDS = {"точка": "dot", "стрелка": "arrow", "нет": "none"}


@component(label="Выноска")
class Callout:
    """LaTeX в стороне от фигуры и линия-выноска к её точке, как на чертеже (ГОСТ 2.316, ISO 128-22): надпись на полке,
    от ближнего к детали конца полки - линия к точке; точку тянут оранжевой ручкой прямо по детали."""
    latex = _LATEX
    size = field(6.0, unit="% вида", range=(0.5, 50.0), label="Размер", help="Высота буквы в долях высоты вида: крупно - видео смотрят с телефона")
    arrange = field("сама", choices=("сама", "вручную"), label="Раскладка",
                    help="Сама - в колонки слева и справа от деталей, по порядку их точек: выноски не пересекаются при любом виде; вручную - по «Где надпись»")
    offset = field((0.6, 0.5, 0.0), unit="м", label="Где надпись", help="Раскладка вручную: от фигуры по осям мира")
    point = field((0.0, 0.0, 0.0), unit="м", handle="point", label="Точка на детали",
                  help="В осях фигуры: сюда показывает выноска; тяните оранжевую точку по детали")
    end = field("точка", choices=tuple(_ENDS), label="Конец выноски", help="Точка - на детали, стрелка - на её контуре, нет - без знака")
    colour = field((1.0, 0.86, 0.3), **_COLOUR)
    outline = _EDGE


# How a callout comes up, as a hand draws it, s - an explaining video's pace, slower than a button's 0.2-0.4 s (Material
# Design): a ring pulses once from the point on the part, the line grows to the words, the reference line slides out, the
# words come up - then it holds still: the eye goes to what starts moving, not to what keeps moving (motion onset captures
# attention, Abrams and Christ 2003). The three newest stay bright, an older one fades as a newer one comes: working memory
# holds about four things (Cowan 2001); words beside their part, when it is put (Mayer's spatial and temporal contiguity).
PULSE, LINE, SHELF, WORDS_AT, WORDS, FADE, DIM, BRIGHT = 0.6, 0.4, 0.15, 0.45, 0.3, 0.6, 0.45, 3
_born = {}   # a callout's drawing key -> when it first stood (time.monotonic())
_calls = {}  # its key -> what draw.fill draws it with, to draw it again frame by frame
_animating = {"thread": None}


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3  # eased out: a hand slows where it stops


def look(key, now):
    """A callout's appearance now - its leader's grow, pulse and words, its opacity - and whether it still changes."""
    t = now - _born[key]
    grow = 0.75 * _ease(t / LINE) + 0.25 * _ease((t - LINE) / SHELF)
    leader = {"grow": grow, "words": _ease((t - WORDS_AT) / WORDS), "pulse": t / PULSE if t < PULSE else None}
    newer = sorted(b for k, b in _born.items() if k != key and (b, k) > (_born[key], key))
    opacity, changing = 1.0, t < WORDS_AT + WORDS
    if len(newer) >= BRIGHT:  # the third newer one came: this one fades to grey
        since = now - newer[BRIGHT - 1]
        opacity, changing = 1 - (1 - DIM) * _ease(since / FADE), changing or since < FADE
    return leader, opacity, changing


def _show(key, now):
    outlines, kwargs = _calls[key]
    leader, opacity, changing = look(key, now)
    draw.fill(key, outlines, leader={**kwargs["leader"], **leader}, opacity=opacity, **{k: v for k, v in kwargs.items() if k != "leader"})
    return changing


def _animate():
    """The callouts that still come up or fade, drawn again some 30 times a second until all hold still."""
    try:
        while True:
            now = time.monotonic()
            moving = [key for key in list(_calls) if key in _born and _show(key, now)]
            if not moving:
                break
            time.sleep(1 / 30)
    finally:
        _animating["thread"] = None


def _wake():
    if _animating["thread"] is None:
        _animating["thread"] = threading.Thread(target=_animate, daemon=True)
        _animating["thread"].start()


class Problem(Exception):
    """Why a source was not typeset, in words for the person who typed it."""


class Missing(Problem):
    """latex or dvisvgm is not on this computer: the built-in typesetter sets the formula instead."""


_typeset = {}  # source -> its outlines: typesetting takes a third of a second


def typeset(latex):
    """LaTeX mathematics, as between $ and $, into outlines for draw.fill: rows "M x y L x y C x1 y1 x2 y2 x y Z" in ems,
    x right and y up, the formula's centre at 0 0 - by the computer's LaTeX when it has one, else by the built-in
    typesetter (ziamath): nothing to install. Raises Problem: the source's own fault, in words."""
    if latex not in _typeset:
        try:
            svg = _svg(latex)
        except Missing:
            svg = _builtin(latex)
        _typeset[latex] = _outlines(svg)
    return _typeset[latex]


def _builtin(latex):
    """The formula's SVG by ziamath - three pure-Python submodules of third_party/ (ziamath, ziafont, latex2mathml; MIT) and
    the STIX Two Math font (SIL OFL), each pinned to its release - drawn with paths and rectangles alone, as dvisvgm draws
    them: one converter (_outlines) reads both. 10 units an em, as LaTeX's 10 pt type."""
    if "ziamath" not in sys.modules:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
        sys.path[:0] = [os.path.join(folder, repo) for repo in ("ziamath", "ziafont", "latex2mathml")]
        sys.path.append(folder)  # latex2mathml asks for its version: the dist-info beside the submodule says it
    import ziamath
    ziamath.config.svg2 = False  # each glyph its own path, no <use> of <symbol>s
    try:
        return ziamath.Latex(latex, size=10).svg()
    except Exception as error:  # noqa: BLE001 - the parser's words for a source it cannot read
        raise Problem(tr("встроенный набор формул не понял «{latex}»: {why}").format(latex=latex, why=error)) from None


def _svg(latex):
    quiet = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}  # no console flashing up on Windows
    missing = tr("не установлен: формулы набирают программы latex и dvisvgm (TeX Live или MiKTeX)")
    with tempfile.TemporaryDirectory(prefix="kinetica-formula-") as folder:
        cyrillic = "" if latex.isascii() else "\\usepackage[T2A]{fontenc}\n\\usepackage[utf8]{inputenc}\n"
        with open(os.path.join(folder, "f.tex"), "w", encoding="utf-8") as tex:
            tex.write("\\documentclass[preview]{standalone}\n" + cyrillic + "\\usepackage{amsmath,amssymb}\n\\begin{document}\n"
                      "$\\displaystyle " + latex + "$\n\\end{document}\n")
        try:
            subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "f.tex"], cwd=folder,
                           capture_output=True, timeout=60, **quiet)
        except FileNotFoundError:
            raise Missing("LaTeX " + missing) from None
        if not os.path.exists(os.path.join(folder, "f.dvi")):
            with open(os.path.join(folder, "f.log"), encoding="utf-8", errors="replace") as log:
                said = next((line[2:].strip() for line in log if line.startswith("!")), tr("страница не получилась"))
            raise Problem("LaTeX: " + said)
        try:
            subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "--precision=4", "-o", "f.svg", "f.dvi"], cwd=folder,
                           capture_output=True, timeout=60, **quiet)
            with open(os.path.join(folder, "f.svg"), encoding="utf-8") as svg:
                return svg.read()
        except FileNotFoundError:
            raise Missing("dvisvgm " + missing) from None


def _path(data, dx, dy):
    """An SVG path's commands - every one but the arc, absolute or relative - as M, L, C and Z moved by (dx, dy)."""
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", data)
    out, at = [], 0
    x = y = sx = sy = cx = cy = 0.0
    command = last = ""
    count = {"Z": 0, "H": 1, "V": 1, "M": 2, "L": 2, "T": 2, "S": 4, "Q": 4, "C": 6}
    while at < len(tokens):
        if tokens[at].isalpha():
            command, at = tokens[at], at + 1
        elif command in ("", "Z", "z"):
            raise Problem(tr("dvisvgm записал путь, который не читается"))
        kind, relative = command.upper(), command.islower()
        if kind not in count:
            raise Problem(tr("dvisvgm записал команду пути «{command}», которой этот плагин не знает").format(command=command))
        n = [float(t) for t in tokens[at:at + count[kind]]]
        at += count[kind]
        if relative:
            n = [v + (x if kind == "H" else y if kind == "V" else (y if k % 2 else x)) for k, v in enumerate(n)]
        if kind == "Z":
            out.append(("Z",))
            x, y = sx, sy
        elif kind == "M":
            out.append(("M", n[0], n[1]))
            x, y, sx, sy = n[0], n[1], n[0], n[1]
            command = "l" if relative else "L"  # the pairs after it are lines
        elif kind in "LHV":
            x, y = (n[0], n[1]) if kind == "L" else (n[0], y) if kind == "H" else (x, n[0])
            out.append(("L", x, y))
        elif kind in "CS":
            if kind == "S":  # its first control point mirrors the last curve's second
                n = [2 * x - cx, 2 * y - cy] + n if last == "C" else [x, y] + n
            out.append(("C", *n))
            cx, cy, x, y = n[2], n[3], n[4], n[5]
        else:  # Q, T: a quadratic curve as the cubic that is the same curve
            qx, qy = (n[0], n[1]) if kind == "Q" else (2 * x - cx, 2 * y - cy) if last == "Q" else (x, y)
            ex, ey = n[-2], n[-1]
            out.append(("C", x + (qx - x) * 2 / 3, y + (qy - y) * 2 / 3, ex + (qx - ex) * 2 / 3, ey + (qy - ey) * 2 / 3, ex, ey))
            cx, cy, x, y = qx, qy, ex, ey
        last = "C" if kind in "CS" else "Q" if kind in "QT" else kind
    return [(op[0], *[v + (dy if k % 2 else dx) for k, v in enumerate(op[1:])]) for op in out]


def _outlines(svg):
    """What dvisvgm writes - the glyphs defined once, their uses, the rules - as rows for draw.fill: the page's points
    (y down) into ems of its 10 pt type (y up) about the formula's centre."""
    root = ET.fromstring(svg)
    tag = lambda element: element.tag.rsplit("}", 1)[-1]  # noqa: E731
    number = lambda element, name: float(element.get(name, "0"))  # noqa: E731
    shapes, glyphs = {}, []
    for defs in root.iter():
        if tag(defs) == "defs":
            shapes.update({p.get("id"): p.get("d", "") for p in defs.iter() if tag(p) == "path"})
    defined = {id(p) for defs in root.iter() if tag(defs) == "defs" for p in defs.iter()}
    for element in root.iter():
        if element.get("transform"):
            raise Problem(tr("dvisvgm записал преобразование, которого этот плагин не знает"))
        if id(element) in defined:
            continue
        if tag(element) == "use":
            link = next((v for k, v in element.attrib.items() if k.endswith("href")), "")
            glyphs.append(_path(shapes.get(link.lstrip("#"), ""), number(element, "x"), number(element, "y")))
        elif tag(element) == "path":
            glyphs.append(_path(element.get("d", ""), 0.0, 0.0))
        elif tag(element) == "rect":  # a rule: a fraction's bar, a root's roof
            x, y, w, h = (number(element, name) for name in ("x", "y", "width", "height"))
            glyphs.append([("M", x, y), ("L", x + w, y), ("L", x + w, y + h), ("L", x, y + h), ("Z",)])
    glyphs = [g for g in glyphs if g]
    xs = [v for g in glyphs for op in g for v in op[1::2]]
    ys = [v for g in glyphs for op in g for v in op[2::2]]
    if not xs:
        raise Problem(tr("формула пуста: LaTeX ничего не нарисовал"))
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    word = lambda v: format(v, ".4f").rstrip("0").rstrip(".") or "0"  # noqa: E731
    return [" ".join(" ".join([op[0]] + [word((v - (my if k % 2 else mx)) / 10 * (-1 if k % 2 else 1)) for k, v in enumerate(op[1:])])
                     for op in g) for g in glyphs]


_drawn = set()  # the keys of the pictures on the view


@on_stop
def _after_run(scene):
    """Stop takes off what was drawn while the scene ran - a formula typeset late, after ▶, among it: all drawn again."""
    redraw(scene)


@on_change
def redraw(scene):
    """Every formula of the scene on the view: typeset (once a source) and painted where its component says."""
    keys, problems = set(), []
    for kind in (Label, World, Screen, Callout):
        for id, v in kind.instances(scene):
            try:
                outlines = typeset(str(v["latex"]))
            except Problem as problem:
                problems.append(tr("«{latex}» — {problem}").format(latex=v["latex"], problem=problem))
                continue
            key = f"{kind.__name__.lower()}/{id}"
            edge = EDGE if v["outline"] else {}
            if kind is Label:
                draw.fill(key, outlines, object=id, at=v["offset"], space="label", scale=v["size"] / 100, colour=v["colour"], **edge)
            elif kind is Callout:  # drawn as it comes up (look): the first time, and again only while it changes
                _calls[key] = (outlines, dict(object=id, at=v["offset"], space="label", scale=v["size"] / 100, colour=v["colour"],
                                              leader={"point": v["point"], "end": _ENDS.get(str(v["end"]), "dot"),
                                                      "place": "here" if v.get("arrange") == "вручную" else "aside"}, **edge))
                _born.setdefault(key, time.monotonic())
                if _show(key, time.monotonic()):
                    _wake()
            elif kind is World:
                draw.fill(key, outlines, object=id, at=v["offset"], turn=v["turn"], scale=v["size"], colour=v["colour"], **edge)
            else:
                draw.fill(key, outlines, object=id, space="screen", place=(v["left"] / 100, v["top"] / 100), scale=v["size"] / 100, colour=v["colour"], **edge)
            keys.add(key)
    for gone in _drawn - keys:
        draw.clear(gone)
        _calls.pop(gone, None), _born.pop(gone, None)
    _drawn.clear()
    _drawn.update(keys)
    for words in problems:
        print("formula:", words)
    show(state="⚠ " + problems[0] if problems else tr("формул в сцене: {n}").format(n=len(keys)))


def _put(scene, kind, latex, target=0):
    """The component on the shape `target` (an id; 0: every selected one); with none, on a small hidden shape of its own."""
    if target and int(target) not in [o.id for o in scene.objects]:
        raise ValueError(tr("нет объекта {id} в сцене").format(id=int(target)))
    shapes = [scene.objects[int(target)]] if target else [o for o in scene.editor.selection if o.kind == "entity"]
    point = None
    if target and shapes[0].kind == "joint":  # a joint: on its moving body, the leader to the joint's axis there (τ of a servo)
        link = next(line for line in scene.ed.scene_get(id=int(target))["lines"] if line[0] == "link")
        shapes, point = [scene.objects[int(link[link.index("a") + 1])]], [float(v) for v in link[link.index("anchor_a") + 1:link.index("anchor_a") + 4]]
    if any(o.kind != "entity" for o in shapes):  # a battery: no place in space to point at or hang from
        raise ValueError(tr("формула - у фигуры или сустава: «{name}» - ни то ни другое").format(name=next(o.name for o in shapes if o.kind != "entity")))
    if not shapes:
        holder = scene.add("box", name=f"{kind._kinetica['label']} {1 + sum(1 for _ in kind.instances(scene))}", position=(0.0, 1.0, 0.0),
                           size=(0.05, 0.05, 0.05))
        holder.visible = False
        shapes = [holder]
    type = f"formula.{kind._kinetica['name']}"
    for shape in shapes:
        values = {"latex": latex}
        if kind is Callout and shape.visible:  # aside and above the shape, the leader to the middle of its top - each next
            # callout on the other side, then a row higher, so that two never lie on each other (a drawing's notes alternate)
            n = sum(1 for _ in Callout.instances(scene))
            side, row = (1 if n % 2 == 0 else -1), (n // 2) % 2
            values["offset"] = [side * (shape.size[0] / 2 + 0.45), shape.size[1] / 2 + 0.3 + 0.35 * row, 0.0]
            values["point"] = point or [0.0, shape.size[1] / 2, 0.0]
        elif kind is not Screen and shape.visible:  # over the shape, or on its top
            values["offset"] = [0.0, shape.size[1] / 2 + (0.25 if kind is Label else 0.02), 0.0]
        scene.ed.component_set(id=shape.id, type=type, values=values)
    if not target:  # a button's: the shapes it labelled, selected; a call naming its shape (an agent's) leaves the selection be
        scene.editor.select(*shapes)
    typeset(latex)  # a source LaTeX refuses is said now, with the command's answer
    return {"ids": [shape.id for shape in shapes]}


@command(label="Контуры формулы")
def outlines(scene, latex: str = _LATEX):
    """Контуры формулы для draw.fill - другому плагину или агенту, который рисует её сам: строки «M x y L x y C … Z» в эм."""
    return {"outlines": typeset(latex)}


@command(label="Над объектом")
def label(scene, latex: str = _LATEX):
    """Формула над каждой выделенной фигурой: рисуется на экране лицом к зрителю и едет вместе с фигурой."""
    return _put(scene, Label, latex)


@command(label="В мире")
def world(scene, latex: str = _LATEX):
    """Формула в пространстве, на выделенной фигуре: лежит на своей плоскости, видна с перспективой."""
    return _put(scene, World, latex)


@command(label="Титр")
def screen(scene, latex: str = _LATEX):
    """Формула в кадре, как титр: камера её не двигает."""
    return _put(scene, Screen, latex)


@command(label="Выноска")
def callout(scene, latex: str = _LATEX, target: int = field(0, range=(0, 10**9), label="Фигура", help="Её id или сустава; 0 - выделенные")):
    """Формула в стороне от фигуры и линия-выноска к ней, как на чертеже: агент подписывает то, что строит; у сустава -
    на его звене, линия к оси. Точку на детали потом тянут оранжевой ручкой, конец выноски - точка, стрелка или нет."""
    return _put(scene, Callout, latex, target)


ui.panel("Формулы", ui.section("Новая формула", ui.text("latex", "F = ma", label="LaTeX"),
                               ui.row(ui.button(label, "Над объектом"), ui.button(world, "В мире"), ui.button(screen, "Титр"),
                                      ui.button(callout, "Выноска"))),
         ui.label("Выделите фигуру и нажмите кнопку; потом правьте LaTeX в её карточке — формула набирается заново."),
         ui.value("state", "Состояние", "—"))
