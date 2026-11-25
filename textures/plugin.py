"""textures - a simple procedural generator of surface textures.

Six patterns - wood, brick, tiles, concrete, marble, brushed metal - made from a few numbers: how coarse, which variant,
two colours of your own if you want them. «Применить» makes a seamless picture (a PNG in ~/Kinetica/textures, named by
what it was made from: the same numbers give the same file) with its relief beside it, and puts it on every selected
shape as an ordinary image texture - with its repeats a metre and its relief, one undo step. The relief goes as a height
map (bump mapping: the editor works the slopes out, and its height and the repeats can be changed afterwards) or as a
normal map (the slopes themselves, baked for this height and these repeats - what ready texture sets bring). The scene
keeps only the files' paths: nothing of this plugin is needed to open it. «Образцы» puts a cube of each pattern into
the scene.

Only Python's standard library: the noise is a lattice of random numbers smoothly blended, summed over octaves, and
every pattern repeats at the picture's edges.
"""
import hashlib
import math
import os
import random
import struct
import zlib

from kinetica.plugin import command, field, show, tr, ui

SIZE = 256  # pixels a side: it repeats over the surface, so a small picture is enough

#        name       file        base colour           mark colour           what the mark is
KINDS = {"дерево": ("wood", (0.74, 0.54, 0.34), (0.46, 0.29, 0.15)),      # the rings' dark
         "кирпич": ("brick", (0.63, 0.29, 0.21), (0.80, 0.78, 0.73)),     # the mortar
         "плитка": ("tiles", (0.86, 0.87, 0.89), (0.36, 0.37, 0.40)),     # the grout
         "бетон": ("concrete", (0.66, 0.66, 0.64), (0.44, 0.44, 0.43)),   # the stains
         "мрамор": ("marble", (0.94, 0.94, 0.91), (0.36, 0.37, 0.42)),    # the veins
         "металл": ("metal", (0.80, 0.82, 0.85), (0.52, 0.54, 0.58))}     # the brush's scratches


def _noise(across, down, rng):
    """SIZE x SIZE of smooth noise from 0 to 1 that repeats at its edges: a lattice of across x down random numbers,
    blended smoothly between them."""
    lattice = [[rng.random() for _ in range(across)] for _ in range(down)]
    columns = []
    for x in range(SIZE):
        u = x * across / SIZE
        t = u - int(u)
        columns.append((int(u) % across, (int(u) + 1) % across, t * t * (3 - 2 * t)))
    out = []
    for y in range(SIZE):
        v = y * down / SIZE
        s = v - int(v)
        s = s * s * (3 - 2 * s)
        a, b = lattice[int(v) % down], lattice[(int(v) + 1) % down]
        row = [p + (q - p) * s for p, q in zip(a, b)]
        out.append([row[i] + (row[j] - row[i]) * t for i, j, t in columns])
    return out


def _clouds(across, down, octaves, rng):
    """Noise over octaves, each twice as fine and half as strong: from 0 to 1, repeating at the edges."""
    total, strength, whole = [[0.0] * SIZE for _ in range(SIZE)], 1.0, 0.0
    for octave in range(octaves):
        layer = _noise(min(SIZE, across << octave), min(SIZE, down << octave), rng)
        total = [[v + strength * w for v, w in zip(row, add)] for row, add in zip(total, layer)]
        whole += strength
        strength *= 0.5
    return [[v / whole for v in row] for row in total]


def _field(kind, grain, rng):
    """The pattern as two fields of SIZE x SIZE, each 0 to 1: how much of the mark's colour a pixel has, and how high it
    stands. `grain`: 1 as made, 2 twice as coarse, 0.5 twice as fine."""
    count = lambda n: max(1, round(n / grain))  # noqa: E731
    turn = 2 * math.pi / SIZE
    if kind == "wood":  # rings across the planks, pulled about by slow noise; fine streaks along them
        rings, bend, streaks = count(5), _clouds(count(2), count(3), 3, rng), _clouds(count(3), count(48), 2, rng)
        mark = [[0.5 + 0.5 * math.sin(turn * rings * y + 5 * bend[y][x]) for x in range(SIZE)] for y in range(SIZE)]
        mark = [[0.75 * m * m + 0.25 * s for m, s in zip(row, fine)] for row, fine in zip(mark, streaks)]
        return mark, [[1 - 0.6 * m for m in row] for row in mark]
    if kind == "marble":  # veins along a diagonal, torn by noise
        veins, tear = count(2), _clouds(count(3), count(3), 5, rng)
        mark = [[(1 - abs(math.sin(turn * veins * (x + y) / 2 + 7 * tear[y][x]))) ** 6 for x in range(SIZE)] for y in range(SIZE)]
        mark = [[min(1.0, m + 0.25 * t * t) for m, t in zip(row, torn)] for row, torn in zip(mark, tear)]
        return mark, [[1 - 0.15 * m for m in row] for row in mark]
    if kind == "concrete":  # stains, and pores on top of them
        stains, pores = _clouds(count(3), count(3), 5, rng), _noise(min(SIZE, count(96)), min(SIZE, count(96)), rng)
        mark = [[min(1.0, max(0.0, (s - 0.3) * 1.6) * 0.8 + (0.5 if p > 0.86 else 0.0)) for s, p in zip(row, pore)] for row, pore in zip(stains, pores)]
        return mark, [[1 - m for m in row] for row in mark]
    if kind == "metal":  # scratches along x: noise long across and fine down
        brush, sheen = _clouds(count(2), min(SIZE, count(96)), 2, rng), _clouds(count(1), count(2), 2, rng)
        mark = [[0.7 * b + 0.3 * s for b, s in zip(row, slow)] for row, slow in zip(brush, sheen)]
        return mark, [[1 - 0.5 * m for m in row] for row in mark]
    # bricks and tiles: cells with a seam around each, every cell of its own shade, a little dirt over all
    rows = 2 * count(4) if kind == "brick" else count(4)
    cells = rows // 2 if kind == "brick" else rows
    high, wide, seam = SIZE / rows, SIZE / cells, max(1.5, SIZE * (0.012 if kind == "brick" else 0.008))
    shades = [[rng.random() for _ in range(cells)] for _ in range(rows)]
    dirt = _clouds(count(4), count(4), 3, rng)
    mark, height = [], []
    for y in range(SIZE):
        r = int(y / high) % rows
        shift = wide / 2 if kind == "brick" and r % 2 else 0.0  # every other course of bricks half a brick along
        from_edge_y = min(y - int(y / high) * high, (int(y / high) + 1) * high - y)
        m_row, h_row = [], []
        for x in range(SIZE):
            u = (x + shift) / wide
            from_edge = min(from_edge_y, (u - int(u)) * wide, (int(u) + 1 - u) * wide)
            if from_edge < seam:
                m_row.append(1.0), h_row.append(0.25 * from_edge / seam)
            else:
                m_row.append(0.22 * shades[r][int(u) % cells] + 0.18 * dirt[y][x]), h_row.append(1.0 - 0.15 * dirt[y][x])
        mark.append(m_row), height.append(h_row)
    return mark, height


def _slopes(height, steep):
    """The field's slopes as a normal map's three channels, each SIZE x SIZE levels: red - the normal's lean to the
    right, green - up (OpenGL's way), blue - out of the surface. `steep`: the rise of the whole field's height over one
    pixel's width."""
    red, green, blue = bytearray(), bytearray(), bytearray()
    for y in range(SIZE):
        above, row, below = height[y - 1], height[y], height[(y + 1) % SIZE]
        for x in range(SIZE):
            nx, ny = -steep * (row[(x + 1) % SIZE] - row[x - 1]) / 2, -steep * (above[x] - below[x]) / 2
            long = math.sqrt(nx * nx + ny * ny + 1)
            red.append(int((nx / long + 1) * 127.5 + 0.5)), green.append(int((ny / long + 1) * 127.5 + 0.5)), blue.append(int((1 / long + 1) * 127.5 + 0.5))
    return bytes(red), bytes(green), bytes(blue)


def _png(path, grey, colours=None, channels=None):
    """A SIZE x SIZE picture written as a PNG: the field `grey` (rows of 0 to 1) as it is, or coloured from the first of
    `colours` at 0 to the second at 1; or three `channels` of levels as they are."""
    levels = b"" if channels else bytes(min(255, max(0, int(v * 255 + 0.5))) for row in grey for v in row)
    if channels:
        pixels = bytearray(3 * SIZE * SIZE)
        for channel in range(3):
            pixels[channel::3] = channels[channel]
        colours = True
    elif colours:
        a, b = colours
        pixels = bytearray(3 * len(levels))
        for channel in range(3):  # a table from the level to this channel: one pass in C, not one a pixel here
            table = bytes(min(255, max(0, int((a[channel] + (b[channel] - a[channel]) * k / 255) * 255 + 0.5))) for k in range(256))
            pixels[channel::3] = levels.translate(table)
    else:
        pixels = levels
    wide = len(pixels) // SIZE
    rows = b"".join(b"\x00" + bytes(pixels[y * wide:(y + 1) * wide]) for y in range(SIZE))
    chunk = lambda name, data: struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))  # noqa: E731
    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2 if colours else 0, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def make(kind, grain=1.0, variant=0, colours=None, folder=None, steep=0.0):
    """The pattern's picture, its relief as a height map and - with `steep` above 0, the rise of the whole height over
    one pixel's width - as a normal map: PNG files made once for the same words. Their paths; the normal map's is "" when
    none was asked for."""
    name, base, mark = KINDS[kind]
    colours = colours or (base, mark)
    folder = folder or os.path.join(os.path.expanduser("~"), "Kinetica", "textures")
    words = repr((name, round(float(grain), 3), int(variant), [[round(float(c), 3) for c in colour] for colour in colours], SIZE))
    stem = os.path.join(folder, f"{name}-{hashlib.sha1(words.encode()).hexdigest()[:10]}")
    normals = f"{stem}-normal-{steep:.3f}.png" if steep > 0 else ""
    if not all(os.path.exists(file) for file in (stem + ".png", stem + "-relief.png", normals) if file):
        os.makedirs(folder, exist_ok=True)
        amount, height = _field(name, max(0.25, min(4.0, float(grain))), random.Random(f"{name}/{int(variant)}"))
        _png(stem + "-relief.png", height)
        _png(stem + ".png", amount, colours)
        if normals:
            _png(normals, None, channels=_slopes(height, steep))
    return stem + ".png", stem + "-relief.png", normals


def _put(scene, shape, kind, grain, variant, colours, repeats, relief, maps="высоты"):
    """The picture on the shape with its relief: as a height map (bump), or baked into a normal map for this height
    and these repeats - a pixel is 1 / (repeats * SIZE) metres wide."""
    baked = maps == "нормали" and relief > 0
    picture, heights, normals = make(kind, grain, variant, colours, steep=float(relief) / 1000 * float(repeats) * SIZE if baked else 0.0)
    for key, values in (("kind", ["image"]), ("repeats", [float(repeats)]), ("file", [picture]), ("bump", [0.0 if baked else float(relief) / 1000]),
                        ("height_file", ["" if baked else heights]), ("normal_file", [normals])):
        scene.ed.scene_set(id=shape.id, line="texture", key=key, values=values)
    return picture


_KIND = field("дерево", choices=list(KINDS), label="Рисунок")
_GRAIN = field(1.0, range=(0.25, 4.0), label="Крупность", help="1 - как есть, 2 - вдвое крупнее, 0.5 - вдвое мельче")
_VARIANT = field(0, range=(0, 999), label="Вариант", help="Другое число - тот же рисунок с другими случайностями")
_REPEATS = field(1.0, unit="на м", range=(0.05, 100.0), label="Повторений")
_RELIEF = field(2.0, unit="мм", range=(0.0, 20.0), label="Рельеф", help="Насколько выпуклым рисунок выглядит на свету; 0 - гладко")
_MAPS = field("высоты", choices=["высоты", "нормали"], label="Рельеф картой",
              help="высоты - карта высот (bump): высоту и повторения потом можно менять; нормали - карта нормалей (normal map), как в готовых наборах")


@command(label="Текстура на выделенное")
def apply(scene, kind: str = _KIND, grain: float = _GRAIN, variant: int = _VARIANT, repeats: float = _REPEATS, relief: float = _RELIEF,
          maps: str = _MAPS, own: bool = field(False, label="Свои цвета"), base=field((0.7, 0.7, 0.7), range=(0.0, 1.0), label="Цвет основы"),
          mark=field((0.3, 0.3, 0.3), range=(0.0, 1.0), label="Цвет рисунка")):
    """Рисунок, сделанный из этих чисел, на каждую выделенную фигуру: бесшовная картинка и её рельеф, один шаг отмены."""
    shapes = [o for o in scene.editor.selection if o.kind == "entity"]
    if not shapes:
        raise ValueError(tr("выделите фигуру: текстура ложится на выделенное"))
    picture = ""
    for shape in shapes:
        picture = _put(scene, shape, kind, grain, variant, (tuple(base), tuple(mark)) if own else None, repeats, relief, maps)
    show(state=tr("{kind}: {file} на {n} фиг.").format(kind=kind, file=os.path.basename(picture), n=len(shapes)))
    return {"file": picture, "ids": [shape.id for shape in shapes]}


@command(label="Образцы")
def samples(scene):
    """По кубу каждого рисунка в ряд - посмотреть и выбрать."""
    made = []
    for k, kind in enumerate(KINDS):
        cube = scene.add("box", name=kind.capitalize(), position=(0.7 * (k - (len(KINDS) - 1) / 2), 0.27, 0.0), size=(0.5, 0.5, 0.5))
        _put(scene, cube, kind, 1.0, 0, None, 2.0, 2.0)
        made.append(cube)
    scene.editor.select(*made)
    show(state=tr("образцы: {n} кубов").format(n=len(made)))
    return {"ids": [cube.id for cube in made]}


ui.panel("Текстуры", ui.section("Рисунок", ui.choice("kind", list(KINDS), "дерево", label="Рисунок"),
                                ui.slider("variant", 0, range=(0, 999), label="Вариант"),
                                ui.number("grain", 1.0, label="Крупность", range=(0.25, 4.0), step=0.25),
                                ui.number("repeats", 1.0, label="Повторений", unit="на м", range=(0.05, 100.0)),
                                ui.number("relief", 2.0, label="Рельеф", unit="мм", range=(0.0, 20.0)),
                                ui.choice("maps", ["высоты", "нормали"], "высоты", label="Рельеф картой")),
         ui.section("Цвета", ui.toggle("own", False, label="Свои цвета"), ui.vector("base", (0.7, 0.7, 0.7), label="Основа"),
                    ui.vector("mark", (0.3, 0.3, 0.3), label="Рисунок")),
         ui.row(ui.button(apply, "Применить к выделенному"), ui.button(samples, "Образцы")),
         ui.value("state", "Сделано", "—"))
