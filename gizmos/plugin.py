"""gizmos - a plugin's gizmos, as Cinema 4D's handles and tools (an example).

«Заслонка» is a component: «Куб с заслонкой» puts a cube with it into the scene (or add it to any shape with
«+ Добавить компонент») - drag its orange points: the angle on its circle, the size on the faces, the reach along x.
«Линейка» is a tool of the shelf at the left of the view: press on a point, drag, let go - a line from one to the
other with its length.
"""
from kinetica.plugin import command, component, draw, field


@component(label="Заслонка")
class Flap:
    angle = field(30.0, unit="°", range=(0.0, 360.0), label="Угол", handle="dial")
    size = field((0.4, 0.2, 0.6), label="Габарит", handle="box")
    reach = field(0.5, unit="м", range=(0.1, 2.0), label="Вылет", handle="radius")


@command(label="Куб с заслонкой")
def demo(scene):
    """A cube with the «Заслонка» component, selected: its orange points are there to drag."""
    cube = scene.add("box", position=(0.8, 0.3, 0.8), size=(0.4, 0.2, 0.6))
    scene.ed.component_set(id=cube.id, type="gizmos.flap")
    scene.editor.select(cube)
    return {"id": cube.id}


@command(label="Линейка", tool=True)
def ruler(scene, point, to):
    """A line from the point pressed to the point the drag is let go at, and how long it is."""
    length = sum((b - a) ** 2 for a, b in zip(point, to)) ** 0.5
    draw.line("линейка", [point, to])
    draw.label("длина", [(a + b) / 2 for a, b in zip(point, to)], f"{length:.3f} м")
    print("линейка:", round(length, 3), "м")
    return {"length": length}


@command(label="Стереть линейку")
def wipe(scene):
    """The ruler's line and number off the view."""
    draw.clear()
    return {}
