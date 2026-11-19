"""cubes - cubes for the robot to grip, a plugin of the Kinetica editor (an example: Плагины → «Ряд кубиков…»).

One file is the whole plugin: each @command is a button in «Плагины», a line in Ctrl+K, ed.cubes_row(count=3) in the
console, an operation for scripts and a tool for AI agents over MCP. Its fields are the form the button opens. All it
makes is one undo step.
"""
from kinetica.plugin import command, field, run


@command(label="Ряд кубиков", key="Ctrl+Shift+R")
def row(scene, count: int = field(5, range=(1, 30), label="Сколько"),
        gap: float = field(0.12, unit="м", range=(0.06, 0.5), label="Шаг"),
        size: float = field(0.05, unit="м", range=(0.02, 0.15), label="Ребро"),
        x: float = field(0.45, unit="м", range=(-1.5, 1.5), label="Отступ по x")):
    """A row of rigid cubes on the floor along z, in front of the robot: something to pick up."""
    made = []
    for k in range(count):
        z = (k - (count - 1) / 2) * gap
        answer = scene.client.call("scene_add", shape="box", size=[size, size, size], role="rigid", position=[x, size / 2, z])
        made.append(answer.get("id"))
    return {"made": len(made), "ids": made}


run()
