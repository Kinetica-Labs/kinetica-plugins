"""A car's drive (plugin.py in play, evolution.py in training - the same numbers in both): what its wheels would do
for a block on the floor. The brain plans two numbers, a turn and a speed; the drive holds them:

- the tyres - mechanics, which read the physics as a tyre does: they cancel the car's sideways speed, as far as their
  grip goes (past it the car slides: a skid);
- the motor holds the speed the brain asks for (its second output, between the card's least and most speed), as far
  as it can push - the brain plans, slow into a tight turn, fast on a straight;
- the steering holds the turn rate the brain asks for (its first output times the card's «Поворот»), by the car's own
  gyroscope - its IMU's reading: the loop that keeps the heading reads a sensor, as a robot's does;
- a reflex under the brain turns away from the nearer walls (the card's «Рефлекс отворота» times how much nearer the
  right ones are than the left, by the lidar): a car that keeps off the walls before it learns anything - the brain's
  turn is a correction over it (a layered control: the reflex low, the plan high).

Only the standard library."""
import math

GRIP = 9.0       # m/s^2 the tyres hold sideways (a grip of about 0.9)
PUSH = 6.0       # m/s^2 the motor gives at most, forward or back
HOLD = 8.0       # 1/s: how fast the motor makes up a speed short
STICK = 20.0     # 1/s: how fast the tyres take a sideways speed out
STEER = 25.0     # 1/s: how fast the steering makes up a turn rate short


def axes(rotation):
    """The body's x (forward) and z (sideways) in the world, from its Euler degrees [x, y, z] (R = Ry Rx Rz)."""
    x, y, z = (math.radians(a) for a in rotation)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    # R = Ry Rx Rz: its first and third columns
    forward = (cy * cz + sy * sx * sz, cx * sz, -sy * cz + cy * sx * sz)
    side = (sy * cx, -sx, cy * cx)
    return forward, side


def tyres(state, speed):
    """The force of the tyres and the motor on the car, world, N: its state as the physics hook gives it (rotation,
    velocity, mass)."""
    forward, side = axes(state["rotation"])
    v, m = state["velocity"], float(state["mass"])
    along = sum(a * b for a, b in zip(v, forward))
    across = sum(a * b for a, b in zip(v, side))
    push = max(-PUSH, min(PUSH, HOLD * (float(speed) - along)))
    grip = max(-GRIP, min(GRIP, -STICK * across))
    return [m * (push * forward[k] + grip * side[k]) if k != 1 else 0.0 for k in range(3)]


def steering(gyro, turn, inertia):
    """The torque about the world's up, N m, that brings the turn rate the gyroscope reads (rad/s about the car's up)
    to `turn`."""
    return [0.0, float(inertia) * STEER * (float(turn) - float(gyro)), 0.0]


def plan(outputs, card, away=0.0):
    """The brain's two outputs as the drive's aims: (turn rate, rad/s; speed, m/s); `away` - Wiring.away, the reflex's."""
    least, most = float(card.get("speed_min", 1.0)), float(card.get("speed", 4.0))
    turn = max(-1.0, min(1.0, outputs[0] + float(card.get("reflex", 0.0)) * away))
    return turn * float(card.get("turn", 3.0)), least + (outputs[1] + 1.0) / 2.0 * (most - least)


def inertia(mass, size):
    """A block's moment about its up: m (x^2 + z^2) / 12."""
    return float(mass) * (size[0] ** 2 + size[2] ** 2) / 12.0
