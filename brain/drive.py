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


# --- A car of joints (robot.JointCar): its own wheels, knuckles and springs, the physics' tyres ---------------------------
# The planner's two numbers held by a stabiliser under it: the speed by the driven wheels' motors (v / r, each wheel at its
# own track's speed round the turn - an electronic differential), the turn rate
# by the knuckles' servos - the kinematic angle of a bicycle, atan(L w / v), corrected by a PI loop on the IMU's
# gyroscope (the tyres slip, the knuckles lag: what the angle alone gives is short of the turn asked), and split by
# Ackermann - each front wheel square to its line to the turn's centre, the inner one turned more.
GAIN, RESET = 0.8, 1.5     # the gyroscope loop: the proportional part, and the integral's rate, 1/s
WINDUP = 1.0               # the integral's bound, rad/s


def ackermann(turn, speed, base, track, most):
    """The left and the right front wheels' angles, rad (positive: to the left), for a turn rate `turn` (rad/s, positive
    to the left) at `speed` (m/s, negative: backwards) on a wheelbase `base` and a track `track`, m; each at most `most`
    rad. Backwards the same turn takes the wheels the other way (yaw rate = speed · tan(angle) / base) - with the speed's
    size alone the gyroscope's correction would push the wrong way and spin the car round."""
    v = speed if abs(speed) >= 0.3 else math.copysign(0.3, speed)  # standing: as if rolling slowly forward
    centre = math.atan(base * turn / v)
    if abs(centre) < 1e-6:
        return 0.0, 0.0
    radius = base / math.tan(centre)  # signed: positive, the centre on the left
    s = 1.0 if radius > 0 else -1.0  # atan2 keeps the turn's side even where the centre is inside the track
    left, right = s * math.atan2(base, s * (radius - track / 2)), s * math.atan2(base, s * (radius + track / 2))
    clamp = lambda a: max(-most, min(most, a))
    return clamp(left), clamp(right)


class Stabiliser:
    """The turn rate held by the gyroscope: the asked turn plus a PI correction of what it reads short. A gyroscope reads a
    turn where there is none (an MPU-6050's zero is off by up to 20 °/s by its datasheet), so first it stands still - half
    a second to stop and straighten, half a second to hear its zero - and takes that zero off every reading after, as a
    robot does at its start (`calibrating` while it does)."""

    SETTLE, LISTEN = 0.5, 0.5  # s

    def __init__(self):
        self.integral, self.clock, self.zero, self.heard = 0.0, 0.0, 0.0, []

    @property
    def calibrating(self):
        return self.clock < self.SETTLE + self.LISTEN

    def hear(self, gyro, dt):
        """One frame of the start: the gyroscope's reading while the car stands; its mean is the zero."""
        self.clock += dt
        if self.clock > self.SETTLE:
            self.heard.append(gyro)
            self.zero = sum(self.heard) / len(self.heard)

    def turn(self, asked, gyro, dt, moving=True):
        """The turn rate to steer for: the asked one plus the correction of what the gyroscope reads short. A car told
        to stand gets the asked one alone and forgets its integral: standing, it turns no way the correction could
        mend, and a wound-up integral would hold its wheels turned to the stop."""
        if not moving:
            self.integral = 0.0
            return asked
        error = asked - (gyro - self.zero)
        self.integral = max(-WINDUP, min(WINDUP, self.integral + RESET * error * dt))
        return asked + GAIN * error + self.integral
