"""The orientation board as a robot arm's puppet strings (plugin.py's «Рука за датчиком»): the board's turn from the pose
it had when taken, as three angles, each led to joints of the arm - heading to the base, tilt to the shoulder and, bent
back, the elbow, the twist to the wrist's last joint - smoothed so the servos get no jolts. No editor here: the
arithmetic only, tested by test_depz.py."""
import math

import bno086

# (the board's angle, the arm's joint - its place in the chain from the base, how much of the angle it turns)
STRINGS = (("yaw", 0, 1.0), ("pitch", 1, 1.0), ("pitch", 2, -0.6), ("roll", 5, 1.0))
TAU = 0.08      # s: the smoothing's time constant - jitter of the hand gone, the motion still the hand's
LIMIT = 2.6     # rad: no joint is led further than this from where it stood


def relative(q, zero):
    """The board's turn from its zero pose: zero⁻¹ · q, quaternions (w x y z), unit."""
    w0, x0, y0, z0 = zero[0], -zero[1], -zero[2], -zero[3]
    w, x, y, z = q
    return (w0 * w - x0 * x - y0 * y - z0 * z, w0 * x + x0 * w + y0 * z - z0 * y,
            w0 * y - x0 * z + y0 * w + z0 * x, w0 * z + x0 * y - y0 * x + z0 * w)


def angles(q):
    """A turn as roll, pitch, yaw in radians (about the board's x, y, z; z-y-x order: bno086.euler)."""
    return dict(zip(("roll", "pitch", "yaw"), bno086.euler(q)))


def targets(turn, rest, flip=()):
    """Each joint's target for the board's turn (angles) from the joints' rest coordinates: the strings' sum, each angle
    reversed when it is in `flip`, a joint the arm has not (a chain of three: no wrist) skipped."""
    out = list(rest)
    for angle, joint, share in STRINGS:
        if joint < len(rest):
            led = share * turn[angle] * (-1 if angle in flip else 1)
            out[joint] = rest[joint] + max(-LIMIT, min(LIMIT, out[joint] - rest[joint] + led))
    return out


def smooth(shown, wanted, dt):
    """Where the joints are told to be after `dt` s on their way from `shown` to `wanted` (first-order lag, TAU)."""
    share = 1 - math.exp(-max(dt, 0.0) / TAU)
    return [s + (w - s) * share for s, w in zip(shown, wanted)]
