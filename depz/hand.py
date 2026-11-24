"""The orientation board turns the arm's wrist (plugin.py's «Запястье за датчиком»): the board's turn since the pose it
had when taken, carried into the world by how it is held - its x into the screen, its z up - and the wrist's three joints
solved so that the hand turns in the world exactly so. Their rotations about their axes (the arm at rest, the product of
exponentials) composed must give that turn: a few Newton steps from the last frame's angles - no jumps, any three axes.
No editor here: the arithmetic only, tested by test_depz.py."""
import math

TAU = 0.06      # s: the smoothing's time constant - the board's jitter gone, the motion still the hand's


def mul(a, b):
    """The quaternion product a · b (w x y z): b's turn, then a's."""
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def about(axis, angle):
    """The turn by `angle` radians about a unit `axis`."""
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def rotate(q, v):
    """The vector v turned by q."""
    return mul(mul(q, (0.0, *v)), conj(q))[1:]


def log(q):
    """A turn as its rotation vector (axis times angle, radians), the shorter way round."""
    w, x, y, z = q if q[0] >= 0 else (-q[0], -q[1], -q[2], -q[3])
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (2 * x, 2 * y, 2 * z)
    k = 2 * math.atan2(s, w) / s
    return (x * k, y * k, z * k)


def euler(degrees):
    """A scene file's rotation (Euler degrees x, y, z) as a quaternion: Ry · Rx · Rz, as the editor's objectRotation."""
    x, y, z = (math.radians(d) for d in degrees)
    return mul(mul(about((0, 1, 0), y), about((1, 0, 0), x)), about((0, 0, 1), z))


def held(forward):
    """How the board is held: its x along the view's `forward` laid flat, its y to the left of it, its z up - the turn
    from the board's axes to the world's (Y up)."""
    fx, fz = forward[0], forward[2]
    n = math.hypot(fx, fz) or 1.0
    fx, fz = fx / n, fz / n
    # columns: x -> (fx, 0, fz), y -> (fz, 0, -fx) (up × forward), z -> (0, 1, 0)
    return matrix(((fx, fz, 0.0), (0.0, 0.0, 1.0), (fz, -fx, 0.0)))


def matrix(m):
    """A rotation matrix (rows) as a quaternion - Shepperd's way: from its largest diagonal term, so no angle is lost."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        return (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    k = max(range(3), key=lambda i: m[i][i])
    i, j, l = k, (k + 1) % 3, (k + 2) % 3
    s = 2 * math.sqrt(1 + m[i][i] - m[j][j] - m[l][l])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0], q[1 + i], q[1 + j], q[1 + l] = (m[l][j] - m[j][l]) / s, s / 4, (m[j][i] + m[i][j]) / s, (m[l][i] + m[i][l]) / s
    return tuple(q)


def wanted(q, zero, mount):
    """The turn the hand is to make in the world: the board's since its zero pose, in the board's own frame (zero⁻¹ · q),
    carried into the world by how it is held."""
    return mul(mul(mount, mul(conj(zero), q)), conj(mount))


def solve(turn, axes, start, steps=4):
    """The three joint angles whose rotations about `axes` (world, the arm at rest), composed in turn, give `turn` - Newton
    from `start`, each step at most half a radian."""
    th = list(start)
    for _ in range(steps):
        got = mul(mul(about(axes[0], th[0]), about(axes[1], th[1])), about(axes[2], th[2]))
        err = log(mul(turn, conj(got)))  # what is left, in the world
        if math.sqrt(sum(e * e for e in err)) < 1e-6:
            break
        q1 = about(axes[0], th[0])
        cols = (axes[0], rotate(q1, axes[1]), rotate(mul(q1, about(axes[1], th[1])), axes[2]))  # the axes where they are now
        step = _solve3(cols, err)
        if step is None:
            break
        top = max(abs(s) for s in step)
        scale = 0.5 / top if top > 0.5 else 1.0
        th = [t + s * scale for t, s in zip(th, step)]
    return th


def _solve3(cols, b):
    """x with x0·c0 + x1·c1 + x2·c2 = b (Cramer; a little damping), or None at a singularity."""
    (a, d, g), (b1, e, h), (c, f, i) = cols  # columns
    det = a * (e * i - f * h) - b1 * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-9:
        return None

    def with_b(k):
        m = [list(cols[0]), list(cols[1]), list(cols[2])]
        m[k] = list(b)
        (p, q, r), (s, t, u), (v, w, x) = m
        return p * (t * x - u * w) - s * (q * x - r * w) + v * (q * u - r * t)

    return [with_b(k) / det for k in range(3)]


def smooth(shown, wanted_angles, dt):
    """Where the joints are told to be after `dt` s on their way from `shown` to `wanted_angles` (first-order lag, TAU)."""
    share = 1 - math.exp(-max(dt, 0.0) / TAU)
    return [s + (w - s) * share for s, w in zip(shown, wanted_angles)]
