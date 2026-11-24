"""The orientation board turns the arm's wrist (plugin.py's «Запястье за датчиком»), each of its angles one joint: the
board's turn since the press, in its own axes, split into three turns about its axes - each the axis that, as the board is
held (x to the right of the view, y into it, z up), lies nearest one of the wrist's joints (match) - and each added to that
joint. A turn about one axis of the board moves one joint. Their rotations about their axes (the arm at rest, the product of
exponentials) composed must give that turn: a few damped least squares steps from the last frame's angles - no jumps,
no spinning at the wrist's singularity, any three axes.
No editor here: the arithmetic only, tested by test_depz.py."""
import itertools
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


def held(forward, turn=0.0):
    """How the board is held: its x to the right of the view, its y along the view's `forward` laid flat, its z up;
    `turn` - degrees the board is turned about its z from that. The turn from the board's axes to the world's (Y up)."""
    fx, fz = forward[0], forward[2]
    n = math.hypot(fx, fz) or 1.0
    fx, fz = fx / n, fz / n
    # columns: x -> right (-fz, 0, fx), y -> forward (fx, 0, fz), z -> up (0, 1, 0)
    return mul(matrix(((-fz, fx, 0.0), (0.0, 0.0, 1.0), (fx, fz, 0.0))), about((0.0, 0.0, 1.0), math.radians(turn)))


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


def match(mount, joint_axes):
    """Which of the board's axes leads which joint of the wrist: each board axis, carried into the world by how it is held
    (`mount`), to the joint whose axis at rest it lies nearest - the one assignment of the three that fits best - and
    signed so that turning the board one way turns the joint the same way. The board's axes, signed, in the joints' order."""
    board = [rotate(mount, e) for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    order = max(itertools.permutations(range(3)), key=lambda p: sum(abs(dot(board[p[i]], joint_axes[i])) for i in range(3)))
    unit = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return [tuple(v * (1 if dot(board[k], joint_axes[i]) >= 0 else -1) for v in unit[k]) for i, k in enumerate(order)]


DAMPING, STEP = 0.15, 0.3  # damped least squares (Wampler, Nakamura): near a singularity the wrist lags, never spins; rad a step at most


def solve(turn, axes, start, steps=6):
    """The three joint angles whose rotations about `axes` (world, the arm at rest), composed in turn, give `turn` - from
    `start` by damped least squares, each step at most STEP: where two of the wrist's axes come into line (the wrist's
    singularity) the hand turns as far as it can and waits, instead of the two joints spinning against each other."""
    th = list(start)
    for _ in range(steps):
        got = mul(mul(about(axes[0], th[0]), about(axes[1], th[1])), about(axes[2], th[2]))
        err = log(mul(turn, conj(got)))  # what is left, in the world
        if math.sqrt(sum(e * e for e in err)) < 1e-7:
            break
        q1 = about(axes[0], th[0])
        cols = (axes[0], rotate(q1, axes[1]), rotate(mul(q1, about(axes[1], th[1])), axes[2]))  # the axes where they are now
        jjt = [[sum(c[r] * c[k] for c in cols) + (DAMPING ** 2 if r == k else 0.0) for k in range(3)] for r in range(3)]
        y = _gauss(jjt, list(err))
        step = [sum(c[r] * y[r] for r in range(3)) for c in cols]
        top = max(abs(v) for v in step)
        scale = STEP / top if top > STEP else 1.0
        th = [t + v * scale for t, v in zip(th, step)]
    return th


def _gauss(a, b):
    """x of a x = b for a 3x3 symmetric positive definite a (J Jᵀ + λ² I): elimination with partial pivoting."""
    m = [row[:] + [v] for row, v in zip(a, b)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, 3):
            f = m[r][c] / m[c][c]
            m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][k] * x[k] for k in range(r + 1, 3))) / m[r][r]
    return x


def smooth(shown, wanted_angles, dt):
    """Where the joints are told to be after `dt` s on their way from `shown` to `wanted_angles` (first-order lag, TAU)."""
    share = 1 - math.exp(-max(dt, 0.0) / TAU)
    return [s + (w - s) * share for s, w in zip(shown, wanted_angles)]
