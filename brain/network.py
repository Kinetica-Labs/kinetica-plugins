"""The brain's network (plugin.py has the rest: the commands, the robot's wiring, the run's tick; evolution.py breeds
it): numbers in from the robot's sensors, numbers out to its drives. Layers of any width and depth - a perceptron,
a deep one, or a recurrent one whose hidden layers also see their own values of the step before (an Elman network: a
memory of one step, enough to tell a speed from two positions; a loop needs that step of delay). One with an echo
sees its own outputs of the step before among its inputs - what it told its drives (an efference copy): a car's brain
knows the speed it asked for, which no sensor of it reads. Every weight and
bias is one number of one flat list, so the genetic algorithm breeds a brain as a list. Only the standard library."""
import json
import math
import random

def tr(text):
    """The window's language for a word shown to a person (kinetica.plugin.tr); the word itself where the plugin SDK is not at hand."""
    try:
        from kinetica.plugin import tr as words
    except ImportError:
        return text
    return words(text)


FORMAT, VERSION = "kinetica-brain", 1  # a trained brain's file, *.brain.json (brain_file)

ACTIVATIONS = {
    "tanh": math.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "relu": lambda x: x if x > 0.0 else 0.0,
}


class Net:
    """inputs -> hidden layers -> outputs. A neuron: its inputs' weighted sum and a bias, through the activation; the
    output layer through tanh always, so each output is a command from -1 to 1. Weights in the list neuron by neuron,
    layer by layer: a neuron's weights over what its layer sees (the layer before, then its own memory), then its bias."""

    def __init__(self, inputs, hidden, outputs, activation="tanh", recurrent=False, echo=False):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation is one of {', '.join(ACTIVATIONS)}, not {activation!r}")
        self.echo = int(outputs) if echo else 0  # its outputs of the step before, after its sensors' numbers
        self.sizes = [int(inputs) + self.echo] + [int(n) for n in hidden] + [int(outputs)]
        if min(self.sizes) < 1:
            raise ValueError(f"every layer needs a neuron: {self.sizes}")
        self.activation, self.recurrent = activation, bool(recurrent)
        self.reset()

    def layers(self):
        """(what a neuron sees, how many neurons) of each layer after the inputs."""
        out = []
        for k in range(1, len(self.sizes)):
            hidden = k < len(self.sizes) - 1
            out.append((self.sizes[k - 1] + (self.sizes[k] if self.recurrent and hidden else 0), self.sizes[k]))
        return out

    def count(self):
        """How many numbers its weights are."""
        return sum((seen + 1) * n for seen, n in self.layers())

    def reset(self):
        """The memory of a recurrent net back to zeros: a run begins."""
        self.memory = [[0.0] * n for n in self.sizes[1:-1]]
        self.seen = [[0.0] * n for n in self.sizes]

    def forward(self, weights, inputs):
        """The outputs for these inputs, by these weights; a recurrent net remembers its hidden values for the next."""
        if len(weights) != self.count():
            raise ValueError(f"{len(weights)} weights for a net of {self.count()}")
        if len(inputs) != self.sizes[0] - self.echo:
            raise ValueError(f"{len(inputs)} inputs for a net of {self.sizes[0] - self.echo}")
        act, x, at = ACTIVATIONS[self.activation], [float(v) for v in inputs] + (self.seen[-1] if self.echo else []), 0
        layers = self.layers()
        self.seen = [x]  # each layer's values of this step, the inputs first: what a picture of the net lights
        for k, (seen, n) in enumerate(layers):
            last = k == len(layers) - 1
            sees = x + (self.memory[k] if self.recurrent and not last else [])
            y = []
            for _ in range(n):
                s = weights[at + seen] + sum(w * v for w, v in zip(weights[at:at + seen], sees))
                y.append(math.tanh(s) if last else act(s))
                at += seen + 1
            if self.recurrent and not last:
                self.memory[k] = y
            self.seen.append(y)
            x = y
        return x

    def random_weights(self, rng):
        """A first brain: each weight drawn normal with a spread of 1 / sqrt(what its neuron sees), the biases zero."""
        out = []
        for seen, n in self.layers():
            spread = 1.0 / math.sqrt(seen)
            for _ in range(n):
                out.extend(stored(rng.gauss(0.0, spread)) for _ in range(seen))
                out.append(0.0)
        return out


def _glyph(activation, x, y, r, ink):
    """A neuron's activation drawn in it: its curve across the circle - tanh and sigmoid an S between two flats, relu a
    hinge - in `ink`, dark on a lit neuron, light on a dark one."""
    if activation == "relu":
        d = f"M{x - r * .65:.1f},{y + r * .35:.1f} L{x:.1f},{y + r * .35:.1f} L{x + r * .6:.1f},{y - r * .5:.1f}"
    else:
        lo = y + r * (0.45 if activation == "tanh" else 0.5)
        d = (f"M{x - r * .7:.1f},{lo:.1f} L{x - r * .32:.1f},{lo - r * .04:.1f} C{x - r * .02:.1f},{lo - r * .08:.1f} {x + r * .02:.1f},"
             f"{y - r * .41:.1f} {x + r * .32:.1f},{y - r * .41:.1f} L{x + r * .7:.1f},{y - r * .45:.1f}")
    return f'<path d="{d}" fill="none" stroke="{ink}" stroke-width="{max(0.8, r / 7):.1f}"/>'


def picture(net, weights, inputs=(), outputs=()):
    """The net as an SVG picture (the «Мозг» panel's): its layers left to right, a neuron a circle lit by what it holds
    now (net.seen: yellow excited, blue inhibited, dark near zero) with its activation's curve drawn in it, a weight a
    line - orange excites, blue inhibits, the stronger the thicker (the faintest left out); `inputs`, `outputs`: their
    words, the outputs' with their values."""
    sizes, layers = net.sizes, net.layers()
    height = 34 + 11.0 * max(max(sizes), 16)
    xs = [120 + k * (252 - 120) / (len(sizes) - 1) for k in range(len(sizes))] if len(sizes) > 1 else [120]
    steps = [min(26.0, (height - 34) / n) for n in sizes]  # a layer of few neurons: bigger ones, its curve readable
    places = [[(xs[k], 26 + (height - 34 - steps[k] * n) / 2 + steps[k] * (j + 0.5)) for j in range(n)] for k, n in enumerate(sizes)]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 {height:.0f}" font-family="sans-serif">']
    at, links = 0, []
    strongest = max((abs(w) for w in weights), default=1.0) or 1.0
    for k, (seen, n) in enumerate(layers):
        for i in range(n):
            for j in range(seen):
                w = weights[at + j]
                if j < sizes[k] and abs(w) > 0.12 * strongest:  # a recurrent net's memory links are not drawn
                    links.append((abs(w) / strongest, w > 0, places[k][j], places[k + 1][i]))
            at += seen + 1
    for m, up, (x1, y1), (x2, y2) in sorted(links):  # the weight's size its thickness, the strongest drawn last, on top
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{"#f0a040" if up else "#4a90e2"}" '
                   f'stroke-width="{0.2 + 3.8 * m * m:.2f}" stroke-opacity="{0.2 + 0.7 * m:.2f}"/>')
    values = getattr(net, "seen", [[0.0] * n for n in sizes])
    for k, n in enumerate(sizes):
        r = min(steps[k] * 0.42, 10.0)
        for j, (x, y) in enumerate(places[k]):
            v = values[k][j] if k < len(values) and j < len(values[k]) else 0.0
            a = min(1.0, abs(v))
            fill = (f"rgb({int(40 + 215 * a)},{int(40 + 190 * a)},{int(40 + 40 * a)})" if v >= 0
                    else f"rgb({int(40 + 30 * a)},{int(40 + 110 * a)},{int(40 + 200 * a)})")
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="#888" stroke-width="0.5"/>')
            if k > 0:
                out.append(_glyph("tanh" if k == len(sizes) - 1 else net.activation, x, y, r, "#111" if a > 0.5 else "#ddd"))
            words = inputs if k == 0 else outputs if k == len(sizes) - 1 else ()
            if j < len(words):
                left = k == 0
                anchor = ' text-anchor="end"' if left else ""
                out.append(f'<text x="{x + (-r - 3 if left else r + 3):.1f}" y="{y + 2.5:.1f}"{anchor} fill="#ccc" font-size="7">{words[j]}</text>')
    heads = [tr("входы {n}").format(n=sizes[0])] + [f"{n} · {net.activation}" + (tr(" · память") if net.recurrent else "") for n in sizes[1:-1]] + [f"{sizes[-1]} · tanh"]
    for x, word in zip(xs, heads):
        out.append(f'<text x="{x:.1f}" y="12" text-anchor="middle" fill="#eee" font-size="8">{word}</text>')
    out.append("</svg>")
    return "".join(out)


def brain_file(net, weights, card, wiring, trained=None):
    """A trained brain as one JSON document - a *.brain.json file: its net (what it sees, its layers, what it gives),
    the robot it was wired to (Wiring.said), its card (the «Мозг» component's values but the weights: the drive, its
    speeds and reflex, the rate), how it was trained, and its weights. Readable by eye and by any JSON reader."""
    return {"format": FORMAT, "version": VERSION,
            "net": {"inputs": net.sizes[0] - net.echo, "layers": net.sizes[1:-1], "outputs": net.sizes[-1], "activation": net.activation,
                    "recurrent": net.recurrent, "echo": bool(net.echo), "weights": net.count()},
            "wiring": wiring, "card": card, "trained": trained or {}, "weights": [stored(w) for w in weights]}


def read_brain_file(text):
    """The document of a *.brain.json file, checked: its format, a version this plugin reads, its weights' count."""
    doc = json.loads(text)
    if doc.get("format") != FORMAT:
        raise ValueError(f"not a brain's file: its format is {doc.get('format')!r}, not {FORMAT!r}")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"a brain's file of version {doc['version']}: this plugin reads up to {VERSION} - update it")
    if len(doc.get("weights", [])) != int(doc["net"]["weights"]):
        raise ValueError(f"the file has {len(doc.get('weights', []))} weights for a net of {doc['net']['weights']}")
    return doc


def hidden_of(text):
    """The hidden layers' widths from their words: "8" or "16, 8" (or the list the editor reads "16, 8" back as);
    "" or "0": none (the inputs straight to the outputs)."""
    items = text if isinstance(text, (list, tuple)) else str(text).strip("[] ").replace(";", ",").split(",")
    widths = [int(float(w)) for w in items if str(w).strip()]
    return [w for w in widths if w > 0]


def stored(w):
    """A weight as the scene keeps it: 6 significant digits (the scene's numbers are single precision). Training
    breeds weights already so, so the brain played is the very one trained."""
    return float(f"{w:.6g}")


def weights_text(weights):
    """The weights as one word of the scene file: numbers with 6 significant digits, joined by commas."""
    return ",".join(f"{w:.6g}" for w in weights)


def weights_of(value):
    """The weights from the scene: its word, or the list the editor reads it back as (single precision: rounded back)."""
    items = value if isinstance(value, (list, tuple)) else str(value).strip("[] ").split(",")
    return [stored(float(w)) for w in items if str(w).strip()]


def new_rng(seed):
    return random.Random(int(seed))
