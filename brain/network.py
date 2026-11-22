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
