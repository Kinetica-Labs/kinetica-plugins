"""A game controller on Linux (/dev/input/js*) by the standard library, in the words of a controller's bindings
(the editor's src/scene/SceneOpsControllers.cpp: left_stick, left_x, right_y, left_trigger, a, b, x, y, left_bumper,
start, back...). What each axis and button is comes from the device itself (ioctl JSIOCGAXMAP, JSIOCGBTNMAP: the Linux
input codes), not from their order: a Switch Pro, an Xbox or a DualSense pad read alike. A stick's y is up positive, an
analog trigger 0..1; a stick's few hundredths around the middle are 0. The other systems: none yet."""
import array
import glob
import os
import select
import struct
import threading

try:
    import fcntl
except ImportError:  # not a Unix
    fcntl = None

DEAD = 0.08  # a stick's share around its middle read as 0: a resting stick is seldom at 0


def _ior(number, size):
    return (2 << 30) | (size << 16) | (ord("j") << 8) | number


JSIOCGAXMAP, JSIOCGBTNMAP = _ior(0x32, 64), _ior(0x34, 1024)
JSIOCGNAME = _ior(0x13, 128)

AXES = {0x00: ("left_x", 1), 0x01: ("left_y", -1), 0x03: ("right_x", 1), 0x04: ("right_y", -1),
        0x02: ("left_trigger", 0), 0x05: ("right_trigger", 0)}  # sign; 0: a trigger, rest -1 → 0..1
BUTTONS = {0x130: "a", 0x131: "b", 0x133: "y", 0x134: "x", 0x136: "left_bumper", 0x137: "right_bumper",
           0x138: "left_trigger", 0x139: "right_trigger", 0x13a: "back", 0x13b: "start"}  # BTN_SOUTH... by place


class Pad:
    """A pad's controls now. `axes` and `buttons`: the device's numbers → Linux input codes."""

    def __init__(self, axes, buttons, name="", path=""):
        self.axes, self.buttons, self.name, self.path = list(axes), list(buttons), name, path
        self.state, self.lost, self.fd = {}, False, None

    def feed(self, data):
        """Raw js_event bytes (u32 time, s16 value, u8 type, u8 number; type 0x80: the state on opening) into the state."""
        for k in range(0, len(data) - 7, 8):
            _, value, kind, number = struct.unpack("<IhBB", data[k:k + 8])
            kind &= 0x7F
            if kind == 2 and number < len(self.axes) and self.axes[number] in AXES:
                word, sign = AXES[self.axes[number]]
                v = value / 32767.0
                self.state[word] = (v + 1) / 2 if sign == 0 else (0.0 if abs(v) < DEAD else sign * v)
            elif kind == 1 and number < len(self.buttons) and self.buttons[number] in BUTTONS:
                self.state[BUTTONS[self.buttons[number]]] = float(value)

    def value(self, word):
        """A control's value: a stick (left_stick, right_stick) as (x, y), the rest a number."""
        if word in ("left_stick", "right_stick"):
            side = word.split("_")[0]
            return self.state.get(side + "_x", 0.0), self.state.get(side + "_y", 0.0)
        return self.state.get(word, 0.0)

    @classmethod
    def open(cls, path):
        """The pad at `path` (a js device), read in its own thread until close() or until it is unplugged (lost)."""
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        axes, buttons, name = array.array("B", [0] * 64), array.array("H", [0] * 512), array.array("B", [0] * 128)
        for request, buf in ((JSIOCGAXMAP, axes), (JSIOCGBTNMAP, buttons), (JSIOCGNAME, name)):
            fcntl.ioctl(fd, request, buf)
        pad = cls(axes, buttons, bytes(name).split(b"\0")[0].decode("utf-8", "replace"), path)
        pad.fd, pad._stop = fd, threading.Event()
        threading.Thread(target=pad._read, daemon=True).start()
        return pad

    def _read(self):
        while not self._stop.is_set():
            try:
                if select.select([self.fd], [], [], 0.2)[0]:
                    data = os.read(self.fd, 8 * 64)
                    if not data:
                        raise OSError("closed")
                    self.feed(data)
            except BlockingIOError:
                continue
            except OSError:
                self.lost = True
                break
        os.close(self.fd)

    def close(self):
        if self.fd is not None:
            self._stop.set()


def find():
    """The first game controller's js device, or None: none plugged in, or not Linux."""
    found = sorted(glob.glob("/dev/input/js*"))
    return found[0] if found and fcntl else None
