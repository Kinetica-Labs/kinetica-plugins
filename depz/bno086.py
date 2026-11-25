"""The DEPZ-IMU-BNO086 board (depz.ai) over USB: its open transport protocol (v0.62 - packets A5 C3, the header checked
by CRC-8/MAXIM) carries the sensor's own SHTP frames, whose input reports (CEVA SH-2) are the samples. Here: building
the host's packets, cutting the board's stream into packets, decoding the reports, opening the port and reading it on a
thread. Standard library only. plugin.py (this folder) keeps and shows the samples; test_depz.py checks this file
against bytes worked by hand."""
import glob
import math
import os
import queue
import struct
import threading
import time

MAGIC = b"\xa5\xc3"
GET_DEVICE_NAME, GET_SERIAL, SEND_SHTP = 0x03, 0x05, 0x34     # the host's commands
RPT_STATUS, RPT_TEXT, RPT_DATA = 0x80, 0x81, 0x91             # the board's reports
ERR_BUSY = 0x06

# SH-2 input reports: id -> (stream, the report's length in bytes, Q point, values). The orientation (rotation vector -
# with the compass -, or the game rotation vector - without it) comes as i j k real and goes out as w x y z.
REPORTS = {0x01: ("accel", 10, 8, 3),    # m/s², gravity in it
           0x02: ("gyro", 10, 9, 3),     # rad/s, calibrated
           0x05: ("quat", 14, 14, 4),    # + the heading's accuracy estimate, not read
           0x08: ("quat", 12, 14, 4)}
# The others' lengths only, to step over them when the board was set up for more than this reads (it keeps its setup).
OTHERS = {0x03: 10, 0x04: 10, 0x06: 10, 0x07: 16, 0x09: 14, 0x10: 5, 0x11: 12, 0x12: 6, 0x13: 6, 0x14: 16, 0x15: 16,
          0x16: 14, 0x18: 8, 0x19: 6, 0x1C: 6, 0x1E: 16, 0x28: 14, 0x29: 12}


def crc8(data):
    """CRC-8/MAXIM: poly 0x31 reflected, init 0 - the packet header's check."""
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ 0x8C if c & 1 else c >> 1
    return c


def packet(cmd, seq, data=b""):
    """A host packet: magic, size (no payload CRC), command, sequence, the header's CRC, the data."""
    head = struct.pack("<HBB", len(data), cmd, seq & 0xFF)
    return MAGIC + head + bytes([crc8(head)]) + data


def set_feature(report, interval_us, seq):
    """An SHTP frame for the sensor's control channel (2): Set Feature (0xFD) - send `report` every `interval_us`, 0 stops it."""
    body = struct.pack("<BBBHIII", 0xFD, report, 0, 0, interval_us, 0, 0)
    return struct.pack("<HBB", 4 + len(body), 2, seq & 0xFF) + body


class Packets:
    """The board's bytes in pieces of any size -> its whole packets (cmd, seq, data). A header failing its CRC is a false
    start: one byte is dropped and the magic looked for again. The payload CRC, if the board was set to add one, is
    skipped (USB checks its own)."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                del self.buf[:-1 if self.buf.endswith(MAGIC[:1]) else len(self.buf)]
                return out
            del self.buf[:i]
            if len(self.buf) < 7:
                return out
            if crc8(self.buf[2:6]) != self.buf[6]:
                del self.buf[:1]
                continue
            size, cmd, seq = struct.unpack_from("<HBB", self.buf, 2)
            n = size & 0x3FFF
            end = 7 + n + (0, 1, 2, 4)[size >> 14]
            if len(self.buf) < end:
                return out
            out.append((cmd, seq, bytes(self.buf[7:7 + n])))
            del self.buf[:end]


def samples(frame, ref_us):
    """One SHTP frame of RPT_DATA -> [(stream, t_us, values, accuracy 0..3)]. Only the input channels (3, 4); a report's
    time is the board's clock at capture (`ref_us`) less the frame's base delta (0xFB, a rebase 0xFA moves it) plus the
    report's own delay, all in 100 µs ticks. A report of an unknown id ends the frame: its length is unknown."""
    if len(frame) < 4 or frame[2] not in (3, 4):
        return []
    end = min(struct.unpack_from("<H", frame)[0] & 0x7FFF, len(frame))
    i, base, out = 4, ref_us, []
    while i < end:
        rid = frame[i]
        if rid in (0xFB, 0xFA) and i + 5 <= end:
            delta = struct.unpack_from("<i", frame, i + 1)[0] * 100
            base = ref_us - delta if rid == 0xFB else base + delta
            i += 5
            continue
        kind = REPORTS.get(rid)
        size = kind[1] if kind else OTHERS.get(rid)
        if size is None or i + size > end:
            break
        if kind:
            stream, _, q, n = kind
            status = frame[i + 2]
            v = [r / (1 << q) for r in struct.unpack_from(f"<{n}h", frame, i + 4)]
            if n == 4:
                v = [v[3], v[0], v[1], v[2]]
            out.append((stream, base + ((status >> 2) << 8 | frame[i + 3]) * 100, v, status & 3))
        i += size
    return out


def euler(q):
    """A quaternion (w x y z) -> roll, pitch, yaw in radians (about x, y, z; z-y-x order)."""
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def find_port():
    """The board's port: on Linux its name under /dev/serial/by-id; on Windows the first USB serial port the registry
    lists (the board is a USB CDC device). None: not plugged in."""
    found = sorted(glob.glob("/dev/serial/by-id/*BNO086*"))
    if found:
        return found[0]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as key:
            for i in range(winreg.QueryInfoKey(key)[1]):
                name, value, _ = winreg.EnumValue(key, i)
                if "USBSER" in name.upper():
                    return r"\\.\{}".format(value)
    except (ImportError, OSError):
        pass
    return None


class Port:
    """The serial port as bytes in and out. Linux: raw, a read waits 0.1 s at most. Windows: a COM port is a file, read
    in small pieces (a read waits until its bytes come)."""

    def __init__(self, path):
        self.path = path
        try:
            import termios
            import tty
        except ImportError:
            self.file = open(path, "r+b", buffering=0)
            return
        self.file = None
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[6][termios.VMIN], attrs[6][termios.VTIME] = 0, 1
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def read(self):
        if self.file:
            return self.file.read(64)
        data = os.read(self.fd, 4096)
        if not data and not os.path.exists(self.path):
            raise OSError("the board was unplugged")
        return data

    def write(self, data):
        if self.file:
            self.file.write(data)
        else:
            os.write(self.fd, data)

    def close(self):
        if self.file:
            self.file.close()
        else:
            os.close(self.fd)


class Board:
    """The open board: commands out; a thread reading it, handing every sample to `on_sample(stream, t_us, values,
    accuracy)` and the replies to the command that waits for them. `lost` counts the packets the board's sequence skipped."""

    def __init__(self, path, on_sample):
        self.port, self.on_sample, self.packets = Port(path), on_sample, Packets()
        self.seq = self.shtp_seq = 0
        self.lost, self.last = 0, None
        self.replies, self.stopped = queue.Queue(), threading.Event()
        self.thread = threading.Thread(target=self._read, name="depz-bno086", daemon=True)
        self.thread.start()

    def ask(self, cmd, data=b"", timeout=0.2):
        """Send a command and wait for the report naming it: (report, the rest of its data), None if none came in time
        (the protocol's 200 ms)."""
        self.port.write(packet(cmd, self.seq, data))
        self.seq = (self.seq + 1) & 0xFF
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                report, reply = self.replies.get(timeout=left)
            except queue.Empty:
                break
            if reply[:1] == bytes([cmd]):
                return report, reply[1:]
        return None

    def text(self, cmd):
        got = self.ask(cmd)
        return got[1].split(b"\0")[0].decode("ascii", "replace") if got and got[0] == RPT_TEXT else ""

    def feature(self, report, hz):
        """Start one of the sensor's reports at `hz` (0 stops it); True once the board has queued it. Busy (its two
        buffers full): back off a timeout and try again, as the protocol says."""
        for _ in range(3):
            got = self.ask(SEND_SHTP, set_feature(report, round(1e6 / hz) if hz else 0, self.shtp_seq))
            self.shtp_seq = (self.shtp_seq + 1) & 0xFF
            if got and got[0] == RPT_STATUS and got[1][:1] == b"\0":
                return True
            if got and got[1][:1] != bytes([ERR_BUSY]):
                return False
            time.sleep(0.2)
        return False

    def _read(self):
        try:
            while not self.stopped.is_set():
                for cmd, seq, data in self.packets.feed(self.port.read()):
                    if self.last is not None:
                        self.lost += (seq - self.last - 1) & 0xFF
                    self.last = seq
                    if cmd == RPT_DATA and len(data) >= 9:
                        for s in samples(data[9:], struct.unpack_from("<Q", data, 1)[0]):
                            self.on_sample(*s)
                    elif cmd in (RPT_STATUS, RPT_TEXT):
                        self.replies.put((cmd, data))
        except OSError:
            pass
        finally:
            self.stopped.set()

    def close(self):
        self.stopped.set()
        self.thread.join(1.0)
        self.port.close()
