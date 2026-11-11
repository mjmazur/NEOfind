"""Run the campaign around the night's detection runs (``campaign run --automate``).

The observatory controller (10.24.8.10) announces each detection run with
SLAVEMSG UDP datagrams on port 17633, the protocol of
``IDS_tools/scheduling/automate_night.py``:

    struct struct_slavemsg_s {
        u_int32_t magic;    /* 739567286 */
        u_int32_t type;     /* 0..4, see MSG_TYPES */
        char      text[64];
    };

The programs it starts use the GPU, so the campaign stops (its searches are
killed, freeing the GPU) on SLAVEMSG_DETBEG and works again ``resume_delay``
(30 min) after SLAVEMSG_DETEND. Nothing can be asked about the current
state, so with ``--automate`` the campaign starts closed and waits for a
DETEND; with ``--automate-now`` it starts open and waits for a DETBEG. The
state is saved in ``OUT/automate.json`` with a heartbeat, so a campaign
restarted within ``RESTORE_S`` (a crash, systemd's restart) carries on from
it instead of waiting for the next DETEND.

The socket is bound like ``automate_night.py`` and ``listen_udp.py`` bind
theirs (SO_REUSEADDR, SO_REUSEPORT), so they can still start on this machine;
but the kernel hands a unicast datagram to only one of the sockets sharing a
port, so a warning is logged when another one is found on it.
"""

import json
import os
import socket
import struct
import threading
import time
from pathlib import Path

MAGIC = 739567286
MSG_SIZE = 72  # two uint32 and a 64-byte text field
FORMATS = ("<II64s", ">II64s")  # the controller's byte order isn't fixed
MSG_TYPES = {0: "SLAVEMSG_UNKNOWN", 1: "SLAVEMSG_CAMINIT", 2: "SLAVEMSG_DETBEG", 3: "SLAVEMSG_DETEND",
             4: "SLAVEMSG_TIMER"}
SLAVEMSG_DETBEG = 2
SLAVEMSG_DETEND = 3

UDP_PORT = 17633
CONTROLLER_IP = "10.24.8.10"
RESUME_DELAY_MIN = 30.0
STATE_NAME = "automate.json"
RESTORE_S = 300.0  # a saved state younger than this is still the controller's
HEARTBEAT_S = 30.0


def parse(data: bytes):
    """(type, text) of a SLAVEMSG datagram, or (None, None) without the magic."""
    for fmt in FORMATS:
        try:
            magic, msg_type, text = struct.unpack(fmt, data[:MSG_SIZE])
        except struct.error:
            continue
        if magic == MAGIC:
            return msg_type, text.split(b"\0")[0].decode("utf-8", errors="ignore").strip()
    return None, None


def pack(msg_type: int, text: str = "") -> bytes:
    """A SLAVEMSG datagram (for tests and ``neowise slavemsg``)."""
    return struct.pack("<II64s", MAGIC, msg_type, text.encode()[:63])


def type_name(msg_type) -> str:
    return MSG_TYPES.get(msg_type, f"UNKNOWN_{msg_type}")


def _hhmm(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t))


class SlaveGate:
    """Whether the campaign may use the GPU, from the controller's messages.

    ``mode`` "wait" (``--automate``): closed until a DETEND, then open
    ``delay_s`` later; "now" (``--automate-now``): open until a DETBEG."""

    def __init__(self, out, mode="wait", port=UDP_PORT, controller_ip=CONTROLLER_IP,
                 delay_s=RESUME_DELAY_MIN * 60, log=print, clock=time.time):
        if mode not in ("wait", "now"):
            raise ValueError(mode)
        self.path = Path(out) / STATE_NAME
        self.port, self.delay_s, self.log, self.clock = port, delay_s, log, clock
        self.allowed = {controller_ip, "127.0.0.1"}
        self.lock = threading.Lock()
        self.sock = None
        self.stop = threading.Event()
        self.state, self.since, self.open_at = ("open", clock(), None) if mode == "now" else ("start", clock(), None)
        saved = self._load()
        if mode == "wait" and saved:  # restarted: the controller's state is still the saved one
            self.state, self.since, self.open_at = saved["state"], saved["since"], saved.get("open_at")
            self.log(f"automate: carrying on from the saved state ({self.describe()})")
        else:
            self.log(f"automate: {self.describe()}")
        self._save()

    # ---- state
    def _load(self):
        try:
            s = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        fresh = self.clock() - s.get("heartbeat", 0) < RESTORE_S
        return s if fresh and s.get("state") in ("start", "busy", "cooldown", "open") else None

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps({"state": self.state, "since": self.since, "open_at": self.open_at,
                                   "heartbeat": self.clock()}))
        os.replace(tmp, self.path)

    def is_open(self) -> bool:
        with self.lock:
            if self.state == "cooldown" and self.clock() >= self.open_at:
                self.state, self.since = "open", self.clock()
                self._save()
                self.log("automate: the GPU is ours again")
            return self.state == "open"

    def describe(self) -> str:
        """Why the campaign is (not) running, for the status."""
        if self.state == "start":
            return "waiting for SLAVEMSG_DETEND before starting (--automate)"
        if self.state == "busy":
            return f"detection run since {_hhmm(self.since)} (SLAVEMSG_DETBEG)"
        if self.state == "cooldown":
            return f"SLAVEMSG_DETEND at {_hhmm(self.since)}; starting at {_hhmm(self.open_at)}"
        return "free to run (until SLAVEMSG_DETBEG)"

    def handle(self, msg_type: int, text: str = "", sender: str = "127.0.0.1") -> None:
        self.log(f"automate: {type_name(msg_type)} from {sender}" + (f" ('{text}')" if text else ""))
        with self.lock:
            now = self.clock()
            if msg_type == SLAVEMSG_DETBEG:
                self.state, self.since, self.open_at = "busy", now, None
            elif msg_type == SLAVEMSG_DETEND:
                self.state, self.since, self.open_at = "cooldown", now, now + self.delay_s
            else:
                return
            self._save()
        self.log(f"automate: {self.describe()}")

    # ---- listening
    def start(self) -> None:
        threading.Thread(target=self._listen, name="automate", daemon=True).start()

    def _bind(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        s.bind(("", self.port))
        s.settimeout(1.0)
        others = sockets_on_port(self.port) - 1
        if others > 0:
            self.log(f"automate: WARNING {others} other socket(s) on UDP port {self.port} (automate_night.py?): "
                     "the controller's messages reach only one of them")
        self.log(f"automate: listening on UDP port {self.port} for {sorted(self.allowed)}")
        return s

    def _listen(self) -> None:
        last_beat = 0.0
        while not self.stop.is_set():
            if self.sock is None:
                try:
                    self.sock = self._bind()
                except OSError as e:
                    self.log(f"automate: can't listen on UDP port {self.port} ({e}); retrying in 60 s")
                    self.stop.wait(60)
                    continue
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                data = None
            except OSError as e:
                self.log(f"automate: socket error ({e}); reopening")
                self.sock.close()
                self.sock = None
                continue
            if time.time() - last_beat > HEARTBEAT_S:
                with self.lock:
                    self._save()
                last_beat = time.time()
            if data is None:
                continue
            if addr[0] not in self.allowed:
                self.log(f"automate: ignored a datagram from {addr[0]} (not the controller)")
                continue
            msg_type, text = parse(data)
            if msg_type is None:
                self.log(f"automate: ignored a datagram from {addr[0]} without the SLAVEMSG magic")
                continue
            self.handle(msg_type, text, addr[0])
        if self.sock is not None:
            self.sock.close()

    def close(self) -> None:
        self.stop.set()


def sockets_on_port(port: int) -> int:
    """UDP sockets bound to ``port`` on this machine (all users')."""
    n = 0
    for name in ("/proc/net/udp", "/proc/net/udp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        n += sum(int(line.split()[1].rsplit(":", 1)[1], 16) == port for line in lines)
    return n


def send(msg_type: int, host: str = "127.0.0.1", port: int = UDP_PORT, text: str = "") -> None:
    """Send a SLAVEMSG datagram (``neowise slavemsg``: try the campaign by hand)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(pack(msg_type, text), (host, port))
