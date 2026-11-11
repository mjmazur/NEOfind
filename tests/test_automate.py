import json
import socket
import time

from neowise import automate as A
from neowise import campaign as C


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _gate(tmp_path, mode="wait", clock=None, **kw):
    return A.SlaveGate(tmp_path, mode, delay_s=1800, log=lambda *a: None, clock=clock or Clock(), **kw)


def test_parse_both_byte_orders_and_reject_bad_magic():
    import struct
    assert A.parse(A.pack(A.SLAVEMSG_DETBEG, "night 1")) == (A.SLAVEMSG_DETBEG, "night 1")
    assert A.parse(struct.pack(">II64s", A.MAGIC, A.SLAVEMSG_DETEND, b"x")) == (A.SLAVEMSG_DETEND, "x")
    assert A.parse(struct.pack("<II64s", 1234, 2, b"")) == (None, None)
    assert A.parse(b"short") == (None, None)


def test_automate_waits_for_detend_then_the_delay(tmp_path):
    clk = Clock()
    g = _gate(tmp_path, "wait", clk)
    assert not g.is_open() and "waiting for SLAVEMSG_DETEND" in g.describe()
    g.handle(A.SLAVEMSG_DETBEG)
    assert not g.is_open()
    g.handle(A.SLAVEMSG_DETEND)
    clk.t += 29 * 60
    assert not g.is_open() and "starting at" in g.describe()
    clk.t += 60
    assert g.is_open()
    g.handle(A.SLAVEMSG_DETBEG)  # the next night
    assert not g.is_open()
    g.handle(4)  # SLAVEMSG_TIMER changes nothing
    assert not g.is_open()


def test_automate_now_starts_open(tmp_path):
    g = _gate(tmp_path, "now")
    assert g.is_open()
    g.handle(A.SLAVEMSG_DETBEG)
    assert not g.is_open()


def test_a_quick_restart_carries_on_but_a_stale_state_waits(tmp_path):
    clk = Clock()
    g = _gate(tmp_path, "wait", clk)
    g.handle(A.SLAVEMSG_DETEND)
    clk.t += 1800
    assert g.is_open()
    clk.t += 60  # crash; systemd restarts it a minute later
    assert _gate(tmp_path, "wait", clk).is_open()
    clk.t += A.RESTORE_S + 1  # down for longer: the controller may have started a run meanwhile
    assert not _gate(tmp_path, "wait", clk).is_open()


def test_listener_acts_on_udp_from_the_controller_only(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    msgs = []
    g = A.SlaveGate(tmp_path, "now", port=port, controller_ip="10.9.9.9", delay_s=0, log=msgs.append)
    g.start()
    try:
        for _ in range(50):
            if g.sock is not None:
                break
            time.sleep(0.05)
        A.send(A.SLAVEMSG_DETBEG, "127.0.0.1", port, "begin")  # localhost is accepted, as in automate_night.py
        for _ in range(50):
            if not g.is_open():
                break
            time.sleep(0.05)
        assert not g.is_open()
        assert any("SLAVEMSG_DETBEG from 127.0.0.1 ('begin')" in m for m in msgs)
        A.send(A.SLAVEMSG_DETEND, "127.0.0.1", port)
        for _ in range(50):
            if g.is_open():
                break
            time.sleep(0.05)
        assert g.is_open()  # delay 0
        assert json.loads((tmp_path / A.STATE_NAME).read_text())["state"] == "open"
    finally:
        g.close()


def test_campaign_waits_while_the_gate_is_closed(tmp_path):
    camp = C.Campaign(tmp_path / "camp", C.Settings(hours="0-0:01", yield_gpu=False, automate="wait"))
    camp.gate = _gate(camp.out, "wait")
    camp._update_active()
    assert camp.state == "waiting" and "SLAVEMSG_DETEND" in camp.reason
    camp.gate = _gate(camp.out, "now")
    camp._update_active()
    assert camp.state == "running"  # --hours doesn't apply with --automate
