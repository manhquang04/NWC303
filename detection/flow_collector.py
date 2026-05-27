"""Poll Ryu REST API and store network snapshots. Thread-safe."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from config import CFG

log = logging.getLogger(__name__)


@dataclass
class NetworkSnapshot:
    """Network state at a point in time."""

    timestamp: float
    flow_stats: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    port_stats: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    arp_table: Dict[str, str] = field(default_factory=dict)
    mac_table: Dict[int, Dict[str, int]] = field(default_factory=dict)


class FlowCollector:
    """Background poller fetching snapshots from Ryu REST API."""

    def __init__(
        self,
        dpids: List[int],
        base_url: str = CFG.ryu.rest_base_url,
        poll_interval_ms: int = CFG.detection.poll_interval_ms,
        timeout_sec: float = CFG.ryu.request_timeout_sec,
    ) -> None:
        self.dpids = list(dpids)
        self.base_url = base_url.rstrip("/")
        self.poll_interval_sec = poll_interval_ms / 1000.0
        self.timeout = timeout_sec

        self._latest: Optional[NetworkSnapshot] = None
        self._history: List[NetworkSnapshot] = []
        self._max_history = 60

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="FlowCollector", daemon=True)
        self._thread.start()
        log.info("FlowCollector started (poll=%.2fs)", self.poll_interval_sec)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        log.info("FlowCollector stopped.")

    def get_latest(self) -> Optional[NetworkSnapshot]:
        with self._lock:
            return self._latest

    def get_history(self) -> List[NetworkSnapshot]:
        with self._lock:
            return list(self._history)

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                snap = self._collect_once()
                with self._lock:
                    self._latest = snap
                    self._history.append(snap)
                    if len(self._history) > self._max_history:
                        self._history.pop(0)
            except Exception:  # pragma: no cover
                log.exception("FlowCollector poll iteration failed.")
            elapsed = time.time() - t0
            self._stop_evt.wait(max(0.0, self.poll_interval_sec - elapsed))

    def _collect_once(self) -> NetworkSnapshot:
        snap = NetworkSnapshot(timestamp=time.time())
        for dpid in self.dpids:
            snap.flow_stats[dpid] = self._get_json(f"/stats/flow/{dpid}", default=[])
            snap.port_stats[dpid] = self._get_json(f"/stats/port/{dpid}", default=[])
        snap.arp_table = self._get_json("/sdnids/arp_table", default={})
        snap.mac_table = self._get_json("/sdnids/mac_table", default={})
        return snap

    def _get_json(self, path: str, default: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.debug("GET %s failed: %s", url, exc)
            return default
