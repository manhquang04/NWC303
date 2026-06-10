"""Network action execution helpers for block/isolate decisions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from config import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_FLAG,
    ACTION_ISOLATE,
    ACTION_NAMES,
    CFG,
)

log = logging.getLogger(__name__)


def make_match_dict(dpid: int, port: int) -> Dict[str, Any]:
    """Build a Ryu REST flow-delete match body."""
    return {"dpid": int(dpid), "match": {"in_port": int(port)}}


def make_drop_rule(
    dpid: int,
    port: int,
    priority: int = CFG.isolation.drop_rule_priority,
    idle_timeout: int = CFG.isolation.rule_idle_timeout_sec,
    hard_timeout: int = CFG.isolation.rule_hard_timeout_sec,
) -> Dict[str, Any]:
    """Build a DROP rule for packets entering ``port`` on switch ``dpid``."""
    return {
        "dpid": int(dpid),
        "priority": int(priority),
        "idle_timeout": int(idle_timeout),
        "hard_timeout": int(hard_timeout),
        "match": {"in_port": int(port)},
        "actions": [],
    }


def make_vlan_rule(
    dpid: int,
    port: int,
    vlan_id: int = CFG.isolation.quarantine_vlan_id,
    priority: int = CFG.isolation.vlan_rule_priority,
    idle_timeout: int = CFG.isolation.rule_idle_timeout_sec,
    hard_timeout: int = CFG.isolation.rule_hard_timeout_sec,
) -> Dict[str, Any]:
    """Build a quarantine VLAN rule for packets entering ``port``."""
    return {
        "dpid": int(dpid),
        "priority": int(priority),
        "idle_timeout": int(idle_timeout),
        "hard_timeout": int(hard_timeout),
        "match": {"in_port": int(port)},
        "actions": [
            {"type": "PUSH_VLAN", "ethertype": 0x8100},
            {"type": "SET_FIELD", "field": "vlan_vid", "value": 0x1000 | int(vlan_id)},
            {"type": "OUTPUT", "port": "NORMAL"},
        ],
    }


class VLANManager:
    """Manage quarantine VLAN rules through the Ryu REST API."""

    def __init__(
        self,
        base_url: str = CFG.ryu.rest_base_url,
        vlan_id: int = CFG.isolation.quarantine_vlan_id,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.vlan_id = int(vlan_id)
        self.quarantined: Set[Tuple[int, int]] = set()

    def quarantine(self, dpid: int, port: int) -> bool:
        """Install a VLAN quarantine rule for one switch port."""
        key = (int(dpid), int(port))
        if key in self.quarantined:
            return True
        rule = make_vlan_rule(dpid, port, vlan_id=self.vlan_id)
        if self._post("/stats/flowentry/add", rule):
            self.quarantined.add(key)
            log.info("Quarantined dpid=%d port=%d vlan=%d", dpid, port, self.vlan_id)
            return True
        return False

    def release(self, dpid: int, port: int) -> bool:
        """Remove quarantine state for one switch port."""
        key = (int(dpid), int(port))
        if key not in self.quarantined:
            return True
        if self._delete("/stats/flowentry/delete", make_match_dict(dpid, port)):
            self.quarantined.discard(key)
            return True
        return False

    def _post(self, path: str, body: Dict[str, Any]) -> bool:
        try:
            r = requests.post(f"{self.base_url}{path}", json=body, timeout=CFG.ryu.request_timeout_sec)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("POST %s failed: %s", path, exc)
            return False

    def _delete(self, path: str, body: Dict[str, Any]) -> bool:
        try:
            r = requests.delete(f"{self.base_url}{path}", json=body, timeout=CFG.ryu.request_timeout_sec)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("DELETE %s failed: %s", path, exc)
            return False


@dataclass
class IsolationEvent:
    """One executed mitigation action."""

    timestamp: float
    action: int
    dpid: Optional[int]
    port: Optional[int]
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class Isolator:
    """Dispatch DRL actions to Ryu REST and keep rollback history."""

    def __init__(
        self,
        base_url: str = CFG.ryu.rest_base_url,
        vlan_manager: Optional[VLANManager] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.vlan_manager = vlan_manager or VLANManager(base_url=base_url)
        self.history: List[IsolationEvent] = []
        self._target: Optional[Tuple[int, int]] = None

    def set_target(self, dpid: int, port: int) -> None:
        """Set the target switch port for the next block/isolate action."""
        self._target = (int(dpid), int(port))

    def apply(self, action: int) -> bool:
        """Apply an allow/flag/block/isolate action."""
        action = int(action)
        ts = time.time()
        if action == ACTION_ALLOW:
            self._record(ts, action, success=True)
            return True
        if action == ACTION_FLAG:
            log.info("[FLAG] target=%s", self._target)
            self._record(ts, action, success=True)
            return True
        if self._target is None:
            log.warning("apply(%s) without target; skipping", ACTION_NAMES[action])
            self._record(ts, action, success=False, note="no_target")
            return False

        dpid, port = self._target
        if action == ACTION_BLOCK:
            ok = self._block(dpid, port)
        elif action == ACTION_ISOLATE:
            ok = self.vlan_manager.quarantine(dpid, port)
        else:
            log.warning("Unknown action: %s", action)
            ok = False
        self._record(ts, action, success=ok, dpid=dpid, port=port)
        return ok

    def rollback_all(self) -> None:
        """Remove active block and quarantine rules installed by this isolator."""
        for ev in list(self.history):
            if not ev.success or ev.dpid is None or ev.port is None:
                continue
            if ev.action == ACTION_BLOCK:
                self._unblock(ev.dpid, ev.port)
            elif ev.action == ACTION_ISOLATE:
                self.vlan_manager.release(ev.dpid, ev.port)
        self.history.clear()

    def get_history(self) -> List[IsolationEvent]:
        """Return a copy of executed action history."""
        return list(self.history)

    def _block(self, dpid: int, port: int) -> bool:
        try:
            r = requests.post(
                f"{self.base_url}/stats/flowentry/add",
                json=make_drop_rule(dpid, port),
                timeout=CFG.ryu.request_timeout_sec,
            )
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("BLOCK failed: %s", exc)
            return False

    def _unblock(self, dpid: int, port: int) -> bool:
        try:
            r = requests.delete(
                f"{self.base_url}/stats/flowentry/delete",
                json=make_match_dict(dpid, port),
                timeout=CFG.ryu.request_timeout_sec,
            )
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("UNBLOCK failed: %s", exc)
            return False

    def _record(
        self,
        ts: float,
        action: int,
        success: bool,
        dpid: Optional[int] = None,
        port: Optional[int] = None,
        **metadata: Any,
    ) -> None:
        self.history.append(IsolationEvent(ts, action, dpid, port, success, metadata))
