"""Dispatch DRL actions to Ryu REST API. Tracks history for rollback."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from config import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_FLAG,
    ACTION_ISOLATE,
    ACTION_NAMES,
    CFG,
)
from isolation.flow_rules import make_drop_rule, make_match_dict
from isolation.vlan_manager import VLANManager

log = logging.getLogger(__name__)


@dataclass
class IsolationEvent:
    timestamp: float
    action: int
    dpid: Optional[int]
    port: Optional[int]
    success: bool
    metadata: Dict = field(default_factory=dict)


class Isolator:
    """Action dispatcher: integer action -> Ryu REST call."""

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
        self._target = (int(dpid), int(port))

    def apply(self, action: int) -> bool:
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
            log.warning("apply(%s) without target — skipping.", ACTION_NAMES[action])
            self._record(ts, action, success=False, note="no_target")
            return False

        dpid, port = self._target

        if action == ACTION_BLOCK:
            ok = self._block(dpid, port)
            self._record(ts, action, success=ok, dpid=dpid, port=port)
            return ok

        if action == ACTION_ISOLATE:
            ok = self.vlan_manager.quarantine(dpid, port)
            self._record(ts, action, success=ok, dpid=dpid, port=port)
            return ok

        log.warning("Unknown action: %s", action)
        return False

    def _block(self, dpid: int, port: int) -> bool:
        rule = make_drop_rule(dpid, port)
        url = f"{self.base_url}/stats/flowentry/add"
        try:
            r = requests.post(url, json=rule, timeout=CFG.ryu.request_timeout_sec)
            r.raise_for_status()
            log.info("BLOCK rule installed: dpid=%d port=%d", dpid, port)
            return True
        except requests.RequestException as exc:
            log.error("BLOCK failed: %s", exc)
            return False

    def _unblock(self, dpid: int, port: int) -> bool:
        body = make_match_dict(dpid, port)
        url = f"{self.base_url}/stats/flowentry/delete"
        try:
            r = requests.delete(url, json=body, timeout=CFG.ryu.request_timeout_sec)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("UNBLOCK failed: %s", exc)
            return False

    def rollback_all(self) -> None:
        """Release all active blocks and quarantines."""
        for ev in list(self.history):
            if not ev.success or ev.dpid is None or ev.port is None:
                continue
            if ev.action == ACTION_BLOCK:
                self._unblock(ev.dpid, ev.port)
            elif ev.action == ACTION_ISOLATE:
                self.vlan_manager.release(ev.dpid, ev.port)
        self.history.clear()
        log.info("All isolation actions rolled back.")

    def _record(self, ts: float, action: int, success: bool,
                dpid: Optional[int] = None, port: Optional[int] = None,
                **meta) -> None:
        self.history.append(
            IsolationEvent(timestamp=ts, action=action, dpid=dpid,
                           port=port, success=success, metadata=meta)
        )

    def get_history(self) -> List[IsolationEvent]:
        return list(self.history)
