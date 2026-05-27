"""VLAN quarantine management via Ryu REST API."""

from __future__ import annotations

import logging
from typing import Dict, Set, Tuple

import requests

from config import CFG
from isolation.flow_rules import make_match_dict, make_vlan_rule

log = logging.getLogger(__name__)


class VLANManager:

    def __init__(
        self,
        base_url: str = CFG.ryu.rest_base_url,
        vlan_id: int = CFG.isolation.quarantine_vlan_id,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.vlan_id = vlan_id
        self.quarantined: Set[Tuple[int, int]] = set()

    def quarantine(self, dpid: int, port: int) -> bool:
        if (dpid, port) in self.quarantined:
            log.debug("(%d,%d) already quarantined.", dpid, port)
            return True
        rule = make_vlan_rule(dpid, port, vlan_id=self.vlan_id)
        if self._post("/stats/flowentry/add", rule):
            self.quarantined.add((dpid, port))
            log.info("Quarantined (dpid=%d, port=%d) → VLAN %d",
                     dpid, port, self.vlan_id)
            return True
        return False

    def release(self, dpid: int, port: int) -> bool:
        if (dpid, port) not in self.quarantined:
            return True
        body = make_match_dict(dpid, port)
        if self._delete("/stats/flowentry/delete", body):
            self.quarantined.discard((dpid, port))
            log.info("Released (dpid=%d, port=%d)", dpid, port)
            return True
        return False

    def is_quarantined(self, dpid: int, port: int) -> bool:
        return (dpid, port) in self.quarantined

    def _post(self, path: str, body: Dict) -> bool:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=body, timeout=CFG.ryu.request_timeout_sec)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("POST %s failed: %s", url, exc)
            return False

    def _delete(self, path: str, body: Dict) -> bool:
        url = f"{self.base_url}{path}"
        try:
            r = requests.delete(url, json=body, timeout=CFG.ryu.request_timeout_sec)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("DELETE %s failed: %s", url, exc)
            return False
