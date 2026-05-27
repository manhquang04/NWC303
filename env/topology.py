"""Mininet topology builder: 3 switches, 6 hosts, remote Ryu controller."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

try:
    from mininet.net import Mininet
    from mininet.node import Controller, OVSSwitch, RemoteController
    from mininet.link import TCLink
    from mininet.log import setLogLevel
    from mininet.cli import CLI
    _MININET_AVAILABLE = True
except ImportError:  # pragma: no cover
    Mininet = object  # type: ignore[assignment,misc]
    RemoteController = object  # type: ignore[assignment,misc]
    OVSSwitch = object  # type: ignore[assignment,misc]
    TCLink = object  # type: ignore[assignment,misc]
    _MININET_AVAILABLE = False

from config import CFG

log = logging.getLogger(__name__)


class SDNTopology:
    """Mininet topology builder and lifecycle manager."""

    def __init__(
        self,
        controller_ip: str = CFG.ryu.controller_ip,
        controller_port: int = CFG.ryu.openflow_port,
    ) -> None:
        if not _MININET_AVAILABLE:
            raise RuntimeError(
                "Mininet not available. Install via: sudo apt-get install mininet"
            )
        self._check_root()
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net: Optional[Mininet] = None
        self.switches: List = []
        self.hosts: List = []

    @staticmethod
    def _check_root() -> None:
        if os.geteuid() != 0:
            raise RuntimeError(
                "Mininet requires root privileges.\n"
                "Run:  sudo python3 main.py train\n"
                "or:   sudo python3 env/topology.py"
            )

    def _build(self) -> None:
        topo_cfg = CFG.topology
        log.info(
            "Building topology: %d switches, %d hosts, controller=%s:%d",
            topo_cfg.num_switches, topo_cfg.num_hosts,
            self.controller_ip, self.controller_port,
        )

        self.net = Mininet(
            controller=RemoteController,
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            build=False,
        )

        self.net.addController(
            "c0",
            controller=RemoteController,
            ip=self.controller_ip,
            port=self.controller_port,
        )

        for i in range(1, topo_cfg.num_switches + 1):
            sw = self.net.addSwitch(f"s{i}", protocols="OpenFlow13")
            self.switches.append(sw)

        for i in range(1, topo_cfg.num_hosts + 1):
            h = self.net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
            self.hosts.append(h)

        link_opts = dict(
            bw=topo_cfg.link_bw_mbps,
            delay=topo_cfg.link_delay,
            loss=topo_cfg.link_loss_pct,
        )
        hosts_per_sw = max(1, topo_cfg.num_hosts // topo_cfg.num_switches)
        for idx, host in enumerate(self.hosts):
            sw_idx = min(idx // hosts_per_sw, len(self.switches) - 1)
            self.net.addLink(host, self.switches[sw_idx], **link_opts)

        for i in range(len(self.switches) - 1):
            self.net.addLink(self.switches[i], self.switches[i + 1], **link_opts)

    def start(self) -> None:
        if self.net is None:
            self._build()
        assert self.net is not None
        log.info("Starting Mininet ...")
        self.net.build()
        self.net.start()
        log.info(
            "Mininet up: switches=%s hosts=%s",
            [s.name for s in self.switches],
            [h.name for h in self.hosts],
        )

    def stop(self) -> None:
        if self.net is not None:
            log.info("Stopping Mininet ...")
            try:
                self.net.stop()
            except Exception:  # pragma: no cover
                log.exception("Mininet.stop() raised — continuing cleanup.")
        self.net = None
        self.switches.clear()
        self.hosts.clear()

    def reset(self) -> None:
        self.stop()
        self.start()

    def get_hosts(self) -> List:
        return list(self.hosts)

    def get_switches(self) -> List:
        return list(self.switches)

    def get_host_by_name(self, name: str) -> Optional[object]:
        for h in self.hosts:
            if getattr(h, "name", None) == name:
                return h
        return None

    def cli(self) -> None:
        if self.net is None:
            raise RuntimeError("Network not started.")
        log.info("Opening Mininet CLI (Ctrl-D / 'exit' to quit).")
        CLI(self.net)


def main() -> None:  # pragma: no cover
    setLogLevel("info")
    topo = SDNTopology()
    topo.start()
    topo.cli()
    topo.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
