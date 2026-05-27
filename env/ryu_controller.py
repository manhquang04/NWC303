"""Ryu controller app: OpenFlow handlers, ARP/MAC learning, REST API."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict

from config import CFG

log = logging.getLogger(__name__)

SDN_IDS_INSTANCE_NAME = "sdn_ids_app"


try:
    from ryu.app.wsgi import ControllerBase, Response, WSGIApplication, route
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import (
        CONFIG_DISPATCHER,
        MAIN_DISPATCHER,
        set_ev_cls,
    )
    from ryu.lib.packet import arp, ethernet, packet
    from ryu.ofproto import ofproto_v1_3
    _RYU_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RYU_AVAILABLE = False


if not _RYU_AVAILABLE:

    class SDNIDSApp:  # type: ignore[no-redef]

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "Ryu not installed. Run: pip install ryu==4.34"
            )

    class SDNIDSRestController:  # type: ignore[no-redef]

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Ryu not installed.")

else:

    def _match_to_dict(of_match) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_name, value in of_match.items():
            if value is not None:
                result[field_name] = value
        return result

    def _actions_to_list(instructions) -> list:
        actions = []
        for inst in instructions:
            if hasattr(inst, "actions"):
                for a in inst.actions:
                    ad = {"type": a.__class__.__name__.replace("OFPAction", "")}
                    for attr in ("port", "max_len", "vlan_vid", "field", "value", "meter_id"):
                        if hasattr(a, attr):
                            ad[attr] = getattr(a, attr)
                    actions.append(ad)
        return actions

    def _dict_to_match(parser, match_dict: Dict[str, Any]):
        return parser.OFPMatch(**match_dict)

    def _list_to_actions(parser, ofp, actions_raw: list) -> list:
        result = []
        for a in actions_raw:
            a_type = a.get("type", "")
            if a_type == "OUTPUT":
                port = a.get("port", "NORMAL")
                if port == "NORMAL":
                    result.append(parser.OFPActionOutput(ofp.OFPP_NORMAL))
                elif port == "FLOOD":
                    result.append(parser.OFPActionOutput(ofp.OFPP_FLOOD))
                elif port == "CONTROLLER":
                    result.append(parser.OFPActionOutput(ofp.OFPP_CONTROLLER))
                else:
                    result.append(parser.OFPActionOutput(int(port)))
            elif a_type == "PUSH_VLAN":
                ethertype = a.get("ethertype", 0x8100)
                result.append(parser.OFPActionPushVlan(ethertype))
            elif a_type == "SET_FIELD":
                field_name = a.get("field", "")
                value = a.get("value", 0)
                if field_name == "vlan_vid":
                    result.append(parser.OFPActionSetField(vlan_vid=int(value)))
                else:
                    result.append(parser.OFPActionSetField(**{field_name: value}))
            elif a_type == "DROP":
                pass
            elif a_type == "METER":
                result.append(parser.OFPActionMeter(meter_id=int(a.get("meter_id", 1))))
            elif a_type == "":
                pass
            else:
                result.append(parser.OFPActionOutput(ofp.OFPP_NORMAL))
        return result

    class SDNIDSApp(app_manager.RyuApp):  # type: ignore[no-redef]

        OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
        _CONTEXTS = {"wsgi": WSGIApplication}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mac_to_port: Dict[int, Dict[str, int]] = {}
            self.arp_table: Dict[str, str] = {}
            self.datapaths: Dict[int, Any] = {}
            self._port_stats: Dict[int, List[Dict[str, Any]]] = {}
            self._flow_stats: Dict[int, List[Dict[str, Any]]] = {}
            self._port_stats_events: Dict[int, threading.Event] = {}
            self._flow_stats_events: Dict[int, threading.Event] = {}
            self._lock = threading.Lock()

            wsgi = kwargs.get("wsgi")
            if wsgi is not None:
                wsgi.register(SDNIDSRestController, {SDN_IDS_INSTANCE_NAME: self})

        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
        def switch_features_handler(self, ev) -> None:
            dp = ev.msg.datapath
            self.datapaths[dp.id] = dp
            self.mac_to_port.setdefault(dp.id, {})
            ofp = dp.ofproto
            parser = dp.ofproto_parser
            match = parser.OFPMatch()
            actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
            self._add_flow(dp, priority=0, match=match, actions=actions)
            log.info("Switch %d connected — table-miss installed.", dp.id)

        @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
        def packet_in_handler(self, ev) -> None:
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto
            parser = dp.ofproto_parser
            in_port = msg.match.get("in_port", None)
            if in_port is None:
                return

            pkt = packet.Packet(msg.data)
            eth = pkt.get_protocol(ethernet.ethernet)
            if eth is None:
                return

            src_mac = eth.src
            dst_mac = eth.dst

            with self._lock:
                self.mac_to_port[dp.id][src_mac] = in_port

            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt is not None:
                with self._lock:
                    self.arp_table[arp_pkt.src_ip] = src_mac
                log.debug("ARP learned: %s → %s", arp_pkt.src_ip, src_mac)

            with self._lock:
                out_port = self.mac_to_port[dp.id].get(dst_mac)

            if out_port is not None and out_port != in_port:
                actions = [parser.OFPActionOutput(out_port)]
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac)
                self._add_flow(dp, priority=10, match=match, actions=actions,
                               idle_timeout=30, hard_timeout=300)
                out = parser.OFPPacketOut(
                    datapath=dp, buffer_id=msg.buffer_id,
                    in_port=in_port, actions=actions, data=msg.data)
            else:
                out_port = ofp.OFPP_FLOOD
                out = parser.OFPPacketOut(
                    datapath=dp, buffer_id=msg.buffer_id,
                    in_port=in_port, actions=[parser.OFPActionOutput(out_port)],
                    data=msg.data)
            dp.send_msg(out)

        @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
        def port_stats_reply_handler(self, ev) -> None:
            dpid = ev.msg.datapath.id
            body = ev.msg.body
            ports = []
            for stat in body:
                ports.append({
                    "port_no": stat.port_no,
                    "rx_packets": stat.rx_packets,
                    "tx_packets": stat.tx_packets,
                    "rx_bytes": stat.rx_bytes,
                    "tx_bytes": stat.tx_bytes,
                    "rx_dropped": stat.rx_dropped,
                    "tx_dropped": stat.tx_dropped,
                    "rx_errors": stat.rx_errors,
                    "tx_errors": stat.tx_errors,
                    "duration_sec": stat.duration_sec,
                })
            with self._lock:
                self._port_stats[dpid] = ports
            if dpid in self._port_stats_events:
                self._port_stats_events[dpid].set()

        @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
        def flow_stats_reply_handler(self, ev) -> None:
            dpid = ev.msg.datapath.id
            body = ev.msg.body
            flows = []
            for stat in body:
                flows.append({
                    "priority": stat.priority,
                    "idle_timeout": stat.idle_timeout,
                    "hard_timeout": stat.hard_timeout,
                    "match": _match_to_dict(stat.match),
                    "actions": _actions_to_list(stat.instructions),
                    "packet_count": stat.packet_count,
                    "byte_count": stat.byte_count,
                    "duration_sec": stat.duration_sec,
                    "duration_nsec": stat.duration_nsec,
                })
            with self._lock:
                self._flow_stats[dpid] = flows
            if dpid in self._flow_stats_events:
                self._flow_stats_events[dpid].set()

        def _add_flow(self, datapath, priority: int, match, actions, **kwargs) -> None:
            ofp = datapath.ofproto
            parser = datapath.ofproto_parser
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=datapath, priority=priority, match=match,
                command=ofp.OFPFC_ADD, instructions=inst, **kwargs)
            datapath.send_msg(mod)

        def add_drop_rule(self, dpid: int, in_port: int, priority: int) -> bool:
            dp = self.datapaths.get(dpid)
            if dp is None:
                log.warning("add_drop_rule: dp %d not found.", dpid)
                return False
            ofp = dp.ofproto
            parser = dp.ofproto_parser
            match = parser.OFPMatch(in_port=int(in_port))
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [])]
            mod = parser.OFPFlowMod(
                datapath=dp, priority=int(priority), match=match,
                command=ofp.OFPFC_ADD, instructions=inst,
                idle_timeout=CFG.isolation.rule_idle_timeout_sec,
                hard_timeout=CFG.isolation.rule_hard_timeout_sec)
            dp.send_msg(mod)
            log.info("Drop rule added: dpid=%d port=%d priority=%d", dpid, in_port, priority)
            return True

        def delete_flow_rule(self, dpid: int, match: Dict[str, Any]) -> bool:
            dp = self.datapaths.get(dpid)
            if dp is None:
                log.warning("delete_flow_rule: dp %d not found.", dpid)
                return False
            ofp = dp.ofproto
            parser = dp.ofproto_parser
            of_match = _dict_to_match(parser, match)
            mod = parser.OFPFlowMod(
                datapath=dp, command=ofp.OFPFC_DELETE,
                out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                priority=CFG.isolation.drop_rule_priority, match=of_match)
            dp.send_msg(mod)
            log.info("Flow rule deleted: dpid=%d match=%s", dpid, match)
            return True

        def get_flow_stats_sync(self, dpid: int, timeout: float = 2.0) -> List[Dict[str, Any]]:
            dp = self.datapaths.get(dpid)
            if dp is None:
                return []
            evt = threading.Event()
            with self._lock:
                self._flow_stats_events[dpid] = evt
            parser = dp.ofproto_parser
            req = parser.OFPFlowStatsRequest(dp, match=parser.OFPMatch())
            dp.send_msg(req)
            evt.wait(timeout=timeout)
            with self._lock:
                self._flow_stats_events.pop(dpid, None)
                return list(self._flow_stats.get(dpid, []))

        def get_port_stats_sync(self, dpid: int, timeout: float = 2.0) -> List[Dict[str, Any]]:
            dp = self.datapaths.get(dpid)
            if dp is None:
                return []
            evt = threading.Event()
            with self._lock:
                self._port_stats_events[dpid] = evt
            parser = dp.ofproto_parser
            req = parser.OFPPortStatsRequest(dp, port_no=dp.ofproto.OFPP_ANY)
            dp.send_msg(req)
            evt.wait(timeout=timeout)
            with self._lock:
                self._port_stats_events.pop(dpid, None)
                return list(self._port_stats.get(dpid, []))

        def snapshot_arp_table(self) -> Dict[str, str]:
            with self._lock:
                return dict(self.arp_table)

        def snapshot_mac_table(self) -> Dict[int, Dict[str, int]]:
            with self._lock:
                return {dpid: dict(t) for dpid, t in self.mac_to_port.items()}

    class SDNIDSRestController(ControllerBase):  # type: ignore[no-redef]

        def __init__(self, req, link, data, **config):
            super().__init__(req, link, data, **config)
            self.app: SDNIDSApp = data[SDN_IDS_INSTANCE_NAME]

        @route("flow", "/stats/flow/{dpid}", methods=["GET"])
        def get_flow_stats(self, req, **kwargs):
            try:
                dpid = int(kwargs["dpid"])
            except (KeyError, ValueError):
                return Response(status=400, content_type="application/json",
                                body=json.dumps({"error": "invalid dpid"}))
            flows = self.app.get_flow_stats_sync(dpid)
            body = json.dumps({str(dpid): flows})
            return Response(content_type="application/json", body=body)

        @route("port", "/stats/port/{dpid}", methods=["GET"])
        def get_port_stats(self, req, **kwargs):
            try:
                dpid = int(kwargs["dpid"])
            except (KeyError, ValueError):
                return Response(status=400, content_type="application/json",
                                body=json.dumps({"error": "invalid dpid"}))
            ports = self.app.get_port_stats_sync(dpid)
            body = json.dumps({str(dpid): ports})
            return Response(content_type="application/json", body=body)

        @route("flowentry_add", "/stats/flowentry/add", methods=["POST"])
        def add_flow_entry(self, req, **_):
            try:
                body = json.loads(req.body.decode("utf-8") if isinstance(req.body, bytes) else req.body)
            except (json.JSONDecodeError, Exception):
                return Response(status=400, content_type="application/json",
                                body=json.dumps({"error": "invalid JSON body"}))
            dpid = body.get("dpid")
            priority = body.get("priority", 100)
            match = body.get("match", {})
            actions_raw = body.get("actions", [])
            idle_timeout = body.get("idle_timeout", CFG.isolation.rule_idle_timeout_sec)
            hard_timeout = body.get("hard_timeout", CFG.isolation.rule_hard_timeout_sec)
            if dpid is None:
                return Response(status=400, content_type="application/json",
                                body=json.dumps({"error": "missing dpid"}))

            dp = self.app.datapaths.get(int(dpid))
            if dp is None:
                return Response(status=404, content_type="application/json",
                                body=json.dumps({"error": f"switch {dpid} not found"}))

            ofp = dp.ofproto
            parser = dp.ofproto_parser
            of_match = _dict_to_match(parser, match)
            actions = _list_to_actions(parser, ofp, actions_raw)
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=dp, priority=int(priority), match=of_match,
                command=ofp.OFPFC_ADD, instructions=inst,
                idle_timeout=int(idle_timeout), hard_timeout=int(hard_timeout))
            dp.send_msg(mod)
            return Response(content_type="application/json",
                            body=json.dumps({"status": "ok", "dpid": dpid}))

        @route("flowentry_del", "/stats/flowentry/delete", methods=["DELETE"])
        def delete_flow_entry(self, req, **_):
            try:
                body = json.loads(req.body.decode("utf-8") if isinstance(req.body, bytes) else req.body)
            except (json.JSONDecodeError, Exception):
                return Response(status=400, content_type="application/json",
                                body=json.dumps({"error": "invalid JSON body"}))
            dpid = body.get("dpid")
            match = body.get("match", {})
            priority = body.get("priority", CFG.isolation.drop_rule_priority)
            if dpid is None:
                return Response(status=400, content_type="application/json",
                                body=json.dumps({"error": "missing dpid"}))

            dp = self.app.datapaths.get(int(dpid))
            if dp is None:
                return Response(status=404, content_type="application/json",
                                body=json.dumps({"error": f"switch {dpid} not found"}))

            parser = dp.ofproto_parser
            of_match = _dict_to_match(parser, match)
            mod = parser.OFPFlowMod(
                datapath=dp, command=dp.ofproto.OFPFC_DELETE,
                out_port=dp.ofproto.OFPP_ANY, out_group=dp.ofproto.OFPG_ANY,
                priority=int(priority), match=of_match)
            dp.send_msg(mod)
            return Response(content_type="application/json",
                            body=json.dumps({"status": "ok", "dpid": dpid}))

        @route("arp_table", "/sdnids/arp_table", methods=["GET"])
        def get_arp_table(self, req, **_):
            body = json.dumps(self.app.snapshot_arp_table())
            return Response(content_type="application/json", body=body)

        @route("mac_table", "/sdnids/mac_table", methods=["GET"])
        def get_mac_table(self, req, **_):
            body = json.dumps(self.app.snapshot_mac_table())
            return Response(content_type="application/json", body=body)
