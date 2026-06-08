"""Test isolation/flow_rules.py."""

from config import CFG
from isolation.flow_rules import (
    make_drop_rule,
    make_match_dict,
    make_rate_limit_rule,
    make_vlan_rule,
)
from isolation.isolator import Isolator


def test_drop_rule_structure():
    rule = make_drop_rule(dpid=1, port=3)
    assert rule["dpid"] == 1
    assert rule["match"] == {"in_port": 3}
    assert rule["actions"] == []
    assert rule["priority"] == CFG.isolation.drop_rule_priority


def test_vlan_rule_has_push_vlan():
    rule = make_vlan_rule(dpid=2, port=4)
    types = [a["type"] for a in rule["actions"]]
    assert "PUSH_VLAN" in types
    assert "SET_FIELD" in types


def test_rate_limit_rule_has_meter():
    rule = make_rate_limit_rule(dpid=3, port=5)
    types = [a["type"] for a in rule["actions"]]
    assert "METER" in types


def test_match_dict_minimal():
    body = make_match_dict(dpid=1, port=2)
    assert body == {"dpid": 1, "match": {"in_port": 2}}


def test_isolator_block_posts_drop_rule(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("isolation.isolator.requests.post", fake_post)
    isolator = Isolator(base_url="http://ryu")
    isolator.set_target(1, 3)

    assert isolator.apply(2) is True
    assert calls[0][0] == "http://ryu/stats/flowentry/add"
    assert calls[0][1]["match"] == {"in_port": 3}


def test_isolator_block_without_target_records_no_target():
    isolator = Isolator(base_url="http://ryu")

    assert isolator.apply(2) is False
    assert isolator.history[-1].success is False
    assert isolator.history[-1].metadata["note"] == "no_target"
