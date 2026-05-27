"""Test isolation/flow_rules.py."""

from config import CFG
from isolation.flow_rules import (
    make_drop_rule,
    make_match_dict,
    make_rate_limit_rule,
    make_vlan_rule,
)


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
