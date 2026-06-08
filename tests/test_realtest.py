"""Test evaluation/realtest.py helpers."""

import pytest

from evaluation.realtest import parse_dpids


def test_parse_dpids_decimal_and_hex():
    assert parse_dpids("1, 2, 0x3") == [1, 2, 3]


def test_parse_dpids_requires_value():
    with pytest.raises(ValueError):
        parse_dpids(" , ")
