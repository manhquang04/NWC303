"""Test evaluation/metrics.py."""

from config import ACTION_ALLOW, ACTION_BLOCK
from evaluation.metrics import MetricsCalculator, StepRecord


def test_metrics_counts_and_timing():
    calc = MetricsCalculator()
    calc.mark_attack_start(10.0)
    calc.mark_detected(12.0)
    calc.mark_isolated(13.5)

    calc.add_record(StepRecord(
        timestamp=12.0,
        ground_truth="attack",
        action=ACTION_BLOCK,
        reward=9.9,
        detected=True,
        isolated=True,
    ))
    calc.add_record(StepRecord(
        timestamp=14.0,
        ground_truth="normal",
        action=ACTION_ALLOW,
        reward=0.9,
        detected=False,
        isolated=False,
    ))

    report = calc.compute(episodes=1)

    assert report.tp == 1
    assert report.tn == 1
    assert report.fp == 0
    assert report.fn == 0
    assert report.f1 == 1.0
    assert report.mttd_sec == 2.0
    assert report.mtti_sec == 3.5
