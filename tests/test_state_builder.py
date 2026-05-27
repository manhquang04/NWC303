"""Test detection/state_builder.py."""

import numpy as np

from config import CFG
from detection.state_builder import FEATURE_ORDER, StateBuilder


def test_state_vector_shape():
    sb = StateBuilder()
    feats = {k: 0.0 for k in FEATURE_ORDER}
    vec = sb.build(feats)
    assert vec.shape == (CFG.detection.state_dim,)
    assert vec.dtype == np.float32


def test_state_vector_clipping():
    sb = StateBuilder()
    feats = {k: 1e9 for k in FEATURE_ORDER}
    vec = sb.build(feats)
    assert np.all(vec <= 1.0)
    assert np.all(vec >= 0.0)


def test_missing_feature_defaults_to_zero():
    sb = StateBuilder()
    vec = sb.build({})        # nothing supplied
    assert np.allclose(vec, 0.0)


def test_feature_order_matches_state_dim():
    assert len(FEATURE_ORDER) == CFG.detection.state_dim
