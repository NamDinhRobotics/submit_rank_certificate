"""Pin the repaired rank arbiter of Phase A5.

A3 decided `rank2_attained = (face_ratio > 1e-6)` from the max-rank face probe
alone.  A5 found that criterion is defective: at `d = 9` the probe buys apparent
rank by violating the cost-optimality constraint it is meant to respect, and A5's
first sweep therefore produced 8 spurious `rho = 2` cells and failed three gates.

These tests are fast because they exercise the DECISION, not the solvers: the
three readings are injected.  The last two use the values actually measured and
committed in `artifacts/a5_rho2_scope.json`, so a regression in the criterion
shows up against real data rather than against a hand-made example.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import a5_rho2_scope as A5                                   # noqa: E402


def _inject(monkeypatch, clarabel, scs, face, residual, cost=1.0):
    def fake(seg, slack=1e-9):
        return dict(clarabel=dict(ratio=clarabel, status="optimal", cost=cost),
                    scs=dict(ratio=scs, status="optimal", cost=cost),
                    max_rank_face=dict(ratio=face, status="optimal",
                                       residual=residual))
    monkeypatch.setattr(A5, "arbitrate_rank", fake)


def test_face_sees_nothing_gives_rank_one(monkeypatch):
    _inject(monkeypatch, 1e-12, 1e-13, 1e-9, 1e-10)
    out = A5.arbitrated_rank(None)
    assert out["verdict"] == 1


def test_clean_second_direction_gives_rank_two(monkeypatch):
    _inject(monkeypatch, 2.871e-01, 2.871e-01, 2.738e-01, 1.632e-03)
    out = A5.arbitrated_rank(None)
    assert out["verdict"] == 2
    assert out["signal_over_resid"] > A5.SIGNAL_OVER_RESIDUAL


def test_probe_off_the_face_is_undetermined_not_rank_two(monkeypatch):
    """THE defect: A3's raw criterion would have said 2 here."""
    _inject(monkeypatch, 1.101e-07, 6.099e-14, 1.225e-04, 1.582e-05)
    out = A5.arbitrated_rank(None)
    assert out["face"] > 1e-6                    # A3's test passes ...
    assert out["verdict"] is None                # ... and ours refuses
    assert out["reason"] == "probe off the face"
    assert out["signal_over_resid"] == pytest.approx(7.74, rel=0.02)


def test_no_ipm_corroboration_is_undetermined(monkeypatch):
    """On the face, but invisible to both interior-point solves."""
    _inject(monkeypatch, 1e-9, 1e-11, 1e-2, 1e-9)
    out = A5.arbitrated_rank(None)
    assert out["verdict"] is None
    assert out["reason"] == "no IPM corroboration"


def test_one_ipm_alone_is_not_enough(monkeypatch):
    _inject(monkeypatch, 3.0e-01, 1e-12, 2.9e-01, 1e-6)
    assert A5.arbitrated_rank(None)["verdict"] is None


def test_solver_failure_is_undetermined_not_a_crash(monkeypatch):
    def boom(seg, slack=1e-9):
        raise RuntimeError("Solver 'CLARABEL' failed")
    monkeypatch.setattr(A5, "arbitrate_rank", boom)
    out = A5.arbitrated_rank(None)
    assert out["verdict"] is None
    assert out["reason"].startswith("solver:")


def test_committed_defect_table_is_classified_as_reported(monkeypatch):
    """Replay the committed rows: the d=9 ones must NOT read as rank 2."""
    import json
    art = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                       "a5_rho2_scope.json")
    if not os.path.exists(art):
        pytest.skip("a5_rho2_scope.json not present")
    rows = json.load(open(art))["gates"]["a5b_screen_was_unreliable"]["defect_table"]
    seen = 0
    for r in rows:
        if "face" not in r:                       # arbitration failed outright
            continue
        seen += 1
        _inject(monkeypatch, r["clarabel"], r["scs"], r["face"], r["face_resid"])
        v = A5.arbitrated_rank(None)["verdict"]
        if r["d"] == 9:
            assert v is None, (r, v)
        else:
            assert v == 2, (r, v)
    assert seen >= 3


def test_corner_radius_keeps_the_endpoints_clear():
    """The construction fix: a fixed radius swallows the endpoint at small s1."""
    for s1 in (0.01, 0.02, 0.05, 0.1, 0.3):
        r = A5.corner_radius(s1)
        assert r < 4.0 * s1, s1                   # centre-to-start distance
        assert r <= 0.25
    assert A5.corner_radius(0.3) == pytest.approx(0.25)
