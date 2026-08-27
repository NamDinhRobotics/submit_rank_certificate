"""The figures must fit the column of the class the paper is actually submitted
under.

This exists because a whole revision round measured the figures against a
class with a *wider* column than the one the paper was set in, and both
overflowed by exactly the difference.  A silent overfull box in a figure is the
kind of thing a reviewer sees and an author does not.

The limits below are the NARROWEST column the manuscript has ever been set in
(245.72pt, against 252pt for the journal class it now uses), so the test stays
a conservative guard: a figure that passes here cannot overflow the wider one.

The widths are read from the PDFs themselves rather than from the `figsize`
arguments, because `bbox_inches="tight"` means the two are not equal.
"""
import os
import re

import pytest

HERE = os.path.dirname(__file__)
PAPER = os.path.join(HERE, "..", "paper")

# from \the\columnwidth and \the\textwidth under the real ieeeconf.cls
COLUMN_PT = 245.71811
TEXT_PT = 505.89
PT_PER_BP = 72.27 / 72.0            # PDF points are big points


def _pdf_width_pt(path):
    """Width of a one-page PDF in TeX points, from its MediaBox."""
    with open(path, "rb") as fh:
        raw = fh.read()
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", raw)
    assert boxes, "no MediaBox in %s" % path
    nums = [float(x) for x in boxes[0].split()]
    return (nums[2] - nums[0]) * PT_PER_BP


@pytest.mark.parametrize("name,limit_pt,kind", [
    ("fig_lobe.pdf", COLUMN_PT, "single-column figure"),
    ("fig_counterexample.pdf", TEXT_PT, "full-width figure*"),
    ("fig_evidence.pdf", TEXT_PT, "full-width figure*"),
])
def test_figure_fits_the_ieeeconf_column(name, limit_pt, kind):
    path = os.path.join(PAPER, name)
    if not os.path.exists(path):
        pytest.skip("%s not built" % name)
    w = _pdf_width_pt(path)
    assert w <= limit_pt, (
        "%s (%s) is %.2fpt wide, over the %.2fpt limit by %.2fpt -- it will "
        "produce an overfull hbox under ieeeconf.cls"
        % (name, kind, w, limit_pt, w - limit_pt))


def test_the_limits_are_the_narrow_class_not_the_wide_one():
    """A guard on the guard: if someone 'fixes' this file by pasting modern
    IEEEtran's 252pt/516pt, the test would pass on figures that overflow the
    class actually used for submission."""
    assert COLUMN_PT < 252.0
    assert TEXT_PT < 516.0
