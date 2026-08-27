import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# `a30`'s acyclicity test is pinned from the test suite, so the experiments
# directory has to be importable the same way `src` is.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))
