"""Random instance generators shared by the Phase 1 and Phase 2 experiments.

Kept in one place so the 'random benchmark' the paper reports on and the
hard-instance benchmark of Gate 3c are never silently different populations.
"""
import numpy as np

START = np.array([-2.0, 0.0])
GOAL = np.array([2.0, 0.0])


def _clear_of_endpoints(c, r, start, goal, margin=0.05):
    return (np.linalg.norm(start - c) > r + margin
            and np.linalg.norm(goal - c) > r + margin)


def random_instance(rng, n_obs=None, d=3, k=1, l=0,
                    start=START, goal=GOAL, rmin=0.35, rmax=0.9):
    """Uniform obstacles in the box around the chord, clear of the endpoints."""
    if n_obs is None:
        n_obs = int(rng.integers(1, 4))
    obs = []
    guard = 0
    while len(obs) < n_obs and guard < 500:
        guard += 1
        c = np.array([rng.uniform(-1.6, 1.6), rng.uniform(-1.3, 1.3)])
        r = rng.uniform(rmin, rmax)
        if not _clear_of_endpoints(c, r, start, goal):
            continue
        if any(np.linalg.norm(c - c2) < r + r2 for c2, r2 in obs):
            continue                    # keep obstacles disjoint
        obs.append((c, float(r)))
    return dict(n=2, d=d, k=k, l=l, obstacles=obs,
                bc0=[start.tolist()], bc1=[goal.tolist()])


# ----------------------------------------------------------------------
# Hard-instance benchmark (Gate 3c).
#
# The authors' random instances are tight at the root 88-100% of the time
# (we measured 88.5%), so on that benchmark the B&B does nothing and the
# averages read as added complexity for zero gain.  These are built to force
# a commitment instead: narrow corridors, symmetric blockers, cul-de-sacs.
# They are reported SEPARATELY, never mixed into the random averages.
# ----------------------------------------------------------------------
def hard_instances(d=3, k=1, l=0):
    """Named hard instances.  Returns {name: instance-kwargs}."""
    S, G = START, GOAL
    out = {}

    # a blocker sitting exactly on the chord: left and right cost the same
    out["symmetric_blocker"] = [(np.array([0.0, 0.0]), 0.8)]

    # two blockers, staggered: the cheap route is not obvious
    out["staggered_pair"] = [(np.array([-0.7, 0.15]), 0.65),
                             (np.array([0.7, -0.15]), 0.65)]

    # narrow corridor: a gap barely wider than nothing
    for gap in (0.30, 0.15):
        r = 0.9
        out[f"corridor_{gap:.2f}"] = [(np.array([0.0, r + gap / 2]), r),
                                      (np.array([0.0, -r - gap / 2]), r)]

    # off-centre corridor: forces an asymmetric commitment
    out["corridor_offset"] = [(np.array([0.0, 1.15]), 0.9),
                              (np.array([0.0, -0.75]), 0.55)]

    # cul-de-sac: a pocket that looks cheap but dead-ends
    out["cul_de_sac"] = [(np.array([0.15, 0.95]), 0.85),
                         (np.array([-0.95, 0.35]), 0.6),
                         (np.array([1.05, 0.30]), 0.6)]

    # three-in-a-row: multiple sequential left/right decisions
    out["three_in_a_row"] = [(np.array([-1.0, 0.0]), 0.45),
                             (np.array([0.0, 0.0]), 0.45),
                             (np.array([1.0, 0.0]), 0.45)]

    # dense field
    out["dense_field"] = [(np.array([-1.1, 0.5]), 0.42),
                          (np.array([-0.4, -0.5]), 0.42),
                          (np.array([0.4, 0.5]), 0.42),
                          (np.array([1.1, -0.5]), 0.42)]

    return {nm: dict(n=2, d=d, k=k, l=l, obstacles=obs,
                     bc0=[S.tolist()], bc1=[G.tolist()])
            for nm, obs in out.items()}


def density_series(rng, counts=(1, 2, 3, 4, 5, 6), per_count=8, d=3, k=1, l=0):
    """Random instances grouped by obstacle count, for the node-count-versus-
    density plot Gate 3c asks for."""
    out = []
    for m in counts:
        made = 0
        guard = 0
        while made < per_count and guard < 4000:
            guard += 1
            inst = random_instance(rng, n_obs=m, d=d, k=k, l=l,
                                   rmin=0.3, rmax=0.6)
            if len(inst["obstacles"]) != m:
                continue
            out.append((m, inst))
            made += 1
    return out


# ----------------------------------------------------------------------
# Confined R^3 (Phase 6).
#
# Random instances in R^3 are 100% tight, so the branch-and-bound has nothing
# to do on them: with a third dimension available the curve escapes over the
# obstacles and the relaxation is exact.  Symmetric blockers give rho = 1 but
# with NO bound gap, so they do not help either.  The only construction we
# found that is both loose and has a real gap is a planar instance that is
# already loose, lifted into R^3 with the z escape blocked -- i.e. a confined
# environment.  That is the physical condition under which the method has
# work to do, and it is why these are built rather than sampled.
# ----------------------------------------------------------------------
def lift_to_confined_r3(obstacles_2d, zr=0.9, zoff=1.25,
                        xs=(-1.2, -0.4, 0.4, 1.2)):
    """Put a planar obstacle set in the z = 0 plane and wall off +-z.

    Accepts either the in-memory form [(centre, radius), ...] or the flat
    [x, y, r] triples the Gate 1d artifact stores, because the family is
    rebuilt from that JSON and the two forms are easy to confuse.
    """
    obs = []
    for item in obstacles_2d:
        if len(item) == 3 and np.isscalar(item[0]):
            x, y, r = float(item[0]), float(item[1]), float(item[2])
        else:
            c, r = item
            x, y, r = float(c[0]), float(c[1]), float(r)
        obs.append((np.array([x, y, 0.0]), r))
    for x in xs:
        for sgn in (1, -1):
            obs.append((np.array([x, 0.0, sgn * zoff]), float(zr)))
    return obs


def confined_r3_family(gate1_records, count=4, skip=1):
    """The four instances quoted in Phase 6, from the Gate 1d records.

    Selection is the loose instances (rho >= 1, ground truth available) in
    descending planar relative gap.  `skip=1` drops the widest-gap one, which
    is where the construction was developed; the four that follow are the
    reported family.  Returns [(instance_index, Segment kwargs)].
    """
    R = [r for r in gate1_records
         if not r["skipped"] and r["gt_ok"] and r["rho"] >= 1]
    R.sort(key=lambda r: -r["rel_gap"])
    out = []
    for rec in R[skip:skip + count]:
        out.append((int(rec["i"]),
                    dict(n=3, d=3, k=1, l=0,
                         obstacles=lift_to_confined_r3(rec["obstacles"]),
                         bc0=[[-2.0, 0.0, 0.0]], bc1=[[2.0, 0.0, 0.0]])))
    return out


def instance_id(inst):
    parts = [f"d{inst['d']}k{inst['k']}l{inst['l']}"]
    for c, r in inst["obstacles"]:
        parts.append(f"({c[0]:+.4f},{c[1]:+.4f};{r:.4f})")
    return "|".join(parts)
