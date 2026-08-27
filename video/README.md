# Supplementary videos

**`submission.mp4` -- 83 s, 1280x720, no audio. START HERE.** The five clips
below, cut into one argument in the paper's own order, with dark chapter cards
carrying the transitions: the relaxation that is globally optimal and returns
no maneuver; the certificate, on a matrix the size of the contact set; that it
does not grow when the problem does; the corner law that settles it before the
instance exists; the fold that recovers at rho = 1; and the closed loop, where
the naive projection's own curve breaks and the certified one's never does.
Regenerate with `python paper/make_video_submission.py` once the five source
clips are in `video/`. The clips below are the same material, uncut, for a
reader who wants one of them on its own.

`corner_law.mp4` -- 11 s, 1280x660, no audio. The corner law
(Proposition `prop:corner`) as a surface. The normalised kernel over the whole
square [0,1]^2, for the SAME degree, the SAME cost and the SAME boundary data,
differing only in whether the curve is one polynomial or two pieces joined at a
knot. Left: the single polynomial dips BELOW zero at the two opposite corners,
reaching -0.2261. Right: one interior knot lifts it, minimum +0.0654, no
negative region anywhere. The corner constants are -2 and +1, and +1 is
k/(k!)^2 -- the corner constant of the CONTINUUM Green function itself. One
knot does not improve the approximation there; it restores it exactly, at every
degree, which no polynomial degree ever does. Regenerate with
`python paper/make_video_corner.py`.

`size_contrast.mp4` -- 19 s, 1280x640, no audio. The certificate does not grow
with the problem (Theorem `thm:contacts`), read from `a50`'s artifact. The
degree runs 5 -> 13 so the lifted free block runs 2x2 -> 10x10, then the
ambient dimension runs 2 -> 5; the matrix the certificate actually
eigendecomposes stays 2x2 in every cell, at simplicity margin never below
0.8631. This is the claim the recovery figure's own instance cannot show:
there f = 2 and the contact count is also 2, so the two objects are the same
size. Regenerate with `python paper/make_video_size.py`.

`certificate.mp4` -- 18 s, 1280x640, no audio. What the certificate IS, in four
acts, on the two-sphere instance of the recovery figure. Act 1: the relaxation
is globally optimal and is not a maneuver -- the projection cuts both spheres
because of the borrowed coordinate `z(s)`. Act 2: the contacts the dual charges
index a 2x2 Green matrix whose weighted spectrum is capped at 1; the top
eigenvalue sits AT the cap (so `dim ker Z = 1`) and the second at 0.17, a
simplicity margin `g = 0.834 > 0` certifying `rho <= 1` -- on a matrix the size
of the contact set, not of the SDP. Act 3: `rho = 1` reduces recovery to a
search over `w` in R^2, over the complement of a union of open balls. Act 4:
the fold, with its clearance decided by an exact interval-polynomial test.
Regenerate with `python paper/make_video_cert.py`.

`loop_recovery.mp4` -- 6 s, 1280x580, no audio. Whose fault? One run of the
disturbed receding-horizon experiment (Section 7.4), side by side from the SAME
obstacle field and the SAME kicks. Step 1: both policies meet `rho = 1`; naive
executes the projection (safe this time), certified folds. Step 2: naive meets
`rho = 1` again, executes the projection again, and its OWN CURVE cuts an
obstacle at exact clearance -0.057 (the violating arc is red); the certified
panel is clean and loses the run only to a kick that lands the state inside an
obstacle. Over the 30 paired runs the executed-curve violation count is 4 for
the naive policy and ZERO for the certified one. Regenerate with
`python paper/make_video_loop.py` after
`python experiments/a48_loop_recovery.py --showcase 28`.

`lift_rotation.mp4` -- 20 s, 1920x1080, no audio, no narration.

Five acts, each answering something a still figure cannot. The instance is the
one the lift and recovery figures both draw: two spheres of radius 0.45, start
and goal pinned, `rho = 1`.

**I. The plane.** Seen from directly above, only the projection exists: the
curve a planner recovers if it discards the extra coordinate. It runs through
both obstacles, and the part inside them is drawn in red.

**II. The lift.** The camera tilts down while the borrowed coordinate grows
from zero to its solved value, so the same solution is seen rising out of the
plane it appeared trapped in. The growth is a reveal and the frame says so; by
the end of the act the geometry is the solved one, and it never changes again.

**III. Every angle.** A full revolution. A single 2-D image of a 3-D scene can
hide an intersection at whatever angle it was drawn from, so "the curve clears
the spheres" is exactly the claim one viewpoint cannot settle. Through the
whole turn the trajectory stays outside both spheres and its shadow stays
inside them.

**IV. The fold.** The bound is not yet a trajectory. The borrowed coordinate is
spent along a spatial direction `w`, `gamma_w = gamma* + z w`, and the curve
comes back into the plane displaced sideways instead of deleted. The obstacles
do not move and the pinned endpoints do not move; what was a shadow through both
spheres becomes a planar curve outside them. At `fold = 0` the frame shows the
solved lift and at `fold = 1` the recovered curve exactly -- the interpolation
between those two is the only thing the camera invents.

**V. The count.** A held view carrying the numbers.

Apart from the camera and the two reveals, nothing moves: the relaxation is
solved once, before the first frame, so no frame can disagree with another.
Aspect ratios are the true ones -- a sphere is drawn as a sphere, because the
claim is about clearing an object of a given radius. Nothing is stretched to
make the arc look bigger than it is.

Every clearance on screen is the same functional, `min_s min_j q_j(s)` in the
units of the obstacle constraint, so the three numbers are comparable with each
other and with the paper. The one distance in the video, in Act I, says so.

Regenerate with:

    python paper/make_figure6.py     # writes artifacts/fig_recovery.json
    python paper/make_video.py --out video

The fold direction `w*` and the recovered and projected clearances are READ from
`artifacts/fig_recovery.json`; the lift clearance comes from
`artifacts/fig_liftpath.json`. The video does not re-solve for `w*`, so it cannot
print a number the paper does not.
