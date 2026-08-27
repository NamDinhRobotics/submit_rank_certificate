#!/usr/bin/env bash
# Regenerate every artifact the paper cites, in dependency order.
# gate1.json first: four phases read the instance population from it.
set -euo pipefail
cd "$(dirname "$0")"
python experiments/e1_reproduce.py
python experiments/a10_knot_dichotomy.py
python experiments/a11_table1.py
python experiments/a12_exact_bernstein.py
python experiments/a13_nodal_mechanism.py
python experiments/a14_exact_conditions.py
python experiments/a15_continuum_green.py
python experiments/a16_rho_vs_looseness.py
python experiments/a1_contact_rank.py
python experiments/a22_multirobot_rank.py
python experiments/a2_perron_route.py
python experiments/a34_certificate_cost.py
python experiments/a35_census_attrition.py
python experiments/a36_parameterization_choice.py
python experiments/a37_bound_and_refusals.py
python experiments/a38_chain_tightness.py
python experiments/a39_slater_margin.py
python experiments/a3_finite_d_certificate.py
python experiments/a40_second_lifting.py
python experiments/a41_exact_rho2.py
python experiments/a42_eta_boundary.py
python experiments/a43_h3_multirobot.py
python experiments/a44_quadrotor_minsnap.py
python experiments/a45_panel10.py
python experiments/a46_rank_one_recovery.py
python experiments/a47_corner_law.py
python experiments/a48_loop_recovery.py
python experiments/a49_estimated_obstacles.py
python experiments/a4_simplicity_margin.py
python experiments/a50_size_contrast.py
python experiments/a5_rho2_scope.py
python experiments/a6_two_mechanisms.py
python experiments/a7_moment_cone.py
python experiments/a8_nondegeneracy_multiseg.py
python experiments/a9_multiseg_rho2.py

# Everything above re-solves the science and re-reads every gate, and needs
# nothing from paper/.  What follows BUILDS the paper -- figures, videos, and
# the audit that ties each number in the text back to its artifact -- and ships
# only with the submission package, not with the public mirror.  So it runs
# when paper/ is present and is skipped, loudly, when it is not.
if [ -d paper ]; then
  # make_figure3.py draws fig_evidence, which the current manuscript does not
  # include; make_figure5.py draws fig_robots, which left it at the 12-page cut.
  # Both are shipped and neither is run here.
  python paper/make_figure6.py
  python paper/make_figure7.py
  python paper/make_figure.py
  python paper/make_figure2.py
  python paper/make_figure4.py
  python paper/audit_paper.py
else
  echo
  echo "paper/ is absent: figures, videos and the audit are not rebuilt."
  echo "They ship with the submission package; every experiment above ran."
fi
