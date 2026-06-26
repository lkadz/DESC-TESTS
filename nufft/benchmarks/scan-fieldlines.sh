#!/bin/bash
# Scan field-line-tracing parameters to find settings that resolve clean flux
# surfaces, for two configs, as separate <=2h Della jobs (full 40GB A100).
#
# What actually controls flux-surface quality in a Poincare plot:
#   * ntransit       -- punctures per surface. A field line on a good (integrable)
#                       surface fills that surface's cross-section over many
#                       transits; more transits => denser, clearer closed curve.
#                       This is the #1 knob. (100 = sparse, 500-1000 = filled.)
#   * offset count   -- how many NESTED surfaces you sample. Because one seed per
#                       offset already fills its surface, --seed-poloidal stays 1
#                       and we raise --seed-offset-count to show nesting cheaply.
#   * offset range   -- which radial band you probe. Too far out => stochastic /
#                       open lines => scatter, not surfaces.
#   * rtol/atol      -- integration accuracy; tighter => sharper curves, slower.
#   * max-steps      -- diffrax step cap PER LINE over all transits. Must be big
#                       enough that lines complete every transit (else they stop
#                       early and the surface is only partly drawn). Rule of thumb
#                       here: ~400 steps/transit, so max-steps ~= 400 * ntransit.
#
# Usage:
#   DRY_RUN=1 bash benchmarks/scan-fieldlines.sh     # print the commands only
#   bash benchmarks/scan-fieldlines.sh               # submit every job
#   SCRIPT=benchmarks/script-fieldlines-vc.py bash benchmarks/scan-fieldlines.sh
#                                                    # run the virtual-casing scan
set -euo pipefail

SCRIPT="${SCRIPT:-benchmarks/script-fieldlines-nufft.py}"
CLUSTER="${CLUSTER:-della40}"
# Per-job wall time is set in the PARAM_SETS rows below (the WALLTIME column =
# the 2bump estimate). The heavy sets (res_high, many_surf, tol_tight) get 3h.
# precise_QA carries ~30-50 min of extra setup + JIT before tracing, so its
# mid-weight jobs (nt>=500) are auto-bumped from 2h to 3h. Set TIME=HH:MM:SS to
# force a single wall time on every job instead.
TIME_OVERRIDE="${TIME:-}"

CONFIGS=(
  "2bump_n0_0.07_n1_0.02_k_iota_-1.0"
)

# One row per job (per config). Columns:
#   POL  OFFMIN  OFFMAX  OFFCOUNT  NTRANSIT  TOL  MAXSTEPS  WALLTIME  TAG
# The four standard-weight (2h) cases. The heavier 3h sets (res_high, many_surf,
# tol_tight) are intentionally left out of this 4-case scan.
PARAM_SETS=(
  "1  0.005 0.04  8   100  1e-5   40000  02:00:00  res_low"     # cheap baseline, sparse
  "1  0.005 0.04  8   300  1e-6  120000  02:00:00  res_med"     # moderate density
  "1  0.001 0.01  8   500  1e-6  200000  02:00:00  near_edge"   # very close to the LCFS
  "1  0.01  0.08  8   500  1e-6  200000  02:00:00  wide_span"   # probe further out
)

submit_one() {
  local config="$1" pol="$2" offmin="$3" offmax="$4" offcount="$5" \
        nt="$6" tol="$7" ms="$8" base_time="$9" tag="${10}"
  # precise_QA setup/JIT overhead: bump its mid/heavy 2h jobs to 3h.
  local time="$base_time"
  if [[ "$config" == "precise_QA" && "$base_time" == "02:00:00" \
        && "$tag" != "res_low" && "$tag" != "res_med" ]]; then
    time="03:00:00"
  fi
  # A global TIME=HH:MM:SS env var overrides everything.
  [[ -n "$TIME_OVERRIDE" ]] && time="$TIME_OVERRIDE"
  local variant="scan_${tag}_pol${pol}_off${offmin}-${offmax}x${offcount}_nt${nt}_tol${tol}_ms${ms}"
  local cmd=(
    python "$SCRIPT"
    --config "$config"
    --cluster "$CLUSTER"
    --time "$time"
    --seed-poloidal "$pol"
    --seed-offset-min "$offmin"
    --seed-offset-max "$offmax"
    --seed-offset-count "$offcount"
    --ntransit "$nt"
    --poincare-rtol "$tol"
    --poincare-atol "$tol"
    --poincare-max-steps "$ms"
    --variant "$variant"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    echo ">>> submit: $config / $variant"
    "${cmd[@]}"
    echo
  fi
}

for config in "${CONFIGS[@]}"; do
  for row in "${PARAM_SETS[@]}"; do
    # shellcheck disable=SC2086
    read -r POL OFFMIN OFFMAX OFFCOUNT NT TOL MS WALL TAG <<< "$row"
    submit_one "$config" "$POL" "$OFFMIN" "$OFFMAX" "$OFFCOUNT" "$NT" "$TOL" "$MS" "$WALL" "$TAG"
  done
done
