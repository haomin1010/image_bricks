#!/usr/bin/env bash

if [[ -n "${__IMAGE_BRICKS_ISAAC_REWARD_TERM_LOADED:-}" ]]; then
  return 0
fi
__IMAGE_BRICKS_ISAAC_REWARD_TERM_LOADED=1

image_bricks__export_default() {
  local var_name="$1"
  local default_value="$2"
  local current="${!var_name:-}"
  if [[ -z "$current" ]]; then
    export "${var_name}=${default_value}"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

image_bricks__export_default IMAGE_BRICKS_ROOT "$REPO_ROOT"

image_bricks__export_default VAGEN_ISAAC_FORMAT_REWARD "0.1"
image_bricks__export_default VAGEN_ISAAC_CORRECT_PLACEMENT_REWARD "1.0"
image_bricks__export_default VAGEN_ISAAC_FLOATING_PLACEMENT_PENALTY "-10.0"
image_bricks__export_default VAGEN_ISAAC_NON_CANDIDATE_PENALTY "-5.0"
image_bricks__export_default VAGEN_ISAAC_MAX_ATTEMPTS_FACTOR "1.5"
image_bricks__export_default VAGEN_ISAAC_GROUND_TRUTH_ROOT "${IMAGE_BRICKS_ROOT}/IsaacLab/scripts/data_gen/convex_json_batch"

# IsaacLab can later pass any of the keys:
#   collapsed / collapse / has_collapsed / tower_collapsed
# through the step info payload. This mock knob keeps local testing simple:
# -1 disables the mock; any positive N triggers a collapse signal at attempt N.
image_bricks__export_default VAGEN_ISAAC_COLLAPSE_MOCK_AFTER_ATTEMPT "-1"
