#!/bin/bash

experiment="$1"
if [[ "$experiment" == "" ]]; then
    echo "E: Usage: $0 <experiment-name>" >&2
    exit 126
fi

set -o errexit
set -o nounset

results_dir="tmp_results_grid_search/latent-dim_$(date +%Y%m%dT%H:%M:%S)"
log_dir="$results_dir/log"
mkdir -p "$results_dir" "$log_dir"

run_ex() {
    latent_dim="$1"
    PYTHONPATH=. RESULTS_DIR="$results_dir" python src/experiment.py "$experiment" "latent_dim=$latent_dim" | tee -a "$log_dir/run-$latent_dim"
}

echo "Running multiple latent dims for experiment $experiment."

for latent_dim in $(seq 1 100); do
    set +o errexit
    run_ex "$latent_dim"
    set -o errexit
done
