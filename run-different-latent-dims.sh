#!/bin/bash

# max_iterations
# g_optimization_max_iterations
# latent_dim

rm -rf log
mkdir log

#source venv/bin/activate

run_ex() {
    printf "%s; max_iterations, g_optimization_max_iterations, latent_dim: %4d, %4d, %3d\n" $(date +%Y%m%dT%H:%M:%S) $1 $2 $3 >>log/run-starts
    NO_NEPTUNE= PYTHONPATH=. python src/experiment_pendulum_damped.py with "max_iterations=$1" "g_optimization_max_iterations=$2" "latent_dim=$3" &>>log/run-$1-$2-$3
}

run_ex 100 100 60
run_ex 100 100 120
run_ex 100 100 150
run_ex 100 100 200
run_ex 100 100 250
run_ex 100 100 300

for max_iterations in 100 10 1 1000; do
    for g_optimization_max_iterations in 100 10 1 1000; do
        #for latent_dim in 2 3 4 5 6 7 8 9 10 15 30 60 120 150; do
        if [[ $max_iterations -eq 100 ]] && [[ $g_optimization_max_iterations -eq 100 ]]; then
            continue
        fi
        for latent_dim in 2 3 4 5 6 7 8 9 10; do
            run_ex $max_iterations $g_optimization_max_iterations $latent_dim
        done
    done
done
