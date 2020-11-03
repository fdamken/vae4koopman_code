import collections
import os
import sys
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch

from investigation.util import ExperimentConfig, ExperimentResult
from src.experiment import run_experiment
from src.rollout import compute_rollout

torch.set_default_dtype(torch.double)

SolutionCandidate = collections.namedtuple('SolutionCandidate', ['latent_dim', 'hidden_layer_size'])

ELITE_QUANTILE = 0.25


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('data_file_name')
    parser.add_argument('-e', '--experiment', required=False)
    parser.add_argument('-o', '--output_file', default='tmp_hyperparameter_search.csv')
    parser.add_argument('-r', '--results_dir', default='tmp_results_hyperparameter_search')
    parser.add_argument('-f', '--seed_from', default=1, type=int)
    parser.add_argument('-t', '--seed_to', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--population_size', default=100, type=int)
    parser.add_argument('--latent_dim_loc', default=7, type=int)
    parser.add_argument('--latent_dim_scale', default=3, type=int)
    parser.add_argument('--latent_dim_min', default=1, type=int)
    parser.add_argument('--latent_dim_max', default=100, type=int)
    parser.add_argument('--hidden_layer_size_loc', default=50, type=int)
    parser.add_argument('--hidden_layer_size_scale', default=20, type=int)
    parser.add_argument('--hidden_layer_size_min', default=1, type=int)
    parser.add_argument('--hidden_layer_size_max', default=200, type=int)
    args = parser.parse_args()
    data_file_name = args.data_file_name
    experiment = args.experiment
    if not experiment:
        print(f'HyperSearch: Experiment name not explicitly set, using data file name <{data_file_name}>.')
        experiment = data_file_name
    output_file = args.output_file
    results_dir = args.results_dir

    if args.latent_dim_min > args.latent_dim_max:
        print('Latent dimension min is greater than hidden layer size max!', file=sys.stderr)
        quit(1)
    if args.hidden_layer_size_min > args.hidden_layer_size_max:
        print('Hidden layer size min is greater than hidden layer size max!', file=sys.stderr)
        quit(1)

    if os.path.exists(output_file):
        print(f'HyperSearch: Output file <{output_file}> exists. Aborting.', file=sys.stderr)
        quit(1)

    dry_run_successful = run_experiment(data_file_name, ['with', experiment], results_dir=results_dir, dry_run=True)
    if not dry_run_successful:
        print('HyperSearch: Dry run of experiment setup failed. Aborting.', file=sys.stderr)
        quit(1)
    print('HyperSearch: Dry run of experiment setup successful. Continuing to hyperparameter search.')

    return args, experiment


def _evaluate_parameters(data_file_name: str, experiment: str, results_dir: str, seed_range: List[int], /, hidden_layer_size: int, latent_dim: int) \
        -> Tuple[float, List[Tuple[int, float]]]:
    def _evaluate_parameters_single_seed(seed: int) -> float:
        config_updates = {
            'seed': seed,
            'latent_dim': latent_dim,
            'observation_model': [f'Linear(in_features, {hidden_layer_size})', 'Tanh()', f'Linear({hidden_layer_size}, out_features)'],
            'max_iterations': 1
        }
        run = run_experiment(data_file_name, ['with', experiment], results_dir=results_dir, config_updates=config_updates)
        config = ExperimentConfig.from_dict(run.config)
        result = ExperimentResult.from_dict(config, run.config, run.experiment_info, run.result)

        _, (obs_rollouts, _), _ = compute_rollout(config, result, config.N)
        fitness = 0.0
        for n, obs_rollout in enumerate(obs_rollouts):
            fitness += np.sqrt(((obs_rollout - result.observations) ** 2).mean())
        return fitness / len(obs_rollouts)

    fitnesses = list([_evaluate_parameters_single_seed(seed) for seed in seed_range])
    return np.mean(fitnesses).item(), list(zip(seed_range, fitnesses))


def _sample_truncated_integer_gaussian(rng: np.random.Generator, loc: int, scale: int, min_val: int, max_val: int) -> int:
    sample = None
    while sample is None or not (min_val <= sample <= max_val):
        sample = int(rng.normal(loc=loc, scale=scale))
    return sample


def _sample_initial_population(rng: np.random.Generator, args) -> List[SolutionCandidate]:
    population: List[SolutionCandidate] = []
    while len(population) < args.population_size:
        latent_dim = _sample_truncated_integer_gaussian(rng, args.latent_dim_loc, args.latent_dim_scale, args.latent_dim_min, args.latent_dim_max)
        hidden_layer_size = _sample_truncated_integer_gaussian(rng, args.hidden_layer_size_loc, args.hidden_layer_size_scale, args.hidden_layer_size_min,
                                                               args.hidden_layer_size_max)
        candidate = SolutionCandidate(latent_dim, hidden_layer_size)
        if candidate not in population:
            population.append(candidate)
    return population


def _evaluate_population(args, experiment: str, evaluation_seed_range: List[int], population: List[SolutionCandidate], fh) -> List[Tuple[SolutionCandidate, float]]:
    result = []
    for candidate in population:
        fitness_mean, seed_fitness = _evaluate_parameters(args.data_file_name, experiment, args.results_dir, evaluation_seed_range,
                                                          hidden_layer_size=candidate.hidden_layer_size,
                                                          latent_dim=candidate.latent_dim)
        result.append((candidate, fitness_mean))
        for seed, fitness in seed_fitness:
            fh.write(f'{candidate.hidden_layer_size},{candidate.latent_dim},{seed},{fitness}\n')
        fh.flush()
    return result


def _select_fittest(rating: List[Tuple[SolutionCandidate, float]]) -> List[SolutionCandidate]:
    index = int(ELITE_QUANTILE * len(rating))
    return rating[:index]


def _recombine(args, fittest):
    n = int(ELITE_QUANTILE * len(fittest))
    xi = n // 5
    a = int(n ** 2 + n * xi - (n ** 2 - n) / 2)

    population = list([x[0] for x in fittest])
    while len(population) < 2 * args.population_size:
        parent_indices = []
        for _ in (1, 2):
            r = np.random.random()
            sample_found = False
            for k in range(n):
                r -= (n + xi - k) / a
                if r <= 0:
                    sample_found = True
                    parent_indices.append(k)
                    break
            if not sample_found:
                print(f'HyperSearch: No recombining sample found! Falling back to the first element. {r=}')
                parent_indices.append(0)
        mother_index, father_index = parent_indices
        mother = fittest[mother_index][0]
        father = fittest[father_index][0]
        population.append(SolutionCandidate(mother.latent_dim, father.hidden_layer_size))
    return population


def main():
    args, experiment = _parse_arguments()

    evaluation_seed_range = list(range(args.seed_from, args.seed_to + 1))

    rng = np.random.Generator(np.random.PCG64(args.seed))

    recombined = _sample_initial_population(rng, args)

    with open(args.output_file, 'a+') as fh:
        fh.write('hidden_layer_size,latent_dim,seed,fitness\n')
        fh.flush()

        # Selection.
        rating = sorted(_evaluate_population(args, experiment, evaluation_seed_range, recombined, fh), key=lambda x: x[1], reverse=True)
        fittest = _select_fittest(rating)

        # Crossover.
        recombined = _recombine(args, fittest)

        # TODO: Mutation.
        ...


if __name__ == '__main__':
    main()
