import collections
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, TextIO, Optional

import numpy as np
import scipy.optimize
import torch
from sacred.observers import FileStorageObserver

from investigation.util import ExperimentConfig, ExperimentResult
from src.experiment import run_experiment
from src.rollout import compute_rollout

torch.set_default_dtype(torch.double)

SolutionCandidate = collections.namedtuple('SolutionCandidate', ['latent_dim', 'hidden_layer_size'])

ELITE_QUANTILE = 0.25
BARRIER_FITNESS = 1_000_000


def _parse_arguments() -> Tuple[Namespace, str]:
    parser = ArgumentParser()
    parser.add_argument('data_file_name')
    parser.add_argument('method', default='nelder-mead', choices=['nelder-mead', 'evolutionary'])
    parser.add_argument('-e', '--experiment', required=False)
    parser.add_argument('-o', '--output_file', default='tmp_hyperparameter_search.csv')
    parser.add_argument('-r', '--results_dir', default='tmp_results_hyperparameter_search')
    parser.add_argument('-f', '--seed_from', default=1, type=int)
    parser.add_argument('-t', '--seed_to', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--latent_dim_loc', default=7, type=int)
    parser.add_argument('--latent_dim_scale', default=3, type=int)
    parser.add_argument('--hidden_layer_size_loc', default=50, type=int)
    parser.add_argument('--hidden_layer_size_scale', default=20, type=int)
    parser.add_argument('--latent_dim_min', default=1, type=int)
    parser.add_argument('--latent_dim_max', default=100, type=int)
    parser.add_argument('--hidden_layer_size_min', default=1, type=int)
    parser.add_argument('--hidden_layer_size_max', default=200, type=int)
    # Parameters for the evolutionary search only.
    parser.add_argument('--population_size', default=100, type=int)
    args = parser.parse_args()
    data_file_name = args.data_file_name
    method = args.method
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


def _create_fitness_storage(args) -> TextIO:
    fh = open(args.output_file, 'a+')
    fh.write('run_id,latent_dim,hidden_layer_size,seed,fitness\n')
    fh.flush()
    return fh


def _save_candidate_fitness(fh: TextIO, candidate: SolutionCandidate, candidate_fitness: List[Tuple[str, int, float]]) -> None:
    for run_dir, seed, fitness in candidate_fitness:
        fh.write(f'{run_dir},{candidate.latent_dim},{candidate.hidden_layer_size},{seed},{fitness}\n')
    fh.flush()


def _evaluate_candidate(args: Namespace, experiment: str, seed_range: List[int], candidate: SolutionCandidate) \
        -> Optional[Tuple[float, List[Tuple[str, int, float]]]]:
    if not (args.latent_dim_min <= candidate.latent_dim <= args.latent_dim_max):
        return None
    if not (args.hidden_layer_size_min <= candidate.hidden_layer_size <= args.hidden_layer_size_max):
        return None

    def _evaluate_parameters_single_seed(seed: int) -> Optional[Tuple[str, float]]:
        config_updates = {
            'seed': seed,
            'latent_dim': candidate.latent_dim,
            'observation_model': [f'Linear(in_features, {candidate.hidden_layer_size})', 'Tanh()', f'Linear({candidate.hidden_layer_size}, out_features)']
        }
        try:
            run = run_experiment(args.data_file_name, ['with', experiment], results_dir=args.results_dir, config_updates=config_updates, debug=True)
        except Exception as e:
            print(f'HyperSearch: A run failed with an exception: {e}', file=sys.stderr)
            return None
        config = ExperimentConfig.from_dict(run.config)
        result = ExperimentResult.from_dict(config, run.config, run.experiment_info, run.result)

        _, (obs_rollouts, _), _ = compute_rollout(config, result, config.N)
        fitness = 0.0
        for n, obs_rollout in enumerate(obs_rollouts):
            fitness += np.sqrt(((obs_rollout - result.observations) ** 2).mean())
        return list(filter(lambda obj: isinstance(obj, FileStorageObserver), run.observers))[0].dir, fitness / len(obs_rollouts)

    run_dirs, seed_fitness = [], []
    for seed in seed_range:
        evaluation = _evaluate_parameters_single_seed(seed)
        if evaluation:
            run_dirs.append(evaluation[0])
            seed_fitness.append(evaluation[1])
    if not run_dirs:
        # No seed worked, be cannot evaluate these parameters.
        return None
    return np.mean(seed_fitness).item(), list(zip(run_dirs, seed_range, seed_fitness))


def _sample_truncated_integer_gaussian(rng: np.random.Generator, loc: int, scale: int, min_val: int, max_val: int) -> int:
    sample = None
    while sample is None or not (min_val <= sample <= max_val):
        sample = int(rng.normal(loc=loc, scale=scale))
    return sample


def _sample_candidates(args: Namespace, rng: np.random.Generator, N: int) -> List[SolutionCandidate]:
    population: List[SolutionCandidate] = []
    while len(population) < N:
        latent_dim = _sample_truncated_integer_gaussian(rng, args.latent_dim_loc, args.latent_dim_scale, args.latent_dim_min, args.latent_dim_max)
        hidden_layer_size = _sample_truncated_integer_gaussian(rng, args.hidden_layer_size_loc, args.hidden_layer_size_scale, args.hidden_layer_size_min,
                                                               args.hidden_layer_size_max)
        candidate = SolutionCandidate(latent_dim, hidden_layer_size)
        if candidate not in population:
            population.append(candidate)
    return population


def _evaluate_population(args: Namespace, experiment: str, evaluation_seed_range: List[int], fh: TextIO, population: List[SolutionCandidate]) \
        -> List[Tuple[SolutionCandidate, float]]:
    result = []
    for candidate in population:
        fitness_mean, candidate_fitness = _evaluate_candidate(args, experiment, evaluation_seed_range, candidate)
        result.append((candidate, fitness_mean))
        _save_candidate_fitness(fh, candidate, candidate_fitness)
    return result


def _select_fittest(rating: List[Tuple[SolutionCandidate, float]]) -> List[SolutionCandidate]:
    index = int(ELITE_QUANTILE * len(rating))
    return rating[:index]


def _recombine(args: Namespace, fittest: List[Tuple[SolutionCandidate, float]]):
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


def _search_evolutionary(args: Namespace, experiment: str, evaluation_seed_range: List[int], rng: np.random.Generator, fh: TextIO) -> Tuple[SolutionCandidate, float]:
    population = _sample_candidates(args, rng, args.population_size)

    while True:
        # Selection.
        rating = sorted(_evaluate_population(args, experiment, evaluation_seed_range, fh, population), key=lambda x: x[1], reverse=True)
        fittest = _select_fittest(rating)

        # Crossover.
        recombined = _recombine(args, fittest)

        # TODO: Mutation.
        ...


def _search_nelder_mead(args: Namespace, experiment: str, evaluation_seed_range: List[int], rng: np.random.Generator, fh: TextIO) -> Tuple[SolutionCandidate, float]:
    def build_candidate_from_array(candidate_arr):
        return SolutionCandidate(*[round(x) for x in candidate_arr])

    def objective(candidate_arr):
        candidate = build_candidate_from_array(candidate_arr)
        evaluation = _evaluate_candidate(args, experiment, evaluation_seed_range, candidate)
        if evaluation is None:
            _save_candidate_fitness(fh, candidate, [('', -1, -1.0)])
            return BARRIER_FITNESS
        fitness_mean, candidate_fitness = evaluation
        _save_candidate_fitness(fh, candidate, candidate_fitness)
        return fitness_mean

    dim = 2
    initial_simplex = np.asarray([[candidate.latent_dim, candidate.hidden_layer_size] for candidate in _sample_candidates(args, rng, dim + 1)])
    result = scipy.optimize.minimize(objective, np.zeros(initial_simplex.shape[1]), method='Nelder-Mead', options={'initial_simplex': initial_simplex})
    return build_candidate_from_array(result.x), result.fun


def main():
    args, experiment = _parse_arguments()
    evaluation_seed_range = list(range(args.seed_from, args.seed_to + 1))
    rng = np.random.Generator(np.random.PCG64(args.seed))

    with _create_fitness_storage(args) as fh:
        if args.method == 'evoluationary':
            best_candidate, fitness = _search_evolutionary(args, experiment, evaluation_seed_range, rng, fh)
        elif args.method == 'nelder-mead':
            best_candidate, fitness = _search_nelder_mead(args, experiment, evaluation_seed_range, rng, fh)
        else:
            assert False

    print('HyperSearch: Finished!')
    print('Best solution candidate is %s with a fitness of %f.' % (str(best_candidate), fitness))


if __name__ == '__main__':
    main()
