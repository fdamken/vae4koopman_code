import os
import shutil
import sys
from argparse import ArgumentParser
from typing import List

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch

from investigation.plot_util import SubplotsAndSave, figsize

jsonpickle_numpy.register_handlers()
torch.set_default_dtype(torch.double)


def plot_observations(out_dir: str, name: str, N: int, h: float, T: int, T_train: int, observation_dim: int, observation_dim_names: List[str], observations: np.ndarray) -> None:
    domain = np.arange(T) * h

    with SubplotsAndSave(out_dir, f'observations-{name}', observation_dim, N,
                         sharex='col',
                         sharey='row',
                         figsize=figsize(observation_dim, N),
                         squeeze=False) as (fig, axss):
        for dim, (axs, dim_name) in enumerate(zip(axss, observation_dim_names)):
            for n, ax in enumerate(axs):
                # Ground truth.
                ax.plot(domain, observations[n, :, dim], color='black', alpha=0.1, zorder=1)
                ax.scatter(domain, observations[n, :, dim], s=1, color='black', label='Truth', zorder=2)

                # Prediction boundary and learned initial value.
                ax.axvline(domain[T_train - 1], color='tuda:red', ls='dotted', label='Prediction Boundary', zorder=3)

                if dim == 0:
                    ax.set_title('Sequence %d' % (n + 1))
                if dim == observation_dim - 1:
                    ax.set_xlabel('Time Steps')
                if n == 0:
                    ax.set_ylabel(dim_name)
                ax.legend(loc='upper right').set_zorder(100)


def plot_data(out_dir: str, data_dir: str, data_file_name: str) -> None:
    data_file_path = f'{data_dir}/{data_file_name}'
    if not os.path.isfile(data_file_path):
        raise Exception(f'Data file {data_file_path} does not exist or is not a file!')

    print(f'Reading data file {data_file_path}.')
    with open(data_file_path, 'r') as file:
        input = jsonpickle.loads(file.read())
        data = input['data']
        N = input['N']
        T = input['T']
        T_train = input['T_train']
        h = input['h']
        observation_dim = input['observation_dim']
        observations = data['observations']
        observation_dim_names = input['observation_dim_names']

    plot_observations(out_dir, data_file_name.replace('.json', ''), N, h, T, T_train, observation_dim, observation_dim_names, observations)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='investigation/tmp_figures')
    parser.add_argument('-f', '--data_file_name', default='<all>')
    args = parser.parse_args()
    out_dir = args.out_dir

    data_dir = os.environ.get('DATA_DIR', 'tmp_data')
    if not os.path.isdir(data_dir):
        raise Exception(f'Data directory <{data_dir}> does not exist or is not a directory!')

    if args.data_file_name == '<all>':
        print('Creating plots for all data files.')
        data_file_names = [x for x in os.listdir(data_dir) if os.path.isfile(f'{data_dir}/{x}')]
    else:
        data_file_names = [x.strip() + '.json' for x in args.data_file_name.split('m')]

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for data_file_name in data_file_names:
        try:
            plot_data(out_dir, data_dir, data_file_name)
        except FileNotFoundError:
            print(f'No data found for {data_file_name}! Ignoring.', file=sys.stderr)


if __name__ == '__main__':
    main()
