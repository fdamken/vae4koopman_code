import os
from typing import Final, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_tuda

from investigation.util import ExperimentConfig, ExperimentResult

matplotlib_tuda.load()

HIDE_DEBUG_INFO: Final[bool] = os.environ.get('HIDE_DEBUG_INFO') is not None


class SubplotsAndSave:
    _out_dir: str
    _file_name: str
    file_types: List[str]

    def __init__(self, out_dir: str, file_name: str, /, nrows: int = 1, ncols: int = 1, file_types: Optional[List[str]] = None, **kwargs):
        if file_types is None:
            file_types = ['png', 'pdf']

        self._out_dir = out_dir
        self._file_name = file_name
        self._file_types = file_types
        self._nrows = nrows
        self._ncols = ncols
        self._kwargs = kwargs

        os.makedirs(out_dir, exist_ok=True)

    def __enter__(self):
        self._fig, self._axs = plt.subplots(nrows=self._nrows, ncols=self._ncols, figsize=figsize(self._nrows, self._ncols), **self._kwargs)
        return self._fig, self._axs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fig.tight_layout()
        for file_type in self._file_types:
            self._fig.savefig('%s/%s.%s' % (self._out_dir, self._file_name, file_type), dpi=200)
        plt.close(self._fig)


def show_debug_info(fig: matplotlib.pyplot.Figure, config: ExperimentConfig, result: ExperimentResult):
    t = None
    if result.repositories is not None:
        repos = []
        for repo in result.repositories:
            if repo not in repos:
                repos.append(repo)
        texts = []
        for repo in repos:
            if len(repos) > 1:
                text = '%s@%s' % (repo.url, repo.commit)
            else:
                text = '%s' % repo.commit
            if repo.dirty:
                text += ', Dirty!'
            texts.append(text)
        t = fig.text(0, 0, '\n'.join(texts), horizontalalignment='left', verticalalignment='bottom')
        if HIDE_DEBUG_INFO:
            t.set_color('white')

    if config.result_dir is not None and config.result_file is not None:
        if config.metrics_file is None:
            text = '%s/{%s,%s}.json' % (config.result_dir, config.result_file, config.metrics_file)
        else:
            text = '%s/%s.json' % (config.result_dir, config.result_file)
        t = fig.text(1, 0, text, horizontalalignment='right', verticalalignment='bottom')
    if HIDE_DEBUG_INFO and t is not None:
        t.set_color('white')


def figsize(nrows: int, ncols: int) -> Tuple[int, int]:
    return 2 + 5 * ncols, 1 + 2 * nrows


def even(x: float) -> int:
    n = int(x)
    n += n % 2
    return n
