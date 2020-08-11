import os
from typing import List, Optional

import matplotlib.pyplot as plt



class SubplotsAndSave:
    _out_dir: str
    _file_name: str
    file_types: List[str]


    def __init__(self, out_dir: str, file_name: str, *args, file_types: Optional[List[str]] = None, **kwargs):
        if file_types is None:
            file_types = ['png']

        self._out_dir = out_dir
        self._file_name = file_name
        self._file_types = file_types
        self._args = args
        self._kwargs = kwargs

        os.makedirs(out_dir, exist_ok = True)


    def __enter__(self):
        self._fig, self._axs = plt.subplots(*self._args, **self._kwargs)
        return self._fig, self._axs


    def __exit__(self, exc_type, exc_val, exc_tb):
        for file_type in self._file_types:
            self._fig.savefig('%s/%s.%s' % (self._out_dir, self._file_name, file_type))
        plt.close(self._fig)
