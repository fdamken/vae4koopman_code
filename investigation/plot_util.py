import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


TUDA_COLORS = {
        '0':  { 'a': '#DCDCDC', 'b': '#B5B5B5', 'c': '#898989', 'd': '#535353' },
        '1':  { 'a': '#5D85C3', 'b': '#005AA9', 'c': '#004E8A', 'd': '#243572' },
        '2':  { 'a': '#009CDA', 'b': '#0083CC', 'c': '#00689D', 'd': '#004E73' },
        '3':  { 'a': '#50B695', 'b': '#009D81', 'c': '#008877', 'd': '#00715E' },
        '4':  { 'a': '#AFCC50', 'b': '#99C000', 'c': '#7FAB16', 'd': '#6A8B22' },
        '5':  { 'a': '#DDDF48', 'b': '#C9D400', 'c': '#B1BD00', 'd': '#99A604' },
        '6':  { 'a': '#FFE05C', 'b': '#FDCA00', 'c': '#D7AC00', 'd': '#AE8E00' },
        '7':  { 'a': '#F8BA3C', 'b': '#F5A300', 'c': '#D28700', 'd': '#BE6F00' },
        '8':  { 'a': '#EE7A34', 'b': '#EC6500', 'c': '#CC4C03', 'd': '#A94913' },
        '9':  { 'a': '#E9503E', 'b': '#E6001A', 'c': '#B90F22', 'd': '#961C26' },
        '10': { 'a': '#C9308E', 'b': '#A60084', 'c': '#951169', 'd': '#732054' },
        '11': { 'a': '#804597', 'b': '#721085', 'c': '#611C73', 'd': '#4C226A' }
}



class SubplotsAndSave:
    _out_dir: str
    _file_name: str
    file_types: List[str]


    def __init__(self, out_dir: str, file_name: str, *args, file_types: Optional[List[str]] = None, **kwargs):
        if file_types is None:
            file_types = ['png', 'pdf']

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
        self._fig.tight_layout()
        for file_type in self._file_types:
            self._fig.savefig('%s/%s.%s' % (self._out_dir, self._file_name, file_type), dpi = 200)
        plt.close(self._fig)



def tuda(code: str):
    code = code.lower()
    if code == 'blue':
        return tuda('2b')
    elif code == 'orange':
        return tuda('7b')
    elif code == 'green':
        return tuda('3b')
    elif code == 'red':
        return tuda('9b')
    elif code == 'purple':
        return tuda('11b')
    elif code == 'pink':
        return tuda('10a')
    elif code == 'gray':
        return tuda('0c')
    elif code == 'black':
        return '#000000'
    elif code == 'white':
        return '#ffffff'
    else:
        hue_code = code[:-1]
        brightness_code = code[-1]
        if hue_code not in TUDA_COLORS:
            raise Exception('Unknown TUDa hue code %s!' % hue_code)
        if brightness_code not in TUDA_COLORS[hue_code]:
            raise Exception('Unknown TUDa brightness code %s for hue code %s!' % (brightness_code, hue_code))
        return TUDA_COLORS[hue_code][brightness_code]



def figsize(nrows: int, ncols: int) -> Tuple[int, int]:
    return 2 + 5 * ncols, 1 + 4 * nrows
