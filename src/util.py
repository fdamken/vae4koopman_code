import json

import numpy as np
from sacred.utils import SacredInterrupt



class LikelihoodDroppingInterrupt(SacredInterrupt):
    STATUS = 'LIKELIHOOD_DROPPED'



class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
