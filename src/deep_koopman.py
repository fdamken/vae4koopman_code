import collections
import pickle
from typing import Dict
import numpy as np

import torch.nn


PARAM_ACT_TYPE = 'act_type'
PARAM_NUM_DECODER_WEIGHTS = 'num_decoder_weights'
PARAM_WIDTHS = 'widths'



def _create_model(params: Dict[str, any]):
    act_type = params[PARAM_ACT_TYPE]
    num_decoder_weights = params[PARAM_NUM_DECODER_WEIGHTS]
    widths = params[PARAM_WIDTHS]
    decoder_widths = widths[-(num_decoder_weights + 1):]

    layers = collections.OrderedDict()
    for layer_id in range(1, num_decoder_weights + 1):
        layers['linear-%d' % layer_id] = torch.nn.Linear(in_features = decoder_widths[layer_id - 1], out_features = decoder_widths[layer_id], bias = True)
        if act_type == 'sigmoid':
            act = torch.nn.Sigmoid()
        elif act_type == 'relu':
            act = torch.nn.ReLU()
        elif act_type == 'elu':
            act = torch.nn.ELU()
        else:
            raise Exception('Unknown act_type <%s>!' % act_type)
        layers['act-%d' % layer_id] = act
    model = torch.nn.Sequential(layers)
    return model



def _populate_model(model: torch.nn.Sequential, result_file_prefix: str):
    trained_state_dict = { }
    for key in model.state_dict().keys():
        (layer_id, param_type) = key.split('-')[1].split('.')
        result_file_name = '%s_%sD%s.csv' % (result_file_prefix, 'W' if param_type == 'weight' else 'b', layer_id)
        param_val = np.loadtxt(result_file_name, delimiter = ',', dtype = np.float64)
        trained_state_dict[key] = torch.tensor(param_val.T)
    model.load_state_dict(trained_state_dict)



def load_model(result_file_prefix: str = 'deep_koopman_results/exp1_best/DiscreteSpectrumExample_2020_07_26_20_36_29_938456') -> torch.nn.Module:
    """
    Loads the model given by the result files of the code of Lusch et al.

    :param result_file_prefix: Prefix of all result files (e.g. %s_model.pkl would be the pickle file where %s is this string).
    :return: The model with loaded parameters of the results of deep koopman.
    """

    with open('%s_model.pkl' % result_file_prefix, 'rb') as f:
        params = pickle.load(f)

    model = _create_model(params)
    _populate_model(model, result_file_prefix)
    return model



if __name__ == '__main__':
    load_model()
