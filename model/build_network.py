#!/usr/bin/env python3

import yaml
import sys
import torch

string_to_torch = {
    # Layers
    "GRU" :  torch.nn.GRU,
    "DENSE" : torch.nn.Linear,
    "LSTM" : torch.nn.LSTM,
    # Activations
    "ReLU": torch.nn.ReLU,
    "Mish": torch.nn.Mish,
    "Softplus": torch.nn.Softplus,
    # Loss Functions
    "MSE" : torch.nn.MSELoss,
    "MAE" : torch.nn.SmoothL1Loss,
    # Optimizers
    "Adam" : torch.optim.Adam,
    "NAdam" : torch.optim.NAdam,
    "AdamW" : torch.optim.AdamW
}


def build_network(param_dict):
    horizon = param_dict["MODEL"]["HORIZON"]
    num_states = len(param_dict["STATE"])
    num_actions = len(param_dict["ACTIONS"])
    num_parameters = len(param_dict["PARAMETERS"])
    layers = []
    for i in range(len(param_dict["MODEL"]["LAYERS"])):
        if i == 0:
            input_size = (num_states + num_actions) * horizon
        else:
            input_size = param_dict["MODEL"]["LAYERS"][i-1]["OUT_FEATURES"]
        if i == len(param_dict["MODEL"]["LAYERS"]) - 1:
            output_size = num_parameters
        else:
            output_size = param_dict["MODEL"]["LAYERS"][i]["OUT_FEATURES"]
        module = create_module(list(param_dict["MODEL"]["LAYERS"][i].keys())[0], input_size, horizon, output_size, param_dict["MODEL"]["LAYERS"][i].get("LAYERS"), param_dict["MODEL"]["LAYERS"][i].get("ACTIVATION"))
        layers += module
    return layers

def create_module(name, input_size, horizon, output_size, layers=None, activation=None):
    if layers:
        module = [string_to_torch[name](input_size // horizon, horizon, layers, batch_first=True)]
    elif activation:
        module = [string_to_torch[name](input_size, output_size), string_to_torch[activation]()]
    else:
        module = [string_to_torch[name](input_size, output_size)]
    return module
