from torch import nn
import torch
from build_network import build_network
import yaml

class ExtractTensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor[:, -1, :]


class DeepDynamicsModel(nn.Module):
    def __init__(self, param_file):
        super().__init__()
        layers = build_network(param_file)
        with open(param_file, 'rb') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)["MODEL"]
        self.batch_size = params["OPTIMIZATION"]["BATCH_SIZE"]
        if params["LAYERS"][0].get("LAYERS"):
            self.is_rnn = True
            self.rnn_n_layers = params["LAYERS"][0].get("LAYERS")
            self.rnn_hiden_dim = params["HORIZON"]
            layers.insert(1, ExtractTensor())
        else:
            self.is_rnn = False
        print(layers)
        self.feed_forward = nn.ModuleList(layers)


    def forward(self, x, h0):
        print(x.shape)
        for layer in self.feed_forward:
            if isinstance(layer, torch.nn.RNNBase):
                x = layer(x, h0)
            else:
                x = layer(x)
        output = self.differential_equation(x)
        return output
        

    def differential_equation(self, params):
        pass

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, self.batch_size, self.rnn_hiden_dim).zero_()
        return hidden.data


if __name__ == "__main__":
    import sys
    import pickle
    model = DeepDynamicsModel(sys.argv[1])
    with open(sys.argv[2], 'rb') as f:
        input = pickle.load(f)
    if model.is_rnn:
        h0 = model.init_hidden()
        model(torch.from_numpy(input[0:4]).float(), h0.float())

