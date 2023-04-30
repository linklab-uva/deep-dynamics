from torch import nn
import torch
from build_network import build_network, string_to_torch
import yaml
import pickle


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DeepDynamicsDataset(torch.utils.data.Dataset):
    def __init__(self, X_file, y_file):
        with open(X_file, 'rb') as f:
            X = pickle.load(f)
        self.X_data = torch.from_numpy(X).float().to(device)
        with open(y_file, 'rb') as f:
            Y = pickle.load(f)
        self.y_data = torch.from_numpy(Y).float().to(device)
    def __len__(self):
        return(self.X_data.shape[0])
    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        return x, y
    def split(self, percent):
        split_id = int(len(self)* 0.8)
        return torch.utils.data.random_split(self, [split_id, (len(self) - split_id)])


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
            layers.insert(1, nn.Flatten())
        else:
            self.is_rnn = False
        self.feed_forward = nn.ModuleList(layers)
        self.gru = nn.GRU(7,15, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(225, 3)
        self.loss_function = string_to_torch[params["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[params["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=params["OPTIMIZATION"]["LR"])
        self.epochs = params["OPTIMIZATION"]["NUM_EPOCHS"]


    def forward(self, x, h0):
        if isinstance(self.feed_forward[0], torch.nn.RNNBase):
            o, h = self.feed_forward[0](x, h0)
            for i in range(1, len(self.feed_forward)):
                o = self.feed_forward[i](o)
        else:
            pass
        return o, h
        # output = self.differential_equation(x)
        # return output
        

    def differential_equation(self, params):
        pass


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden
