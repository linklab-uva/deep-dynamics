from torch import nn
import torch
from build_network import build_network, string_to_torch
import yaml
import pickle
import numpy as np


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


class DeepDynamicsModel(nn.Module):
    def __init__(self, param_file):
        super().__init__()
        layers = build_network(param_file)
        with open(param_file, 'rb') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        self.batch_size = params["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"]
        if params["MODEL"]["LAYERS"][0].get("LAYERS"):
            self.is_rnn = True
            self.rnn_n_layers = params["MODEL"]["LAYERS"][0].get("LAYERS")
            self.rnn_hiden_dim = params["MODEL"]["HORIZON"]
            layers.insert(1, nn.Flatten())
        else:
            self.is_rnn = False
        self.feed_forward = nn.ModuleList(layers)
        self.gru = nn.GRU(7,15, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(225, 3)
        self.loss_function = string_to_torch[params["MODEL"]["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[params["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=params["MODEL"]["OPTIMIZATION"]["LR"])
        self.epochs = params["MODEL"]["OPTIMIZATION"]["NUM_EPOCHS"]
        self.state = list(params["STATE"])
        self.actions = list(params["ACTIONS"])
        self.sys_params = list(params["PARAMETERS"])
        self.vehicle_specs = params["VEHICLE_SPECS"]


    def forward(self, x, h0):
        for i in range(len(self.feed_forward)):
            if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                o, h = self.feed_forward[0](x, h0)
            else:
                o = self.feed_forward[i](o)
        o = self.differential_equation(x, o)
        return o, h
        

    def differential_equation(self, x, output):
        sys_param_dict = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        alphaf = state_action_dict["STEERING_FB"] - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"]))
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"]))
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*state_action_dict["THROTTLE_FB"] - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf))
        Fry = sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar))
        dxdt = torch.zeros(len(x), 5).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(state_action_dict["STEERING_FB"])) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(state_action_dict["STEERING_FB"])) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/sys_param_dict["Izz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(state_action_dict["STEERING_FB"]) - Fry*self.vehicle_specs["lr"])
        dxdt[:,3] = torch.minimum(state_action_dict["STEERING_CMD"], sys_param_dict["Max_steer_roc"])
        dxdt[:,4] = torch.minimum(state_action_dict["THROTTLE_CMD"], sys_param_dict["Max_throttle_roc"])
        return x[:,-1,:len(self.state)] + dxdt*state_action_dict["delta_t"][:, None]


    def unpack_sys_params(self, o):
        sys_params_dict = dict()
        for i in range(len(self.sys_params)):
            sys_params_dict[self.sys_params[i]] = o[:,i]
        return sys_params_dict 

    def unpack_state_actions(self, x):
        state_action_dict = dict()
        global_index = 0
        for i in range(len(self.state)):
            state_action_dict[self.state[i]] = x[:,-1, global_index]
            global_index += 1
        for i in range(len(self.actions)):
            state_action_dict[self.actions[i]] = x[:,-1, global_index]
            global_index += 1
        state_action_dict["delta_t"] = x[:,-1, global_index]
        return state_action_dict

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden
