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
        self.Y_data = torch.from_numpy(Y).float().to(device)
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
            layers.insert(1, ExtractTensor())
        else:
            self.is_rnn = False
        self.feed_forward = nn.ModuleList(layers)
        self.loss_function = string_to_torch[params["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[params["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=params["OPTIMIZATION"]["LR"])
        self.epochs = params["OPTIMIZATION"]["EPOCHS"]


    def forward(self, x, h0):
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
        hidden = weight.new(self.rnn_n_layers, self.batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden.data


    def fit(self, train_data_loader, val_data_loader):
        valid_loss_min = torch.inf
        self.train()
        for i in range(self.epochs):
            h = self.init_hidden().float()
            for inputs, labels in train_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.zero_grad()
                output, h = self.forward(inputs, h)
                loss = self.loss_function(output.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()

            val_h = self.init_hidden()
            val_losses = []
            self.eval()
            for inp, lab in val_data_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = self.forward(inp, val_h)
                val_loss = self.loss_function(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
            self.train()
            if torch.mean(val_losses) <= valid_loss_min:
                torch.save(self.state_dict(), "temp.pth")
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,torch.mean(val_losses)))
                valid_loss_min = torch.mean(val_losses)
            print("Epoch: {}/{}...".format(i+1, self.epochs),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(torch.mean(val_losses)))

if __name__ == "__main__":
    import sys
    model = DeepDynamicsModel(sys.argv[1])
    with open(sys.argv[2], 'rb') as f:
        input = pickle.load(f)
    if model.is_rnn:
        h0 = model.init_hidden()
        model(torch.from_numpy(input[0:4]).float(), h0.float())

