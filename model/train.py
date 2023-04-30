from models import DeepDynamicsModel, DeepDynamicsDataset
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(model, train_data_loader, val_data_loader):
        valid_loss_min = torch.inf
        model.train()
        model.cuda()
        for i in range(model.epochs):
            h = model.init_hidden(model.batch_size)
            for inputs, labels in train_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                h = h.data
                model.zero_grad()
                output, h = model(inputs, h)
                loss = model.loss_function(output.squeeze(), labels.float())
                loss.backward()
                model.optimizer.step()
            val_losses = []
            model.eval()
            for inp, lab in val_data_loader:
                val_h = model.init_hidden(inp.shape[0])
                inp, lab = inp.to(device), lab.to(device)
                val_h = val_h.data
                out, val_h = model(inp, val_h)
                val_loss = model.loss_function(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
            model.train()
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), "temp.pth")
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
            print("Epoch: {}/{}...".format(i+1, model.epochs),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)))

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Label point clouds with bounding boxes.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("features_file", type=str, help="Features pickle file")
    parser.add_argument("labels_file", type=str, help="Labels pickle file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    model = DeepDynamicsModel(argdict["model_cfg"])
    dataset = DeepDynamicsDataset(argdict["features_file"], argdict["labels_file"])
    train_dataset, val_dataset = dataset.split(0.85)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model.batch_size, shuffle=True)
    train(model, train_data_loader, val_data_loader)
        

    