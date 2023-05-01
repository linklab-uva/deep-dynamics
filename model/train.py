from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from models import DeepDynamicsModel, DeepDynamicsDataset
import torch
import numpy as np
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(model, train_data_loader, val_data_loader, experiment_name, log_comet):
        if log_comet:
            experiment = Experiment(
                api_key = "xaMmqHU4KZj6mbGh99EmEUBKp",
                project_name = "deep-dynamics",
                workspace="deep-dynamics",
                )
            experiment.set_name(experiment_name)
            experiment.add_tag("lr=%f" % model.param_dict["MODEL"]["OPTIMIZATION"]["LR"])
            if model.is_rnn:
                experiment.add_tag("RNN")
            else:
                experiment.add_tag("FFNN")
            experiment.log_parameters(model.param_dict)
        valid_loss_min = torch.inf
        model.train()
        model.cuda()
        for i in range(model.epochs):
            h = model.init_hidden(model.batch_size)
            for inputs, labels in train_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                h = h.data
                model.zero_grad()
                output, h, _ = model(inputs, h)
                loss = model.loss_function(output.squeeze(), labels.float())
                loss.backward()
                model.optimizer.step()
            val_losses = []
            model.eval()
            for inp, lab in val_data_loader:
                val_h = model.init_hidden(inp.shape[0])
                inp, lab = inp.to(device), lab.to(device)
                val_h = val_h.data
                out, val_h, _ = model(inp, val_h)
                val_loss = model.loss_function(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
                if log_comet:
                    experiment.log_metric("val_loss", val_loss)
            if log_comet:
                experiment.log_epoch_end(i+1)
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), "../output/%s/epoch_%s.pth" % (experiment_name, i+1))
                if log_comet:
                    log_model(experiment, model, model_name="epoch_%s.pth" % (i+1))
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
            print("Epoch: {}/{}...".format(i+1, model.epochs),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)))
            model.train()

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Train a deep dynamics model.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset", type=str, help="Dataset file")
    parser.add_argument("experiment_name", type=str, help="Name for experiment")
    parser.add_argument("--log_comet", type=bool, default=False, help="Log experiment in comet.ml")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    if not os.path.exists("../output/"):
         os.mkdir("../output/")
    if os.path.exists("../output/%s" % (argdict["experiment_name"])):
         print("Experiment already exists. Choose a different name")
         exit(0)
    os.mkdir("../output/%s" % (argdict["experiment_name"]))
    model = DeepDynamicsModel(argdict["model_cfg"])
    dataset = DeepDynamicsDataset(argdict["dataset"])
    train_dataset, val_dataset = dataset.split(0.85)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model.batch_size, shuffle=True)
    train(model, train_data_loader, val_data_loader, argdict["experiment_name"], argdict["log_comet"])
        

    