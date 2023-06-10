from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from models import DeepDynamicsModel, DeepDynamicsDataset, DeepPacejkaModel
from models import string_to_model
import torch
import numpy as np
import os
import yaml

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(model, train_data_loader, val_data_loader, experiment_name, log_comet, output_dir):
        if log_comet:
            if type(model) is DeepDynamicsModel:
                experiment = Experiment(
                    api_key = "xaMmqHU4KZj6mbGh99EmEUBKp",
                    project_name = "deep-dynamics",
                    workspace="deep-dynamics",
                    )
            elif type(model) is DeepPacejkaModel:
                experiment = Experiment(
                api_key = "xaMmqHU4KZj6mbGh99EmEUBKp",
                project_name = "deep-pacejka",
                workspace="deep-dynamics"
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
            if model.is_rnn:
                h = model.init_hidden(model.batch_size)
            for inputs, labels in train_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if model.is_rnn:
                    h = h.data
                model.zero_grad()
                if model.is_rnn:
                    output, h, _ = model(inputs, h)
                else:
                    output, _, _ = model(inputs)
                loss = model.loss_function(output.squeeze(), labels.squeeze().float())
                loss.backward()
                model.optimizer.step()
            val_losses = []
            model.eval()
            for inp, lab in val_data_loader:
                if model.is_rnn:
                    val_h = model.init_hidden(inp.shape[0])
                inp, lab = inp.to(device), lab.to(device)
                if model.is_rnn:
                    val_h = val_h.data
                    out, val_h, _ = model(inp, val_h)
                else:
                    out, _, _ = model(inp)
                val_loss = model.loss_function(out.squeeze(), lab.squeeze().float())
                val_losses.append(val_loss.item())
                if log_comet:
                    experiment.log_metric("val_loss", val_loss)
            if log_comet:
                experiment.log_epoch_end(i+1)
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), "%s/epoch_%s.pth" % (output_dir, i+1))
                if log_comet:
                    log_model(experiment, model, model_name="epoch_%s.pth" % (i+1))
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
            print("Epoch: {}/{}...".format(i+1, model.epochs),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)))
            model.train()
        if log_comet:
            experiment.end()

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Train a deep dynamics model.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset", type=str, help="Dataset file")
    parser.add_argument("experiment_name", type=str, help="Name for experiment")
    parser.add_argument("--log_comet", action='store_true', default=False, help="Log experiment in comet.ml")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    with open(argdict["model_cfg"], 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    dataset = DeepDynamicsDataset(argdict["dataset"])
    if not os.path.exists("../output"):
        os.mkdir("../output")
    if not os.path.exists("../output/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0])):
        os.mkdir("../output/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0]))
    output_dir = "../output/%s/%s" % (os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0], argdict["experiment_name"])
    if not os.path.exists(output_dir):
         os.mkdir(output_dir)
    else:
         print("Experiment already exists. Choose a different name")
         exit(0)
    train_dataset, val_dataset = dataset.split(0.85)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model.batch_size, shuffle=True)
    train(model, train_data_loader, val_data_loader, argdict["experiment_name"], argdict["log_comet"], output_dir)
        

    