import wandb
from ray.air import Checkpoint, session
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

def train(model, train_data_loader, val_data_loader, experiment_name, log_wandb, output_dir, use_ray_tune=False):
    print("Starting experiment: {}".format(experiment_name))
    if log_wandb:
        if model.is_rnn:
            architecture = "RNN"
            gru_layers = model.param_dict["MODEL"]["LAYERS"][0]["LAYERS"]
            hidden_layer_size = model.param_dict["MODEL"]["LAYERS"][1]["OUT_FEATURES"]
            hidden_layers = len(model.param_dict["MODEL"]["LAYERS"]) - 2
        else:
            architecture = "FFNN"
            gru_layers = 0
            hidden_layer_size = model.param_dict["MODEL"]["LAYERS"][0]["OUT_FEATURES"]
            hidden_layers = len(model.param_dict["MODEL"]["LAYERS"]) - 1
        if type(model) is DeepDynamicsModel:
            project = "deep_dynamics"
        elif type(model) is DeepPacejkaModel:
            project = "deep_pacejka"
        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            name = experiment_name,
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": model.param_dict["MODEL"]["OPTIMIZATION"]["LR"],
            "hidden_layers" : hidden_layers,
            "hidden_layer_size" : hidden_layer_size,
            "batch_size" : model.param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"],
            "timestamps" : model.param_dict["MODEL"]["HORIZON"],
            "architecture": architecture,
            "gru_layers": gru_layers
            }
        )
    valid_loss_min = torch.inf
    model.train()
    model.cuda()
    for i in range(model.epochs):
        train_steps = 0
        train_loss_accum = 0.0
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
            train_loss_accum += loss.item()
            train_steps += 1
            loss.backward()
            model.optimizer.step()
        model.eval()
        for inp, lab in val_data_loader:
            val_steps = 0
            val_loss_accum = 0.0
            if model.is_rnn:
                val_h = model.init_hidden(inp.shape[0])
            inp, lab = inp.to(device), lab.to(device)
            if model.is_rnn:
                val_h = val_h.data
                out, val_h, _ = model(inp, val_h)
            else:
                out, _, _ = model(inp)
            val_loss = model.loss_function(out.squeeze(), lab.squeeze().float())
            val_loss_accum += val_loss.item()
            val_steps += 1
        mean_train_loss = train_loss_accum / train_steps
        mean_val_loss = val_loss_accum / val_steps
        if log_wandb:
            wandb.log({"train_loss": mean_train_loss })
            wandb.log({"val_loss": mean_val_loss})
        if mean_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), "%s/epoch_%s.pth" % (output_dir, i+1))
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,mean_val_loss))
            valid_loss_min = mean_val_loss
            if log_wandb:
                wandb.log({"best_val_loss" : mean_val_loss})
        print("Epoch: {}/{}...".format(i+1, model.epochs),
            "Loss: {:.6f}...".format(mean_train_loss),
            "Val Loss: {:.6f}".format(mean_val_loss))
        if use_ray_tune:
            checkpoint_data = {
                "epoch": i,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            session.report(
                {"loss": mean_val_loss},
                checkpoint=checkpoint,
            )
        model.train()
    wandb.finish()

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Train a deep dynamics model.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset", type=str, help="Dataset file")
    parser.add_argument("experiment_name", type=str, help="Name for experiment")
    parser.add_argument("--log_wandb", action='store_true', default=False, help="Log experiment in wandb")
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
    train(model, train_data_loader, val_data_loader, argdict["experiment_name"], argdict["log_wandb"], output_dir)
        

    
