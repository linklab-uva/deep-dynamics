import yaml
import os
import wandb
import torch
import numpy as np
import pickle
from deep_dynamics.model.models import string_to_dataset, string_to_model
from deep_dynamics.model.evaluate import evaluate_predictions

def numbers(x):
    try:
        idx = int(x.split("_")[1].split(".")[0])
    except:
        idx = 0
    return idx

def test_hyperparams(model_cfg, log_wandb):
    model_name = os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0]
    for dir in os.listdir("../output/{}".format(model_name)):
        hyperparam_values = []
        previous_c_digit = False
        for c in dir:
            if c.isdigit() or c == '.':
                if previous_c_digit:
                     hyperparam_values[-1] += c
                else:
                    hyperparam_values.append(c)
                previous_c_digit = True
            else:
                previous_c_digit = False
        layers = int(hyperparam_values[0])
        neurons = int(hyperparam_values[1])
        batch_size = int(hyperparam_values[2])
        learning_rate = float(hyperparam_values[3])
        horizon = int(hyperparam_values[4])
        gru_layers = int(hyperparam_values[5])
        dataset_file = "../data/LVMS_23_01_04_A_{}.npz".format(horizon)
        # dataset_file = "../data/DYN-PP-ETHZMobil_{}.npz".format(horizon)
        # dataset_file = "../data/DYN-PP-ETHZ_{}.npz".format(horizon)
        with open(model_cfg, 'rb') as f:
            param_dict = yaml.load(f, Loader=yaml.SafeLoader)
        data_npy = np.load(dataset_file)
        with open(os.path.join("../output/{}".format(model_name), dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        test_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
        # dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
        # train_dataset, test_dataset = dataset.split(0.85)
        param_dict["MODEL"]["LAYERS"] = []
        if gru_layers:
            layer = dict()
            layer["GRU"] = None
            layer["OUT_FEATURES"] = horizon ** 2
            layer["LAYERS"] = gru_layers
            param_dict["MODEL"]["LAYERS"].append(layer)
        for i in range(layers):
            layer = dict()
            layer["DENSE"] = None
            layer["OUT_FEATURES"] = neurons
            layer["ACTIVATION"] = "Mish"
            param_dict["MODEL"]["LAYERS"].append(layer)
        param_dict["MODEL"]["HORIZON"] = horizon
        param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = batch_size
        param_dict["MODEL"]["OPTIMIZATION"]["LR"] = learning_rate
        model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
        try:
            model_file = sorted(os.listdir("../output/{}/{}".format(model_name, dir)), key=numbers)[-1]
            model.load_state_dict(torch.load(os.path.join("../output/{}/{}".format(model_name, dir), model_file)))
        except:
            continue
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        print("Starting experiment: {}".format(dir))
        losses = evaluate_predictions(model, test_data_loader, False)
        if log_wandb:
            api = wandb.Api()
            run = api.runs(path="cavalier-autonomous/{}".format(model_name), filters={"display_name" : dir})[0]
            run.summary["test_loss"] = np.mean(losses)
            run.summary.update()

                        


if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("--log_wandb", action="store_true", default=False, help="Log test values to wandb experiment")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    test_hyperparams(argdict["model_cfg"], argdict["log_wandb"])
