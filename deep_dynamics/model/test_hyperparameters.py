import yaml
import os
import wandb
import torch
import numpy as np
from models import DeepDynamicsDataset, DeepDynamicsModel, DeepPacejkaModel, string_to_model
from evaluate import evaluate_predictions
import csv

def numbers(x):
    return int(x.split("_")[1].split(".")[0])

def test_hyperparams(model_cfg, output_csv, log_wandb):
    outfile = open(output_csv, 'w')
    writer = csv.writer(outfile,delimiter=',')
    model_name = os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0]
    print(model_name)
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
        horizon = int(hyperparam_values[4])
        gru_layers = int(hyperparam_values[5])
        dataset_file = "../data/LVMS_23_01_04_A_{}.npz".format(horizon)
        with open(model_cfg, 'rb') as f:
            param_dict = yaml.load(f, Loader=yaml.SafeLoader)
        test_dataset = DeepDynamicsDataset(dataset_file)
        output_layer = param_dict["MODEL"]["LAYERS"][-1]
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
        param_dict["MODEL"]["LAYERS"].append(output_layer)
        param_dict["MODEL"]["HORIZON"] = horizon
        model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
        try:
            model_file = sorted(os.listdir("../output/{}/{}".format(model_name, dir)), key=numbers)[-1]
        except IndexError:
            continue
        model.load_state_dict(torch.load(os.path.join("../output/{}/{}".format(model_name, dir), model_file)))
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        print("Starting experiment: {}".format(dir))
        losses = evaluate_predictions(model, test_data_loader, False)
        writer.writerow([losses[0], losses[1], losses[2]])
        if log_wandb:
            api = wandb.Api()
            run = api.runs(path="cavalier-autonomous/{}".format(model_name), filters={"display_name" : dir})[0]
            run.summary["test_loss"] = np.mean(losses)
            run.summary.update()

                        


if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("output_csv", type=str, help="CSV file to save results to")
    parser.add_argument("--log_wandb", action="store_true", default=False, help="Log test values to wandb experiment")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    test_hyperparams(argdict["model_cfg"], argdict["output_csv"], argdict["log_wandb"])
