import yaml
import os
import comet_ml
import torch
from models import DeepDynamicsDataset, DeepDynamicsModel, DeepPacejkaModel, string_to_model
from train import train

hyperparams = {
    "layers" : [4, 6, 8],
    "neurons" : [64, 128, 256],
    "batch_size": [4],
    "lr" : [0.0002],
    "horizon": [1, 2, 4, 8]
}

def tune_hyperparams(model_cfg, log_comet):
    for layers in hyperparams["layers"]:
        for neurons in hyperparams["neurons"]:
                for batch_size in hyperparams["batch_size"]:
                    for lr in hyperparams["lr"]:
                            for horizon in hyperparams["horizon"]:
                                dataset_file = "../data/DYN-PP-ETHZ_{}.npz".format(horizon)
                                with open(model_cfg, 'rb') as f:
                                    param_dict = yaml.load(f, Loader=yaml.SafeLoader)
                                experiment_name = "%dlayers_%dneurons_%dbatch_%flr_%dhorizon" % (layers, neurons, batch_size, lr, horizon)
                                if not os.path.exists("../output"):
                                    os.mkdir("../output")
                                if not os.path.exists("../output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0])):
                                    os.mkdir("../output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0]))
                                output_dir = "../output/%s/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0], experiment_name)
                                if not os.path.exists(output_dir):
                                        os.mkdir(output_dir)
                                dataset = DeepDynamicsDataset(dataset_file)
                                train_dataset, val_dataset = dataset.split(0.85)
                                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                                val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                                output_layer = param_dict["MODEL"]["LAYERS"][-1]
                                param_dict["MODEL"]["LAYERS"] = []
                                for i in range(layers):
                                    layer = dict()
                                    layer["DENSE"] = None
                                    layer["OUT_FEATURES"] = neurons
                                    layer["ACTIVATION"] = "Mish"
                                    param_dict["MODEL"]["LAYERS"].append(layer)
                                param_dict["MODEL"]["LAYERS"].append(output_layer)
                                param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = batch_size
                                param_dict["MODEL"]["OPTIMIZATION"]["LR"] = lr
                                param_dict["MODEL"]["HORIZON"] = horizon
                                model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
                                print("Starting experiment: {}".format(experiment_name))
                                train(model, train_data_loader, val_data_loader, experiment_name, log_comet, output_dir)

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("--log_comet", action='store_true', default=False, help="Log experiment in comet.ml")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    tune_hyperparams(argdict["model_cfg"], argdict["log_comet"])
