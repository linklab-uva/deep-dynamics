import yaml
import os
import comet_ml
import torch
from models import DeepDynamicsDataset, DeepDynamicsModel, DeepPacejkaModel, string_to_model
from evaluate import evaluate_predictions
import csv

def numbers(x):
    return int(x.split("_")[1].split(".")[0])

hyperparams = {
    "layers" : [4, 6, 8],
    "neurons" : [64, 128, 256],
    "batch_size": [4],
    "lr" : [0.0002],
    "horizon": [1, 2, 4, 8]
}

def test_hyperparams(model_cfg, output_csv):
    outfile = open(output_csv, 'w')
    writer = csv.writer(outfile,delimiter=',')
    for layers in hyperparams["layers"]:
        for neurons in hyperparams["neurons"]:
                for batch_size in hyperparams["batch_size"]:
                    for lr in hyperparams["lr"]:
                            for horizon in hyperparams["horizon"]:
                                dataset_file = "../data/DYN-PP-ETHZMobil_{}.npz".format(horizon)
                                with open(model_cfg, 'rb') as f:
                                    param_dict = yaml.load(f, Loader=yaml.SafeLoader)
                                experiment_name = "%dlayers_%dneurons_%dbatch_%flr_%dhorizon" % (layers, neurons, batch_size, lr, horizon)
                                test_dataset = DeepDynamicsDataset(dataset_file)
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
                                model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
                                model_name = os.path.basename(os.path.normpath(argdict["model_cfg"])).split('.')[0]
                                model_file = sorted(os.listdir("../output/{}/{}".format(model_name, experiment_name)), key=numbers)[-1]
                                model.load_state_dict(torch.load(os.path.join("../output/{}/{}".format(model_name, experiment_name), model_file)))
                                test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
                                print("Starting experiment: {}".format(experiment_name))
                                losses = evaluate_predictions(model, test_data_loader, False)
                                writer.writerow([losses[0], losses[1], losses[2]])
                        


if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("output_csv", type=str, help="CSV file to save results to")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    test_hyperparams(argdict["model_cfg"], argdict["output_csv"])
