import yaml
import os
import comet_ml
import torch
from models import DeepDynamicsDataset, DeepDynamicsModel, DeepPacejkaModel
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

outfile = open("deep_pacejka.csv", 'w')
writer = csv.writer(outfile,delimiter=',')
model_cfg = "../cfgs/model/deep_pacejka.yaml"
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
                            for i in range(8 - layers):
                                del(param_dict["MODEL"]["LAYERS"][0])
                            for i in range(layers):
                                param_dict["MODEL"]["LAYERS"][i]["OUT_FEATURES"] = neurons
                            param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = batch_size
                            param_dict["MODEL"]["OPTIMIZATION"]["LR"] = lr
                            param_dict["MODEL"]["HORIZON"] = horizon
                            ddm = DeepPacejkaModel(param_dict)
                            model_file = sorted(os.listdir("../output/deep_pacejka/{}".format(experiment_name)), key=numbers)[-1]
                            ddm.load_state_dict(torch.load(os.path.join("../output/deep_pacejka/{}".format(experiment_name), model_file)))
                            test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
                            losses = evaluate_predictions(ddm, test_data_loader, False)
                            writer.writerow([losses[0], losses[1], losses[2]])#layers, neurons, batch_size, lr, losses[0], losses[1], losses[2]])
                        
