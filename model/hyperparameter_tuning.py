import yaml
import os
import torch
from models import DeepDynamicsDataset, DeepDynamicsModel, DeepPacejkaModel
from train import train

hyperparams = {
    "layers" : [2, 3, 4],
    "neurons" : [32, 64, 128],
    "batch_size": [4, 8, 16],
    "lr" : [0.0001, 0.0003, 0.001, 0.003]
}
dataset_file = "../data/bayesrace/DYN-PP-ETHZ_2.npz"
model_cfg = "../cfgs/model/deep_pacejka.yaml"
for layers in hyperparams["layers"]:
      for neurons in hyperparams["neurons"]:
            for batch_size in hyperparams["batch_size"]:
                  for lr in hyperparams["lr"]:
                        with open(model_cfg, 'rb') as f:
                            param_dict = yaml.load(f, Loader=yaml.SafeLoader)
                        experiment_name = "%dlayers_%dneurons_%dbatch_%dlr" % (layers, neurons, batch_size, lr)
                        if not os.path.exists("../output"):
                            os.mkdir("../output")
                        if not os.path.exists("../output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0])):
                            os.mkdir("../output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0]))
                        output_dir = "../output/%s/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0], experiment_name)
                        if not os.path.exists(output_dir):
                                os.mkdir(output_dir)
                        dataset = DeepDynamicsDataset(dataset_file)
                        train_dataset, val_dataset = dataset.split(0.85)
                        for i in range(4 - layers):
                            del(param_dict["MODEL"]["LAYERS"][0])
                        for i in range(layers):
                             param_dict["MODEL"]["LAYERS"][i]["OUT_FEATURES"] = neurons
                        param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = batch_size
                        param_dict["MODEL"]["OPTIMIZATION"]["LR"] = lr
                        ddm = DeepPacejkaModel(param_dict)
                        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                        train(ddm, train_data_loader, val_data_loader, experiment_name, True, output_dir)

