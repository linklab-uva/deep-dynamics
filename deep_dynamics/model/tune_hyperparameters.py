import yaml
import os
from functools import partial
import torch
import numpy as np
from deep_dynamics.model.models import DeepDynamicsDataset, DeepDynamicsModel, DeepPacejkaModel, string_to_model
from deep_dynamics.model.train import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def main(model_cfg, log_wandb):
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    config = {
        "layers" : tune.choice(range(1,17)),
        "neurons" : tune.randint(16, 512),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
        "lr" : tune.loguniform(1e-4, 1e-3),
        "horizon": tune.choice(range(1,17)),
        "gru_layers": tune.choice(range(17)),
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=400,
        grace_period=50,
    )
    result = tune.run(
        partial(tune_hyperparams, model_cfg=model_cfg, log_wandb=log_wandb),
        resources_per_trial={"cpu": 1, "gpu": 0.2},
        config=config,
        num_samples=200,
        scheduler=scheduler,
        storage_path="/bigtemp/jlc9wr/ray_results",
        stop={"training_iteration": 400}
        # checkpoint_at_end=True
    )

def tune_hyperparams(hyperparam_config, model_cfg, log_wandb):
    dataset_file = "/u/jlc9wr/deep-dynamics/data/LVMS_23_01_04_A_{}.npz".format(hyperparam_config["horizon"])
    with open(model_cfg, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    experiment_name = "%dlayers_%dneurons_%dbatch_%flr_%dhorizon_%dgru" % (hyperparam_config["layers"], hyperparam_config["neurons"], hyperparam_config["batch_size"], hyperparam_config["lr"], hyperparam_config["horizon"], hyperparam_config["gru_layers"])
    if not os.path.exists("/u/jlc9wr/deep-dynamics/output"):
        os.mkdir("/u/jlc9wr/deep-dynamics/output")
    if not os.path.exists("/u/jlc9wr/deep-dynamics/output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0])):
        os.mkdir("/u/jlc9wr/deep-dynamics/output/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0]))
    output_dir = "/u/jlc9wr/deep-dynamics/output/%s/%s" % (os.path.basename(os.path.normpath(model_cfg)).split('.')[0], experiment_name)
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    data_npy = np.load(dataset_file)
    dataset = DeepDynamicsDataset(data_npy["features"], data_npy["labels"])
    train_dataset, val_dataset = dataset.split(0.85)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparam_config["batch_size"], shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyperparam_config["batch_size"], shuffle=True)
    output_layer = param_dict["MODEL"]["LAYERS"][-1]
    param_dict["MODEL"]["LAYERS"] = []
    if hyperparam_config["gru_layers"]:
        layer = dict()
        layer["GRU"] = None
        layer["OUT_FEATURES"] = hyperparam_config["horizon"] ** 2
        layer["LAYERS"] = hyperparam_config["gru_layers"]
        param_dict["MODEL"]["LAYERS"].append(layer)
    for i in range(hyperparam_config["layers"]):
        layer = dict()
        layer["DENSE"] = None
        layer["OUT_FEATURES"] = hyperparam_config["neurons"]
        layer["ACTIVATION"] = "Mish"
        param_dict["MODEL"]["LAYERS"].append(layer)
    param_dict["MODEL"]["LAYERS"].append(output_layer)
    param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = hyperparam_config["batch_size"]
    param_dict["MODEL"]["OPTIMIZATION"]["LR"] = hyperparam_config["lr"]
    param_dict["MODEL"]["HORIZON"] = hyperparam_config["horizon"]
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    train(model, train_data_loader, val_data_loader, experiment_name, log_wandb, output_dir, os.path.basename(os.path.normpath(model_cfg)).split('.')[0], use_ray_tune=True)
if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Tune hyperparameters of a model")
    parser.add_argument("model_cfg", type=str, help="Config file for model. Hyperparameters listed in the dictionary will be overwritten")
    parser.add_argument("--log_wandb", action='store_true', default=False, help="Log experiment in wandb")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    main(argdict["model_cfg"], argdict["log_wandb"])
