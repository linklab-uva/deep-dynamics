from deep_dynamics.model.models import string_to_dataset, string_to_model
import torch
import yaml
import os
import pickle
import numpy as np
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def compute_percent_error(predicted, target):
    percent_errors = dict()
    for key in predicted.keys():
        if target.get(key):
            percent_errors[key] = np.abs(predicted[key] - target[key]) / target[key] * 100
    return percent_errors

        

def evaluate_predictions(model, test_data_loader, eval_coeffs):
        test_losses = []
        predictions = []
        ground_truth = []
        inference_times = []
        max_errors = [0.0, 0.0, 0.0]
        model.eval()
        model.to(device)
        if eval_coeffs:
             sys_params = []
        for inputs, labels, norm_inputs in test_data_loader:
            if model.is_rnn:
                h = model.init_hidden(inputs.shape[0])
                h = h.data
            inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
            if model.is_rnn:
                start = time.time()
                output, h, sysid = model(inputs, norm_inputs, h)
                end = time.time()
            else:
                start = time.time()
                output, _, sysid = model(inputs, norm_inputs)
                end = time.time()
            inference_times.append(end-start)
#            output = model.test_sys_params(inputs)
            test_loss = model.loss_function(output.squeeze(), labels.squeeze().float())
            error = output.squeeze() - labels.squeeze().float()
            error = np.abs(error.cpu().detach().numpy())
            for i in range(3):
                if error[i] > max_errors[i]:
                    max_errors[i] = error[i]
            test_losses.append(test_loss.cpu().detach().numpy())
            predictions.append(output.squeeze())
            ground_truth.append(labels.cpu())
            if eval_coeffs:
                 sys_params.append(sysid.cpu().detach().numpy())
        print("RMSE:", np.sqrt(np.mean(test_losses, axis=0)))
        print("Maximum Error:", max_errors)
        print("Average Inference Time:", np.mean(inference_times))
        if eval_coeffs:
            means, _ = model.unpack_sys_params(np.mean(sys_params, axis=0))
            std_dev, _ = model.unpack_sys_params(np.std(sys_params, axis=0))
            percent_errors = compute_percent_error(*model.unpack_sys_params(np.mean(sys_params, axis=0)))
            print("Mean Coefficient Values")
            print("------------------------------------")
            pretty(means)
            print("Std Dev Coefficient Values")
            print("------------------------------------")
            pretty(std_dev)
            print("Percent Error")
            print("------------------------------------")
            pretty(percent_errors)
            print("------------------------------------")
        return np.mean(test_losses, axis=0)

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Label point clouds with bounding boxes.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("dataset_file", type=str, help="Dataset file")
    parser.add_argument("model_state_dict", type=str, help="Model weights file")
    parser.add_argument("--eval_coeffs", action="store_true", default=False, help="Print learned coefficients of model")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    with open(argdict["model_cfg"], 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
    model.to(device)
    model.load_state_dict(torch.load(argdict["model_state_dict"]))
    data_npy = np.load(argdict["dataset_file"])
    with open(os.path.join(os.path.dirname(argdict["model_state_dict"]), "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    test_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    losses = evaluate_predictions(model, test_data_loader, argdict["eval_coeffs"])
    print(losses)
