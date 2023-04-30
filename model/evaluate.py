from models import DeepDynamicsModel, DeepDynamicsDataset
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def evaluate(model, test_data_loader):
        test_losses = []
        predictions = []
        ground_truth = []

        model.eval()
        model.to(device)
        for inputs, labels in test_data_loader:
            h = model.init_hidden(inputs.shape[0])
            h = h.data
            inputs, labels = inputs.to(device), labels.to(device)
            output, h = model(inputs, h)
            test_loss = model.loss_function(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())
            predictions.append(output.squeeze())
            ground_truth.append(labels.cpu())
        print("Loss: {:.3f}".format(np.mean(test_losses)))
        return predictions, ground_truth

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Label point clouds with bounding boxes.")
    parser.add_argument("model_cfg", type=str, help="Config file for model")
    parser.add_argument("features_file", type=str, help="Features pickle file")
    parser.add_argument("labels_file", type=str, help="Labels pickle file")
    parser.add_argument("model_state_dict", type=str, help="Model weights file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    model = DeepDynamicsModel(argdict["model_cfg"])
    model.load_state_dict(torch.load(argdict["model_state_dict"]))
    test_dataset = DeepDynamicsDataset(argdict["features_file"], argdict["labels_file"])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=model.batch_size, shuffle=True)
    evaluate(model, test_data_loader)