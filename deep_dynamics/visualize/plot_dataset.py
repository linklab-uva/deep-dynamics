import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(file):
    dataset = np.load(file)
    features = dataset['features'][0:400]
    labels = dataset['labels'][0:400]
    time = np.linspace(0.0, 0.4*len(features), len(features+1))
    vx = features[:,-1,0]
    vy = features[:,-1,1]
    vtheta = features[:,-1,2]
    throttle = features[:,-1,3]
    steering = features[:,-1,4]
    delta_throttle = features[:,-1,5]
    delta_steering = features[:,-1,6]
    plt.plot(time, vx, label= "vx")
    plt.plot(time[1:], labels[:-1,0],'--', label="vx_label")
    plt.legend()
    plt.show()
    plt.plot(time, vy, label= "vy")
    plt.plot(time[1:], labels[:-1,1],'--', label="vy_label")
    plt.legend()
    plt.show()
    plt.plot(time, vtheta, label= "vtheta")
    plt.plot(time[1:], labels[:-1,2],'--', label="vtheta_label")
    plt.legend()
    plt.show()
    plt.plot(time, throttle, label= "throttle")
    plt.legend()
    plt.plot(time, steering, label= "steering")
    plt.legend()
    plt.plot(time, delta_throttle, label= "throttle_command")
    plt.legend()
    plt.plot(time, delta_steering, label= "steering_command")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Visualize a dataset.")
    parser.add_argument("dataset_file", type=str, help="Dataset to plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_dataset(argdict["dataset_file"])