import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

def plot_dataset(file):
    dataset = np.load(file)
    features = dataset['features']
    time = np.linspace(0.0, 0.04*len(features), len(features+1))
    vx = features[:,-1,0]
    vy = features[:,-1,1]
    vtheta = features[:,-1,2]
    throttle = features[:,-1,3]
    steering = features[:,-1,4]
    if not os.path.exists("inputs/"):
        os.mkdir("inputs/")
    matplotlib.use('Agg')
    for idx in range(len(features)):
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].set_xlim([time[0], time[-1]])
        ax[0].set_ylim([np.min(vx), np.max(vx)])
        ax[0].set_xlabel("Time (s)")
        ax[0].plot(time[:idx], vx[:idx], label="$v_x$ ($m/s$)")
        ax[0].plot(time[:idx], vy[:idx], label="$v_y$ ($m/s$)")
        ax[0].plot(time[:idx], vtheta[:idx], label="$\omega$ ($rad/s$)")
        ax[0].legend()
        ax[1].plot(time[:idx], throttle[:idx], label="Throttle (%)")
        ax[1].plot(time[:idx], steering[:idx], label="Steering Angle ($rad$)")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_xlim([time[0], time[-1]])
        ax[1].set_ylim([np.min(steering), np.max(throttle)])
        ax[1].legend()
        plt.savefig(os.path.join("inputs/", '{:0>4}.png'.format(idx)))
        plt.close()





if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Visualize a dataset.")
    parser.add_argument("dataset_file", type=str, help="Dataset to plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    plot_dataset(argdict["dataset_file"])