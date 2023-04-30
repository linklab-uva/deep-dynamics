#!/usr/bin/env python3

import numpy as np
import sys
import pickle
from tqdm import tqdm
import os

def writeData(input_file, horizon):
    data = np.load(input_file)
    dstates = data["dstates"]
    inputs = data["inputs"]
    time = data["time"]
    odometry = dstates[0:3,:].T # vx, vy, yaw_rate
    throttle_fb = inputs[0,:]
    throttle_cmd = np.append([0], inputs[0,:]).T
    steering_fb = inputs[1,:]
    print(odometry.shape, throttle_fb.shape, steering_fb.shape)
    steering_cmd = (inputs[1,:] - np.append([0], inputs[1,:-1])).T
    features = np.zeros((len(odometry) - horizon - 1,  horizon, 8))
    labels = np.zeros((len(odometry) - horizon - 1, 5))
    for i in tqdm(range(len(throttle_fb) - horizon - 1), desc="Compiling dataset"):
        features[i] = np.array([*odometry[i:i+horizon].T, steering_fb[i:i+horizon], steering_cmd[i:i+horizon], throttle_fb[i:i+horizon],throttle_cmd[i:i+horizon], time[i:i+horizon] - time[i]]).T
        labels[i] = np.array([*odometry[i+horizon+1], steering_fb[i+horizon+1], throttle_fb[i+horizon+1]])
    print("Final features shape:", features.shape)
    print("Final labels shape:", labels.shape)
    with open(os.path.join("../data/bayesrace/", os.path.basename(os.path.normpath(input_file)) + "_features_" + str(horizon) + ".pkl"), 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join("../data/bayesrace/", os.path.basename(os.path.normpath(input_file)) + "_labels_" + str(horizon) + ".pkl"), 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./bayesrace_parser.py input_file horizon")
    else:
        writeData(sys.argv[1], int(sys.argv[2]))
