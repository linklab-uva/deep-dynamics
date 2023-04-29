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
    odometry = dstates[0:3,:].T # vx, vy, yaw_rate
    throttle_fb = inputs[0,:]
    throttle_cmd = np.append([0], inputs[0,:]).T
    steering_fb = inputs[1,:]
    steering_cmd = (inputs[1,:] - np.append([0], inputs[1,:-1])).T
    features = np.zeros((len(odometry) - horizon - 1,  horizon, 7))
    labels = np.zeros((len(odometry) - horizon - 1, 3))
    for i in tqdm(range(len(odometry) - horizon - 1), desc="Compiling dataset"):
        features[i] = np.vstack([odometry[i:i+horizon].T, np.expand_dims(steering_fb[i:i+horizon],axis=0), np.expand_dims(steering_cmd[i:i+horizon], axis=0), np.expand_dims(throttle_fb[i:i+horizon], axis=0), np.expand_dims(throttle_cmd[i:i+horizon], axis=0)]).T
        labels[i] = odometry[i+horizon+1]
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
