#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm

def writeData(input_file, horizon):
    data = np.load(input_file)
    states = data["states"]
    inputs = data["inputs"]
    time = data["time"]
    odometry = states[3:,:].T # vx, vy, yaw_rate, throttle_fb, steering_fb
    throttle_cmd = inputs[0,:]
    steering_cmd = inputs[1,:]
    features = np.zeros((len(odometry) - horizon - 1,  horizon, 8))
    labels = np.zeros((len(odometry) - horizon - 1, 5))
    for i in tqdm(range(len(throttle_cmd) - horizon), desc="Compiling dataset"):
        features[i] = np.array([*odometry[i:i+horizon].T, throttle_cmd[i:i+horizon], steering_cmd[i:i+horizon], time[i:i+horizon] - time[i]]).T
        labels[i] = np.array([*odometry[i+horizon]])
    print("Final features shape:", features.shape)
    print("Final labels shape:", labels.shape)
    np.savez(input_file[:input_file.find(".npz")] + "_" + str(horizon) + ".npz", features=features, labels=labels)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./bayesrace_parser.py input_file horizon")
    else:
        writeData(sys.argv[1], int(sys.argv[2]))
