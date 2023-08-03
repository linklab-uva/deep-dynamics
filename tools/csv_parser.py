#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm
import csv

def write_dataset(csv_path, horizon):
    with open(csv_path) as f:
        csv_reader = csv.reader(f, delimiter=",")
        is_column_names = True
        odometry = []
        throttle_cmds = []
        steering_cmds = []
        is_column_names = True
        previous_throttle = 0
        started = False
        for row in csv_reader:
            if is_column_names:
                is_column_names = False
                continue
            vx = float(row[3])
            if abs(vx) < 10:
                if started:
                    break
                if float(row[16]) != 0.0:
                    previous_throttle = -float(row[16]) / 2757.89990234
                else:
                    previous_throttle = float(row[15]) / 100.0
                continue
            started = True
            vy = float(row[4])
            vtheta = float(row[7])
            steering = float(row[6])
            if float(row[16]) > 1.0:
                throttle = -float(row[16]) / 2757.89990234 # max brake pressure
            else:
                throttle = previous_throttle
            steering_cmd = float(row[9]) * 0.04 # sampling time
            throttle_cmd = float(row[15]) / 100.0 - previous_throttle
            odometry.append(np.array([vx, vy, vtheta, throttle, steering]))
            throttle_cmds.append(throttle_cmd)
            steering_cmds.append(steering_cmd)
            if float(row[16]) != 0.0:
                previous_throttle = -float(row[16]) / 2757.89990234
            else:
                previous_throttle = float(row[15]) / 100.0
        odometry = np.array(odometry)
        throttle_cmds = np.array(throttle_cmds)
        steering_cmds = np.array(steering_cmds)
        features = np.zeros((len(odometry) - horizon - 1,  horizon, 7), dtype=np.double)
        labels = np.zeros((len(odometry) - horizon - 1, 3), dtype=np.double)
        for i in tqdm(range(len(throttle_cmds) - horizon - 1), desc="Compiling dataset"):
            features[i] = np.array([*odometry[i:i+horizon].T, throttle_cmds[i:i+horizon], steering_cmds[i:i+horizon]]).T
            labels[i] = np.array([*odometry[i+horizon]])[:3]
        print("Final features shape:", features.shape)
        print("Final labels shape:", labels.shape)
        np.savez(csv_path[:csv_path.find(".csv")] + "_" + str(horizon) + ".npz", features=features, labels=labels)


            
            




if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Convert CSV file to pickled dataset")
    parser.add_argument("csv_path", type=str, help="CSV file to convert")
    parser.add_argument("horizon", type=int, help="Horizon of timestamps used")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    write_dataset(argdict["csv_path"], argdict["horizon"])