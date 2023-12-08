#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm
import csv

MAX_BRAKE_PRESSURE = 2757.89990234
SAMPLING_TIME = 0.04

def write_dataset(csv_path, horizon, save=True):
    with open(csv_path) as f:
        csv_reader = csv.reader(f, delimiter=",")
        odometry = []
        throttle_cmds = []
        steering_cmds = []
        poses = []
        column_idxs = dict()
        previous_throttle = 0.0
        previous_steer = 0.0
        started = False
        for row in csv_reader:
            if len(column_idxs) == 0:
                for i in range(len(row)):
                    column_idxs[row[i].split("(")[0]] = i
                continue
            vx = float(row[column_idxs["vx"]])
            if abs(vx) < 5:
                if started:
                    break
                brake = float(row[column_idxs["brake_ped_cmd"]])
                throttle = float(row[column_idxs["throttle_ped_cmd"]])
                if brake > 0.0:
                    previous_throttle = -brake / MAX_BRAKE_PRESSURE
                else:
                    previous_throttle = throttle / 100.0
                previous_steer = float(row[column_idxs["delta"]])
                continue
            vy = float(row[column_idxs["vy"]])
            vtheta = float(row[column_idxs["omega"]])
            steering = float(row[column_idxs["delta"]])
            brake = float(row[column_idxs["brake_ped_cmd"]])
            if brake > 0.0:
                throttle = -brake / MAX_BRAKE_PRESSURE
            else:
                throttle =  float(row[column_idxs["throttle_ped_cmd"]]) / 100.0
            steering_cmd = steering - previous_steer
            throttle_cmd = throttle - previous_throttle
            odometry.append(np.array([vx, vy, vtheta, throttle, steering]))
            poses.append([float(row[column_idxs["x"]]), float(row[column_idxs["y"]]), float(row[column_idxs["phi"]]), vx, vy, vtheta, throttle, steering])
            previous_throttle += throttle_cmd
            previous_steer += steering_cmd
            if started:
                throttle_cmds.append(throttle_cmd)
                steering_cmds.append(steering_cmd)
            started = True
        odometry = np.array(odometry)
        throttle_cmds = np.array(throttle_cmds)
        steering_cmds = np.array(steering_cmds)
        features = np.zeros((len(throttle_cmds) - horizon - 1,  horizon, 8), dtype=np.double)
        labels = np.zeros((len(throttle_cmds) - horizon - 1, 3), dtype=np.double)
        for i in tqdm(range(len(throttle_cmds) - horizon - 1 - 5), desc="Compiling dataset"):
            features[i] = np.array([*odometry[i:i+horizon].T, throttle_cmds[i:i+horizon], steering_cmds[i:i+horizon], odometry[i+5:i+horizon+5,0]]).T
            labels[i] = np.array([*odometry[i+horizon]])[:3]
        poses = np.array(poses)
        print("Final features shape:", features.shape)
        print("Final labels shape:", labels.shape)
        if save:
            np.savez(csv_path[:csv_path.find(".csv")] + "_" + str(horizon) + ".npz", features=features, labels=labels, poses=poses)
        return features, labels, poses

            
            




if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Convert CSV file to pickled dataset")
    parser.add_argument("csv_path", type=str, help="CSV file to convert")
    parser.add_argument("horizon", type=int, help="Horizon of timestamps used")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    write_dataset(argdict["csv_path"], argdict["horizon"])