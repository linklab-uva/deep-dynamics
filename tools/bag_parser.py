#! /usr/bin/env python3

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml
import pickle
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime


def write_dataset(bag_dir, param_file):
    # Open and parse parameter file
    with open(param_file, 'r') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    horizon = param_dict["INFO"]["HORIZON"]
    topics_of_interest = set()
    for input in param_dict["INPUTS"].values():
        topics_of_interest.add(input["TOPIC"])
    # Open bag file for reading
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    storage_filter = rosbag2_py.StorageFilter(list(topics_of_interest))
    reader.set_filter(storage_filter)
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    total_msgs = 0 
    for topic in topics_of_interest:
        total_msgs += topic_count_dict[topic] 
    # Init variables
    odometry = []
    steering = []
    throttle = []
    gear = []
    odometry_fb, odometry_timestamp, steering_fb, steering_cmd, throttle_fb, throttle_cmd, gear_fb, gear_cmd = None, 0, 0, 0, 0, 0, 0, 0
    odometry_init, steering_init, throttle_init, gear_init = False, False, False, False
    for i in tqdm(range(total_msgs), desc="Parsing bag data"):
        if (reader.has_next()):
            (topic, data, timestamp) = reader.read_next()
            msg_type = type_map[topic]
            msg_type_full = get_message(msg_type)
            msg = deserialize_message(data, msg_type_full)
            if topic == param_dict["INPUTS"]["ODOMETRY"]["TOPIC"]:
                if (odometry_init and steering_init and throttle_init and gear_init):
                    odometry.append([msg.vehicles[0].vx, msg.vehicles[0].vy, (msg.vehicles[0].yaw - odometry_fb.vehicles[0].yaw) / (datetime.fromtimestamp(timestamp * 1e-9) - datetime.fromtimestamp(odometry_timestamp * 1e-9)).total_seconds()])
                    steering.append([steering_fb, steering_cmd])
                    throttle.append([throttle_fb, throttle_cmd])
                    gear.append([gear_fb, gear_cmd])
                odometry_fb = msg
                odometry_timestamp = timestamp
                odometry_init = True
            if topic == param_dict["INPUTS"]["STEERING"]["TOPIC"]:
                steering_fb = int(msg.steering_wheel_angle)
                steering_cmd = int(msg.steering_wheel_angle_cmd)
                steering_init = True
            if topic == param_dict["INPUTS"]["THROTTLE"]["TOPIC"]:
                if msg.rear_brake_pressure * 100 // param_dict["INFO"]["MAX_BRAKE_PRESSURE"]:
                    throttle_fb = msg.rear_brake_pressure * 100 // param_dict["INFO"]["MAX_BRAKE_PRESSURE"]
                    throttle_cmd = msg.rear_brake_pressure * 100 // param_dict["INFO"]["MAX_BRAKE_PRESSURE"]
                else:
                    throttle_fb = int(msg.accel_pedal_output) 
                    throttle_cmd = int(msg.accel_pedal_input)
                throttle_init = True
            if topic == param_dict["INPUTS"]["GEAR"]["TOPIC"]:
                gear_cmd = msg.current_gear - gear_fb
                gear_fb = msg.current_gear
                gear_init = True
    features = np.zeros((len(odometry) - horizon - 1,  9, horizon))
    labels = np.zeros((len(odometry) - horizon - 1, 3))
    for i in tqdm(range(len(odometry) - horizon - 1), desc="Compiling dataset"):
        features[i] = np.hstack([odometry[i:i+horizon], steering[i:i+horizon], throttle[i:i+horizon], gear[i:i+horizon]]).T
        labels[i] = odometry[i+horizon+1]
    print("Final features shape:", features.shape)
    print("Final labels shape:", labels.shape)
    with open(os.path.join("../data/", os.path.basename(os.path.normpath(bag_dir)) + "_features_" + str(horizon) + ".pkl"), 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join("../data/", os.path.basename(os.path.normpath(bag_dir)) + "_labels_" + str(horizon) + ".pkl"), 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Convert bag file to pickled dataset")
    parser.add_argument("bag_dir", type=str, help="Bag file to convert")
    parser.add_argument("param_file", type=str, help="Param file for dataset")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    write_dataset(argdict["bag_dir"], argdict["param_file"])
