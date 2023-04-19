#! /usr/bin/env python3

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml
import pickle
from tqdm import tqdm
import os
import numpy as np


def write_dataset(bag_dir, param_file):
    # Open and parse parameter file
    with open(param_file, 'r') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    horizon = param_dict["HORIZON"]
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
    storage_filter = rosbag2_py.StorageFilter(list(topics_of_interest))
    reader.set_filter(storage_filter)
    metadatafile : str = os.path.join(bag_dir, "metadata.yaml")
    with open(metadatafile, "r") as f:
        metadata_dict : dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    topic_count_dict = {entry["topic_metadata"]["name"] : entry["message_count"] for entry in metadata_dict["topics_with_message_count"]}
    total_msgs = 0 
    for topic in topics_of_interest:
        total_msgs += topic_count_dict[topic] 
    for i in tqdm(range(total_msgs)):
        if (reader.has_next()):
            serialized_message = reader.read_next()






if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Convert bag file to pickled dataset")
    parser.add_argument("bag_dir", type=str, help="Bag file to convert")
    parser.add_argument("param_file", type=str, help="Param file for dataset")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    write_dataset(argdict["bag_dir"], argdict["param_file"])
