from mcap_ros2.reader import read_ros2_messages

import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from tqdm import tqdm
from copy import copy, deepcopy

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import yaml
import sys

# Grab CLI args
ROSBAG_filepath = sys.argv[1]
CSV_filepath    = sys.argv[2]

# Topics we want
EGO_topics = [
    '/vehicle/uva_odometry',
    '/vehicle/uva_acceleration',
    '/raptor_dbw_interface/steering_extended_report',
    '/raptor_dbw_interface/accelerator_pedal_report',
    '/raptor_dbw_interface/brake_2_report'
]


def read_bag_file(bag_file, topics):
    topic_dict = {}
    for topic in topics:
        topic_dict[topic] = []

    for msg in tqdm(read_ros2_messages(source = bag_file, topics = topics)):
        ros_msg = msg.ros_msg
        if msg.channel.topic in topics:
            timestamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec*1e-9
            topic_dict[msg.channel.topic].append((timestamp, ros_msg))

    return topic_dict

ego_data   = read_bag_file(ROSBAG_filepath, EGO_topics)
odom_data  = ego_data[EGO_topics[0]]
accel_data = ego_data[EGO_topics[1]]
steer_data = ego_data[EGO_topics[2]]
throttle_data = ego_data[EGO_topics[3]]
brake_data = ego_data[EGO_topics[4]]

odom_state = np.zeros((len(odom_data),9))
accel_state = np.zeros((len(accel_data),2))
steer_state = np.zeros((len(steer_data),2))
throttle_state = np.zeros((len(throttle_data)))
brake_state = np.zeros((len(brake_data)))

for i in range(len(brake_data)):
    brake_state[i] = (brake_data[i][1].front_brake_pressure + brake_data[i][1].rear_brake_pressure)/2.0

for i in range(len(throttle_data)):
    throttle_state[i] = throttle_data[i][1].pedal_output

for i in range(len(odom_data)):
    odom_state[i,0] = odom_data[i][1].header.stamp.sec + odom_data[i][1].header.stamp.nanosec*1e-9
    odom_state[i,1] = odom_data[i][1].pose.pose.position.x
    odom_state[i,2] = odom_data[i][1].pose.pose.position.y
    odom_state[i,3] = odom_data[i][1].twist.twist.linear.x
    odom_state[i,4] = odom_data[i][1].twist.twist.linear.y

    # Quat to Yaw
    quat = []
    quat.append(odom_data[i][1].pose.pose.orientation.x)
    quat.append(odom_data[i][1].pose.pose.orientation.y)
    quat.append(odom_data[i][1].pose.pose.orientation.z)
    quat.append(odom_data[i][1].pose.pose.orientation.w)
    rotmat = R.from_quat(quat)
    yaw,pitch,roll = rotmat.as_euler('ZYX')
    odom_state[i,5] = yaw
    odom_state[i,6] = odom_data[i][1].twist.twist.angular.z
    odom_state[i,8] = roll

odom_state[1:,7] = np.diff(odom_state[:,5])

for i in range(len(accel_data)):
    accel_state[i,0] = accel_data[i][1].header.stamp.sec + accel_data[i][1].header.stamp.nanosec*1e-9
    accel_state[i,1] = accel_data[i][1].accel.accel.linear.x

for i in range(len(steer_data)):
    steer_state[i,0] = steer_data[i][1].header.stamp.sec + steer_data[i][1].header.stamp.nanosec*1e-9
    steer_state[i,1] = steer_data[i][1].primary_steering_angle_fbk*(1.0/15.0)*(math.pi/180.0)


start_time = max([odom_state[0,0],accel_state[0,0],steer_state[0,0]])
end_time   = min([odom_state[-1,0],accel_state[-1,0],steer_state[-1,0]])
time_range = end_time-start_time

data_cols    = 17
sample_times = np.arange(start_time, end_time, 0.04)
sample_data  = np.zeros((len(sample_times),data_cols))
TB_times    = np.linspace(start_time, end_time, len(throttle_state))
BT_times    = np.linspace(start_time, end_time, len(brake_state))

state_x     = interp1d(odom_state[:,0],odom_state[:,1])
state_y     = interp1d(odom_state[:,0],odom_state[:,2])
state_vx    = interp1d(odom_state[:,0],odom_state[:,3])
state_vy    = interp1d(odom_state[:,0],odom_state[:,4])
state_phi   = interp1d(odom_state[:,0],odom_state[:,5])
state_omega = interp1d(odom_state[:,0],odom_state[:,6])
state_roll  = interp1d(odom_state[:,0],odom_state[:,8])
state_delta = interp1d(steer_state[:,0],steer_state[:,1])
state_accx  = interp1d(accel_state[:,0],accel_state[:,1])
state_deltadelta = np.diff(state_delta(sample_times))/(0.04)
state_thrt = interp1d(TB_times,throttle_state)
state_brk  = interp1d(BT_times,brake_state)
sample_data[:,0] = sample_times
sample_data[:,1] = state_x(sample_times)
sample_data[:,2] = state_y(sample_times)
sample_data[:,3] = state_vx(sample_times)
sample_data[:,4] = state_vy(sample_times)
sample_data[:,5] = state_phi(sample_times)
sample_data[:,6] = state_delta(sample_times)
sample_data[:,7] = state_omega(sample_times)
sample_data[:,8] = state_accx(sample_times)
sample_data[:-1,9] = state_deltadelta
sample_data[:,10] = state_roll(sample_times)
sample_data[:,11] = state_thrt(sample_times)
sample_data[:,12] = state_brk(sample_times)

data_save = True
if data_save:
    header = '{},{},{},{},{},{},{},{},{},{},{},{},{}'.format('time(s)','x(m)','y(m)','vx(m/s)','vy(m/s)','phi(rad)','delta(rad)','omega(rad/s)',
                    'ax(m/s^2)','deltadelta(rad/s)','roll(rad)','throttle_ped_cmd(%)','brake_ped_cmd(kPa)')
    np.savetxt(CSV_filepath,sample_data,delimiter=',',fmt='%0.8f',header=header)
