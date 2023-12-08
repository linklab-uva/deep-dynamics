import pygeodesy

from rosbags.rosbag2 import Reader, Writer
from rosbags.serde import deserialize_cdr, serialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

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
import pygeodesy
import math
import yaml
import sys

ROSBAG_filepath =  sys.argv[1]
EGO_topics      = ['/vehicle/uva_odometry', '/vehicle/uva_acceleration', '/raptor_dbw_interface/steering_report',
                   '/raptor_dbw_interface/accelerator_pedal_report']
ego_types       = ['nav_msgs/msg/Odometry','geometry_msgs/msg/AccelWithCovarianceStamped','raptor_dbw_msgs/msg/SteeringReport',
                   'raptor_dbw_msgs/msg/AcceleratorPedalReport']
CSV_filepath    = sys.argv[2]

def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix('')
    if 'msg' not in name.parts:
        name = name.parent / 'msg' / name.name
    return str(name)

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
# Import ROS2 Types
novatel_msgs = absoluteFilePaths('/home/chros/samirauto/src/drivers/novatel_gps_driver/novatel_oem7_msgs/msg')
raptor_msgs = absoluteFilePaths('/home/chros/samirauto/src/msgs/raptor_dbw_msgs/raptor_dbw_msgs/msg')
add_types = {}

for pathstr in novatel_msgs:
    msgpath = Path(pathstr)
    msgdef = msgpath.read_text(encoding='utf-8')
    add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

for pathstr in raptor_msgs:
    msgpath = Path(pathstr)
    msgdef = msgpath.read_text(encoding='utf-8')
    add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

register_types(add_types)
from rosbags.typesys.types import novatel_oem7_msgs__msg__BESTPOS as BESTPOS
from rosbags.typesys.types import novatel_oem7_msgs__msg__BESTVEL as BESTVEL
from rosbags.typesys.types import novatel_oem7_msgs__msg__Oem7Header as Oem7Header
from rosbags.typesys.types import nav_msgs__msg__Odometry as Odometry
from rosbags.typesys.types import geometry_msgs__msg__PoseWithCovariance as PoseWithCovariance
from rosbags.typesys.types import geometry_msgs__msg__AccelWithCovarianceStamped as AccelWithCovarianceStamped
from rosbags.typesys.types import geometry_msgs__msg__TwistWithCovariance as TwistWithCovariance
from rosbags.typesys.types import geometry_msgs__msg__Pose as Pose
from rosbags.typesys.types import geometry_msgs__msg__Point as Point
from rosbags.typesys.types import geometry_msgs__msg__Quaternion as Quaternion
from rosbags.typesys.types import geometry_msgs__msg__Twist as Twist
from rosbags.typesys.types import geometry_msgs__msg__Vector3 as Vector3


def read_bag_file(bag_file, topics, topic_types):
    topic_dict = {}
    for topic in topics:
        topic_dict[topic] = []

    # create reader instance and open for reading
    with Reader(bag_file) as reader:

        # messages() accepts connection filters
        connections = [x for x in reader.connections if x.topic in topics]
        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections)):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            if connection.msgtype in topic_types:
                topic_dict[connection.topic].append((timestamp,msg))

    return topic_dict

ego_data   = read_bag_file(ROSBAG_filepath, EGO_topics, ego_types)
odom_data  = ego_data[EGO_topics[0]]
accel_data = ego_data[EGO_topics[1]]
steer_data = ego_data[EGO_topics[2]]
throttle_data = ego_data[EGO_topics[3]]

odom_state = np.zeros((len(odom_data),9))
accel_state = np.zeros((len(accel_data),2))
steer_state = np.zeros((len(steer_data),2))
throttle_state = np.zeros((len(throttle_data)))

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
    steer_state[i,1] = steer_data[i][1].steering_wheel_angle*0.0545*(math.pi/180.0)


start_time = max([odom_state[0,0],accel_state[0,0],steer_state[0,0]])
end_time   = min([odom_state[-1,0],accel_state[-1,0],steer_state[-1,0]])
time_range = end_time-start_time

data_cols    = 17
sample_times = np.arange(start_time, end_time, 0.04)
sample_data  = np.zeros((len(sample_times),data_cols))
TB_times    = np.linspace(start_time, end_time, len(throttle_state))

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
'''
if len(accel_state) > len(throttle_state) or len(accel_state) > len(brake_state):
    state_thrt = interp1d(accel_state[:len(throttle_state),0],throttle_state)
    state_brk  = interp1d(accel_state[:len(brake_state),0],brake_state)
elif len(accel_state) < len(throttle_state) or len(accel_state) < len(brake_state):
    state_thrt = interp1d(accel_state[:,0],throttle_state[:len(accel_state)])
    state_brk  = interp1d(accel_state[:,0],brake_state[:len(accel_state)])
else:
    state_thrt = interp1d(accel_state[:,0],throttle_state)
    state_brk  = interp1d(accel_state[:,0],brake_state)
'''
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

# fig, axs = plt.subplots(2)
# fig.suptitle('ANGLES')
# axs[0].set_title('ROLL (rad)')
# axs[1].set_title('PSI (rad)')
# axs[0].plot(sample_times,state_roll(sample_times), color='r')
# axs[1].plot(sample_times,state_phi(sample_times), color='r')
# plt.show()
data_save = True
if data_save:
    header = '{},{},{},{},{},{},{},{},{},{},{},{}'.format('time(s)','x(m)','y(m)','vx(m/s)','vy(m/s)','phi(rad)','delta(rad)','omega(rad/s)',
                    'ax(m/s^2)','deltadelta(rad/s)','roll(rad)','throttle_ped_cmd(%)')
    np.savetxt(CSV_filepath,sample_data,delimiter=',',fmt='%0.8f',header=header)

# def sample_data(i, sample_data, sample_times, odom_data, steer_data, wheel_data, accel_data):
#     x_time = sample_times[i]
#     sample_data[i,0] = x_time
#     times = []
#     for msg in odom_data:
#         times.append(abs(x_time - (msg[1].header.stamp.sec + msg[1].header.stamp.nanosec*1e-9)))
#     odom_index = times.index(min(times))
#     odom_msg = odom_data[odom_index][1]
#     # [x, y, vx, vy, phi, 𝛿, ω, ΔT(or acc),Δ𝛿]

#     # Quat to Yaw
#     quat = []
#     quat.append(odom_msg.pose.pose.orientation.x)
#     quat.append(odom_msg.pose.pose.orientation.y)
#     quat.append(odom_msg.pose.pose.orientation.z)
#     quat.append(odom_msg.pose.pose.orientation.w)
#     rotmat = R.from_quat(quat)
#     sample_data[i,5] = rotmat.as_euler('zyx')[0]

#     sample_data[i,7] = odom_msg.twist.twist.angular.z

#     times = []
#     for msg in steer_data:
#         times.append(abs(x_time - (msg[1].header.stamp.sec + msg[1].header.stamp.nanosec*1e-9)))
#     steer_index = times.index(min(times))
#     steer_msg = steer_data[steer_index][1]
#     sample_data[i,6] = steer_msg.steering_wheel_angle*0.0545*(math.pi/180.0)

#     times = []
#     for msg in accel_data:
#         times.append(abs(x_time - (msg[1].header.stamp.sec + msg[1].header.stamp.nanosec*1e-9)))
#     accel_index = times.index(min(times))
#     accel_msg = accel_data[accel_index][1]

#     sample_data[i,8] = accel_msg.accel.accel.linear.x

#     times = []
#     for msg in wheel_data:
#         times.append(abs(x_time - (msg[1].header.stamp.sec + msg[1].header.stamp.nanosec*1e-9)))
#     wheel_index = times.index(min(times))
#     wheel_msg = wheel_data[wheel_index][1]

#     sample_data[i,10] = wheel_msg.front_left
#     sample_data[i,11] = wheel_msg.front_right
#     sample_data[i,12] = wheel_msg.rear_left
#     sample_data[i,13] = wheel_msg.rear_right

# sample_data[1:,9] = np.diff(sample_data[:,5])
# np.savetxt('input/TMS_11_03.csv',sample_data,delimiter=',',fmt='%0.8f')

