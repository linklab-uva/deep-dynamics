import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import matplotlib.animation as animation 
import csv
import os
import yaml
import torch
import pickle
from tqdm import tqdm
from deep_dynamics.model.models import string_to_model, string_to_dataset
from deep_dynamics.tools.csv_parser import write_dataset

Ts = 0.04
HORIZON = 15

#####################################################################
# load track
TRACK_NAME = "lvms"
inner_bounds = []
with open("tracks/" + TRACK_NAME + "_inner_bound.csv") as f:
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		inner_bounds.append([float(row[0]), float(row[1])])
outer_bounds = []
with open("tracks/" + TRACK_NAME + "_outer_bound.csv") as f:
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		outer_bounds.append([float(row[0]), float(row[1])])
inner_bounds = np.array(inner_bounds, dtype=np.float32)
outer_bounds = np.array(outer_bounds, dtype=np.float32)
#####################################################################
# load inputs used to simulate Dynamic model

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

param_file = "../cfgs/model/deep_dynamics_iac.yaml"
state_dict = "../output/deep_dynamics_iac/2layers_188neurons_64batch_0.001914lr_15horizon_3gru/epoch_53.pth"
dataset_file = "../data/LVMS_23_01_04_A.csv"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
with open(os.path.join(os.path.dirname(state_dict), "scaler.pkl"), "rb") as f:
	ddm_scaler = pickle.load(f)
ddm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
ddm.to(device)
ddm.eval()
ddm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, ddm.horizon, save=False)
stop_idx = len(poses) + ddm.horizon
# for i in range(len(poses)): ## Odometry set to 0 when lap is finished
# 	if poses[i,0] == 0.0 and poses[i,1] == 0.0:
# 		stop_idx = i
# 		break
samples = list(range(50, 300, 50))
driving_inputs = features[:,0,3:5] + features[:,0,5:7]
ddm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels, ddm_scaler)
ddm_predictions = np.zeros((stop_idx, 3))
ddm_data_loader = torch.utils.data.DataLoader(ddm_dataset, batch_size=1, shuffle=False)
idt = 0
states = np.zeros((stop_idx, 3))
for inputs, labels, norm_inputs in tqdm(ddm_data_loader, total=len(ddm_predictions)):
	if idt == len(ddm_predictions):
		break
	if ddm.is_rnn:
		h = ddm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if ddm.is_rnn:
		ddm_state, h, _ = ddm(inputs, norm_inputs, h)
	else:
		ddm_state, _, _ = ddm(inputs, norm_inputs)
	# Simulate model
	ddm_state = ddm_state.cpu().detach().numpy()[0]
	idx = 0
	ddm_predictions[idt+ddm.horizon,:] = ddm_state
	states[idt+ddm.horizon,:] = labels.cpu()
	idt += 1

	
# DPM GT
param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/plus20/epoch_391.pth"
with open(os.path.join(os.path.dirname(state_dict), "scaler.pkl"), "rb") as f:
	dpm_scaler = pickle.load(f)
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels, dpm_scaler)
dpm_predictions = np.zeros((stop_idx, 3))
dpm_data_loader = torch.utils.data.DataLoader(dpm_dataset, batch_size=1, shuffle=False)
idt = 0
for inputs, labels, norm_inputs in tqdm(dpm_data_loader, total=len(dpm_predictions)):
	if idt == len(dpm_predictions):
		break
	if dpm.is_rnn:
		h = dpm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if dpm.is_rnn:
		dpm_state, h, _ = dpm(inputs, norm_inputs, h)
	else:
		dpm_state, _, _ = dpm(inputs, norm_inputs)
	# Simulate model
	dpm_state = dpm_state.cpu().detach().numpy()[0]
	dpm_predictions[idt+dpm.horizon,:] = dpm_state
	idt += 1

#####################################################################
# plots

# Velocities
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 3, figsize=(18,10))
time = np.array(range(max(len(ddm_predictions), len(dpm_predictions)))) * Ts
start_idx = max(ddm.horizon, dpm.horizon)
time = time[start_idx:-20]
ddm_predictions = ddm_predictions[start_idx:-20,:]
dpm_predictions = dpm_predictions[start_idx:-20,:]
states = states[start_idx:-20,:]
ax[0,0].plot(time, states[:,0], 'b', label='Ground Truth')
ax[0,0].plot(time, ddm_predictions[:,0], '--g', label='Deep Dynamics')
ax[0,0].plot(time, dpm_predictions[:,0], '--r', label='Deep Pacejka')
ax[0,0].set(ylabel="Velocity", title="$v_x$ ($m/s$)")
ax[0,1].plot(time, states[:,1], 'b')
ax[0,1].plot(time, ddm_predictions[:,1], '--g')
ax[0,1].plot(time, dpm_predictions[:,1], '--r')
ax[0,1].set(title="$v_y$ ($m/s$)")
ax[0,2].plot(time, states[:,2], 'b',)
ax[0,2].plot(time, ddm_predictions[:,2], '--g')
ax[0,2].plot(time, dpm_predictions[:,2], '--r')
ax[0,2].set(title="$\omega$ ($rad/s$)")


## Residuals
ax[1,0].plot(time, np.abs(states[:,0] - ddm_predictions[:,0]), '--g')
ax[1,0].plot(time, np.abs(states[:,0] - dpm_predictions[:,0]), '--r')
ax[1,0].set(ylabel="Error", xlabel="Time (s)")
ax[1,1].plot(time, np.abs(states[:,1] - ddm_predictions[:,1]), '--g')
ax[1,1].plot(time, np.abs(states[:,1] - dpm_predictions[:,1]), '--r')
ax[1,1].set(xlabel="Time (s)")
ax[1,2].plot(time, np.abs(states[:,2] - ddm_predictions[:,2]), '--g')
ax[1,2].plot(time, np.abs(states[:,2] - dpm_predictions[:,2]), '--r')
ax[1,2].set(xlabel="Time (s)")

handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.0), frameon=False)

plt.show()