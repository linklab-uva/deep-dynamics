import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import csv
import yaml
import torch
from deep_dynamics.model.models import DeepDynamicsDataset, string_to_model
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
state_dict = "../output/deep_dynamics_iac/manual/epoch_192.pth"
dataset_file = "../data/iac/LVMS_23_01_04_A.csv"

with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
ddm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
ddm.to(device)
ddm.eval()
ddm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, ddm.horizon, save=False)
samples = set(range(2500, len(features), 500))
ddm_predictions = np.zeros((len(samples), 6, HORIZON))
ddm_dataset = DeepDynamicsDataset(features, labels)
ddm_data_loader = torch.utils.data.DataLoader(ddm_dataset, batch_size=1, shuffle=False)
global_idx = 0
idt = 0
for inputs, labels, norm_inputs in ddm_data_loader:
	if global_idx in samples:
		if ddm.is_rnn:
			h = ddm.init_hidden(inputs.shape[0])
			h = h.data
		inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
		if ddm.is_rnn:
			_, h, ddm_output, _ = ddm(inputs, norm_inputs, h)
		else:
			_, _, ddm_output, _ = ddm(inputs, norm_inputs)
		# Simulate model
		ddm_predictions[idt,:3,0] = poses[global_idx, :3]
		ddm_predictions[idt,3:,0] = features[global_idx,-1,:3]
		for idh in range(0, HORIZON-1):
			ddm_predictions[idt,0,idh+1] = ddm_predictions[idt,0,idh] + (ddm_predictions[idt,3,idh]*np.cos(ddm_predictions[idt,2,idh]) - ddm_predictions[idt,4,idh]*np.sin(ddm_predictions[idt,2,idh])) * Ts
			ddm_predictions[idt,1,idh+1] = ddm_predictions[idt,1,idh] + (ddm_predictions[idt,3,idh]*np.sin(ddm_predictions[idt,2,idh]) + ddm_predictions[idt,4,idh]*np.cos(ddm_predictions[idt,2,idh])) * Ts
			ddm_predictions[idt,2,idh+1] = ddm_predictions[idt,2,idh] + ddm_predictions[idt,5,idh] * Ts
			ddm_input = np.array([*ddm_predictions[idt,3:,idh], *features[global_idx+idh, -1, 3:]]).reshape(1,1,-1)
			dxdt, _ = ddm.differential_equation(torch.from_numpy(ddm_input).to(device), ddm_output, Ts) 
			dxdt = dxdt.cpu().detach().numpy()[-1]
			ddm_predictions[idt,3,idh+1] = dxdt[0]
			ddm_predictions[idt,4,idh+1] = dxdt[1]
			ddm_predictions[idt,5,idh+1] = dxdt[2]
		idt += 1
	global_idx += 1

	

param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/manual/epoch_321.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
samples = list(range(2500, len(features), 500))
dpm_predictions = np.zeros((len(samples), 6, HORIZON))
dpm_dataset = DeepDynamicsDataset(features, labels)
dpm_data_loader = torch.utils.data.DataLoader(dpm_dataset, batch_size=1, shuffle=False)
global_idx = 0
idt = 0
for inputs, labels, norm_inputs in dpm_data_loader:
	if global_idx in samples:
		if dpm.is_rnn:
			h = dpm.init_hidden(inputs.shape[0])
			h = h.data
		inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
		if dpm.is_rnn:
			_, h, dpm_output, _ = dpm(inputs, norm_inputs, h)
		else:
			_, _, dpm_output, _ = dpm(inputs, norm_inputs)
		# Simulate model
		dpm_predictions[idt,:3,0] = poses[global_idx, :3]
		dpm_predictions[idt,3:,0] = features[global_idx,-1,:3]
		for idh in range(0, HORIZON-1):
			dpm_predictions[idt,0,idh+1] = dpm_predictions[idt,0,idh] + (dpm_predictions[idt,3,idh]*np.cos(dpm_predictions[idt,2,idh]) - dpm_predictions[idt,4,idh]*np.sin(dpm_predictions[idt,2,idh])) * Ts
			dpm_predictions[idt,1,idh+1] = dpm_predictions[idt,1,idh] + (dpm_predictions[idt,3,idh]*np.sin(dpm_predictions[idt,2,idh]) + dpm_predictions[idt,4,idh]*np.cos(dpm_predictions[idt,2,idh])) * Ts
			dpm_predictions[idt,2,idh+1] = dpm_predictions[idt,2,idh] + dpm_predictions[idt,5,idh] * Ts
			dpm_input = np.array([*dpm_predictions[idt,3:,idh], *features[global_idx+idh, -1, 3:]]).reshape(1,1,-1)
			dxdt, _ = dpm.differential_equation(torch.from_numpy(dpm_input).to(device), dpm_output, Ts) 
			dxdt = dxdt.cpu().detach().numpy()[-1]
			dpm_predictions[idt,3,idh+1] = dxdt[0]
			dpm_predictions[idt,4,idh+1] = dxdt[1]
			dpm_predictions[idt,5,idh+1] = dxdt[2]
		idt += 1
	global_idx += 1

#####################################################################
# plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
for idx in range(len(samples)):
	plt.figure(figsize=(12,8))
	plt.axis('equal')
	plt.plot(inner_bounds[:,0], inner_bounds[:,1],'k', lw=0.5, alpha=0.5)
	plt.plot(outer_bounds[:,0], outer_bounds[:,1],'k', lw=0.5, alpha=0.5)
	plt.plot(poses[samples[idx]-500:samples[idx]+500,0], poses[samples[idx]-500:samples[idx]+500:,1], 'b', lw=1, label='Ground Truth')
	plt.xlabel('$x$ [m]')
	plt.ylabel('$y$ [m]')
	plt.plot(ddm_predictions[idx, 0, :], ddm_predictions[idx, 1, :], '--go', label="Deep Dynamics")
	plt.plot(dpm_predictions[idx, 0, :], dpm_predictions[idx, 1, :], '--ro', label="Deep Pacejka")
	plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)
	plt.xlim(np.min(ddm_predictions[idx,0,:]) - 20.0, np.max(ddm_predictions[idx,0,:]) + 20.0)
	plt.ylim(np.min(ddm_predictions[idx,1,:]) - 20.0, np.max(ddm_predictions[idx,1,:]) + 20.0)
	plt.show()