import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import csv
import yaml
import torch
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
state_dict = "../output/deep_dynamics_iac/11layers_258neurons_32batch_0.000930lr_15horizon_15gru/epoch_90.pth"
dataset_file = "../data/LVMS_23_01_04_B.csv"

with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
ddm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
ddm.to(device)
ddm.eval()
ddm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, ddm.horizon, save=False)
samples = set(range(0, len(features), 500))
print(samples)
ddm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels)
ddm_predictions = np.zeros((len(ddm_dataset)-HORIZON, 6, HORIZON+1))
ddm_data_loader = torch.utils.data.DataLoader(ddm_dataset, batch_size=1, shuffle=False)
idt = 0
average_displacement_error = 0.0
final_displacement_error = 0.0
for inputs, labels, norm_inputs in tqdm(ddm_data_loader, total=len(ddm_predictions)):
	if idt == len(ddm_predictions):
		break
	if ddm.is_rnn:
		h = ddm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if ddm.is_rnn:
		_, h, ddm_output = ddm(inputs, norm_inputs, h)
	else:
		_, _, ddm_output = ddm(inputs, norm_inputs)
	# Simulate model
	ddm_predictions[idt,:3,0] = poses[idt, :3]
	ddm_predictions[idt,3:,0] = features[idt,-1,:3]
	displacement_error = 0
	for idh in range(0, HORIZON):
		ddm_predictions[idt,0,idh+1] = ddm_predictions[idt,0,idh] + (ddm_predictions[idt,3,idh]*np.cos(ddm_predictions[idt,2,idh]) - ddm_predictions[idt,4,idh]*np.sin(ddm_predictions[idt,2,idh])) * Ts
		ddm_predictions[idt,1,idh+1] = ddm_predictions[idt,1,idh] + (ddm_predictions[idt,3,idh]*np.sin(ddm_predictions[idt,2,idh]) + ddm_predictions[idt,4,idh]*np.cos(ddm_predictions[idt,2,idh])) * Ts
		ddm_predictions[idt,2,idh+1] = ddm_predictions[idt,2,idh] + ddm_predictions[idt,5,idh] * Ts
		ddm_input = np.array([*ddm_predictions[idt,3:,idh], *features[idt+idh, -1, 3:]]).reshape(1,1,-1)
		dxdt = ddm.differential_equation(torch.from_numpy(ddm_input).to(device), ddm_output, Ts) 
		dxdt = dxdt.cpu().detach().numpy()[-1]
		ddm_predictions[idt,3,idh+1] = dxdt[0]
		ddm_predictions[idt,4,idh+1] = dxdt[1]
		ddm_predictions[idt,5,idh+1] = dxdt[2]
		displacement_error += np.sum((ddm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum((ddm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	idt += 1
average_displacement_error /= len(ddm_predictions)
final_displacement_error /= len(ddm_predictions)
print("DDM Average Displacement Error:", average_displacement_error)
print("DDM Final Displacement Error:", final_displacement_error)

param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/5layers_417neurons_2batch_0.000100lr_3horizon_0gru/epoch_331.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels)
dpm_predictions = np.zeros((len(dpm_dataset)-HORIZON, 6, HORIZON+1))
dpm_data_loader = torch.utils.data.DataLoader(dpm_dataset, batch_size=1, shuffle=False)
idt = 0
average_displacement_error = 0.0
final_displacement_error = 0.0
for inputs, labels, norm_inputs in tqdm(dpm_data_loader, total=len(dpm_predictions)):
	if idt == len(dpm_predictions):
		break
	if dpm.is_rnn:
		h = dpm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if dpm.is_rnn:
		_, h, dpm_output = dpm(inputs, norm_inputs, h)
	else:
		_, _, dpm_output = dpm(inputs, norm_inputs)
	# Simulate model
	dpm_predictions[idt,:3,0] = poses[idt, :3]
	dpm_predictions[idt,3:,0] = features[idt,-1,:3]
	displacement_error = 0
	for idh in range(0, HORIZON):
		dpm_predictions[idt,0,idh+1] = dpm_predictions[idt,0,idh] + (dpm_predictions[idt,3,idh]*np.cos(dpm_predictions[idt,2,idh]) - dpm_predictions[idt,4,idh]*np.sin(dpm_predictions[idt,2,idh])) * Ts
		dpm_predictions[idt,1,idh+1] = dpm_predictions[idt,1,idh] + (dpm_predictions[idt,3,idh]*np.sin(dpm_predictions[idt,2,idh]) + dpm_predictions[idt,4,idh]*np.cos(dpm_predictions[idt,2,idh])) * Ts
		dpm_predictions[idt,2,idh+1] = dpm_predictions[idt,2,idh] + dpm_predictions[idt,5,idh] * Ts
		dpm_input = np.array([*dpm_predictions[idt,3:,idh], *features[idt+idh, -1, 3:]]).reshape(1,1,-1)
		dxdt = dpm.differential_equation(torch.from_numpy(dpm_input).to(device), dpm_output, Ts) 
		dxdt = dxdt.cpu().detach().numpy()[-1]
		dpm_predictions[idt,3,idh+1] = dxdt[0]
		dpm_predictions[idt,4,idh+1] = dxdt[1]
		dpm_predictions[idt,5,idh+1] = dxdt[2]
		displacement_error += np.sum((dpm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum((dpm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	idt += 1
average_displacement_error /= len(dpm_predictions)
final_displacement_error /= len(dpm_predictions)
print("DPM (GT) Average Displacement Error:", average_displacement_error)
print("DPM (GT) Final Displacement Error:", final_displacement_error)

param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/more20/epoch_140.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels)
dpm_plus_predictions = np.zeros((len(dpm_dataset)-HORIZON, 6, HORIZON+1))
dpm_data_loader = torch.utils.data.DataLoader(dpm_dataset, batch_size=1, shuffle=False)
idt = 0
average_displacement_error = 0.0
final_displacement_error = 0.0
for inputs, labels, norm_inputs in tqdm(dpm_data_loader, total=len(dpm_predictions)):
	if idt == len(dpm_predictions):
		break
	if dpm.is_rnn:
		h = dpm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if dpm.is_rnn:
		_, h, dpm_output = dpm(inputs, norm_inputs, h)
	else:
		_, _, dpm_output = dpm(inputs, norm_inputs)
	# Simulate model
	dpm_plus_predictions[idt,:3,0] = poses[idt, :3]
	dpm_plus_predictions[idt,3:,0] = features[idt,-1,:3]
	displacement_error = 0
	for idh in range(0, HORIZON):
		dpm_plus_predictions[idt,0,idh+1] = dpm_plus_predictions[idt,0,idh] + (dpm_plus_predictions[idt,3,idh]*np.cos(dpm_plus_predictions[idt,2,idh]) - dpm_plus_predictions[idt,4,idh]*np.sin(dpm_plus_predictions[idt,2,idh])) * Ts
		dpm_plus_predictions[idt,1,idh+1] = dpm_plus_predictions[idt,1,idh] + (dpm_plus_predictions[idt,3,idh]*np.sin(dpm_plus_predictions[idt,2,idh]) + dpm_plus_predictions[idt,4,idh]*np.cos(dpm_plus_predictions[idt,2,idh])) * Ts
		dpm_plus_predictions[idt,2,idh+1] = dpm_plus_predictions[idt,2,idh] + dpm_plus_predictions[idt,5,idh] * Ts
		dpm_input = np.array([*dpm_plus_predictions[idt,3:,idh], *features[idt+idh, -1, 3:]]).reshape(1,1,-1)
		dxdt = dpm.differential_equation(torch.from_numpy(dpm_input).to(device), dpm_output, Ts) 
		dxdt = dxdt.cpu().detach().numpy()[-1]
		dpm_plus_predictions[idt,3,idh+1] = dxdt[0]
		dpm_plus_predictions[idt,4,idh+1] = dxdt[1]
		dpm_plus_predictions[idt,5,idh+1] = dxdt[2]
		displacement_error += np.sum((dpm_plus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum((dpm_plus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	idt += 1
average_displacement_error /= len(dpm_plus_predictions)
final_displacement_error /= len(dpm_plus_predictions)
print("DPM (+20) Average Displacement Error:", average_displacement_error)
print("DPM (+20) Final Displacement Error:", final_displacement_error)

param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/less20/epoch_73.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels)
dpm_minus_predictions = np.zeros((len(dpm_dataset)-HORIZON, 6, HORIZON+1))
dpm_data_loader = torch.utils.data.DataLoader(dpm_dataset, batch_size=1, shuffle=False)
idt = 0
average_displacement_error = 0.0
final_displacement_error = 0.0
for inputs, labels, norm_inputs in tqdm(dpm_data_loader, total=len(dpm_predictions)):
	if idt == len(dpm_predictions):
		break
	if dpm.is_rnn:
		h = dpm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if dpm.is_rnn:
		_, h, dpm_output = dpm(inputs, norm_inputs, h)
	else:
		_, _, dpm_output = dpm(inputs, norm_inputs)
	# Simulate model
	dpm_minus_predictions[idt,:3,0] = poses[idt, :3]
	dpm_minus_predictions[idt,3:,0] = features[idt,-1,:3]
	displacement_error = 0
	for idh in range(0, HORIZON):
		dpm_minus_predictions[idt,0,idh+1] = dpm_minus_predictions[idt,0,idh] + (dpm_minus_predictions[idt,3,idh]*np.cos(dpm_minus_predictions[idt,2,idh]) - dpm_minus_predictions[idt,4,idh]*np.sin(dpm_minus_predictions[idt,2,idh])) * Ts
		dpm_minus_predictions[idt,1,idh+1] = dpm_minus_predictions[idt,1,idh] + (dpm_minus_predictions[idt,3,idh]*np.sin(dpm_minus_predictions[idt,2,idh]) + dpm_minus_predictions[idt,4,idh]*np.cos(dpm_minus_predictions[idt,2,idh])) * Ts
		dpm_minus_predictions[idt,2,idh+1] = dpm_minus_predictions[idt,2,idh] + dpm_minus_predictions[idt,5,idh] * Ts
		dpm_input = np.array([*dpm_minus_predictions[idt,3:,idh], *features[idt+idh, -1, 3:]]).reshape(1,1,-1)
		dxdt = dpm.differential_equation(torch.from_numpy(dpm_input).to(device), dpm_output, Ts) 
		dxdt = dxdt.cpu().detach().numpy()[-1]
		dpm_minus_predictions[idt,3,idh+1] = dxdt[0]
		dpm_minus_predictions[idt,4,idh+1] = dxdt[1]
		dpm_minus_predictions[idt,5,idh+1] = dxdt[2]
		displacement_error += np.sum((dpm_minus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum((dpm_minus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2])**2)
	idt += 1
average_displacement_error /= len(dpm_minus_predictions)
final_displacement_error /= len(dpm_minus_predictions)
print("DPM (+20) Average Displacement Error:", average_displacement_error)
print("DPM (+20) Final Displacement Error:", final_displacement_error)

#####################################################################
# plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
for idx in samples:
	plt.figure(figsize=(12,8))
	plt.axis('equal')
	plt.plot(inner_bounds[:,0], inner_bounds[:,1],'k', lw=0.5, alpha=0.5)
	plt.plot(outer_bounds[:,0], outer_bounds[:,1],'k', lw=0.5, alpha=0.5)
	plt.plot(poses[idx-500:idx+500,0], poses[idx-500:idx+500:,1], 'b', lw=1)
	plt.plot(poses[idx:idx+HORIZON+1,0], poses[idx:idx+HORIZON+1,1], '--bo', lw=1, label='Ground Truth')
	plt.xlabel('$x$ [m]')
	plt.ylabel('$y$ [m]')
	plt.plot(ddm_predictions[idx, 0, :], ddm_predictions[idx, 1, :], '--go', label="Deep Dynamics")
	plt.plot(dpm_predictions[idx, 0, :], dpm_predictions[idx, 1, :], '--ro', label="Deep Pacejka (GT)")
	plt.plot(dpm_plus_predictions[idx, 0, :], dpm_plus_predictions[idx, 1, :], '--bo', label="Deep Pacejka (+20)")
	plt.plot(dpm_minus_predictions[idx, 0, :], dpm_minus_predictions[idx, 1, :], '--mo', label="Deep Pacejka (-20)")
	plt.legend(loc='upper center', ncol=2, frameon=False)
	plt.xlim(np.min(ddm_predictions[idx,0,:]) - 10.0, np.max(ddm_predictions[idx,0,:]) + 10.0)
	plt.ylim(np.min(ddm_predictions[idx,1,:]) - 10.0, np.max(ddm_predictions[idx,1,:]) + 10.0)
	plt.show()
