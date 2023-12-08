import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import matplotlib.animation as animation 
import csv
import os
import pickle
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
ddm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features[:3000], labels[:3000], ddm_scaler)
ddm_predictions = np.zeros((len(ddm_dataset)-HORIZON, 6, HORIZON+1))
ddm_ades = np.zeros(len(ddm_dataset))
ddm_fdes = np.zeros(len(ddm_dataset))
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
		displacement_error += np.sum(np.linalg.norm(ddm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum(np.linalg.norm(ddm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	ddm_ades[idt] = displacement_error / HORIZON
	ddm_fdes[idt] = np.sum(np.linalg.norm(ddm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	idt += 1
average_displacement_error /= len(ddm_predictions)
final_displacement_error /= len(ddm_predictions)
print("DDM Average Displacement Error:", average_displacement_error)
print("DDM Final Displacement Error:", final_displacement_error)

param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/1layers_254neurons_16batch_0.000439lr_16horizon_5gru/epoch_354.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
with open(os.path.join(os.path.dirname(state_dict), "scaler.pkl"), "rb") as f:
	dpm_scaler = pickle.load(f)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features[:3000], labels[:3000], dpm_scaler)
dpm_predictions = np.zeros((len(dpm_dataset)-HORIZON, 6, HORIZON+1))
dpm_ades = np.zeros(len(dpm_dataset))
dpm_fdes = np.zeros(len(dpm_dataset))
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
		displacement_error += np.sum(np.linalg.norm(dpm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum(np.linalg.norm(dpm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	dpm_ades[idt] = displacement_error / HORIZON
	dpm_fdes[idt] = np.sum(np.linalg.norm(dpm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	idt += 1
average_displacement_error /= len(dpm_predictions)
final_displacement_error /= len(dpm_predictions)
print("DPM (GT) Average Displacement Error:", average_displacement_error)
print("DPM (GT) Final Displacement Error:", final_displacement_error)
# samples = [np.argmax(dpm_ades[:2200] - ddm_ades[:2200])]
samples = np.array(range(len(dpm_predictions)-1110, len(dpm_predictions)-100))
print(samples)
param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/plus20/epoch_391.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features[:3000], labels[:3000])
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
		displacement_error += np.sum(np.linalg.norm(dpm_plus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum(np.linalg.norm(dpm_plus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	dpm_ades[idt] = displacement_error / HORIZON
	dpm_fdes[idt] = np.sum(np.linalg.norm(dpm_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	idt += 1
average_displacement_error /= len(dpm_plus_predictions)
final_displacement_error /= len(dpm_plus_predictions)
print("DPM (+20) Average Displacement Error:", average_displacement_error)
print("DPM (+20) Final Displacement Error:", final_displacement_error)

param_file = "../cfgs/model/deep_pacejka_iac.yaml"
state_dict = "../output/deep_pacejka_iac/minus20/epoch_265.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features[:3000], labels[:3000])
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
		displacement_error += np.sum(np.linalg.norm(dpm_minus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	average_displacement_error += displacement_error / HORIZON
	final_displacement_error += np.sum(np.linalg.norm(dpm_minus_predictions[idt,:2,idh+1] - poses[idt+idh+1,:2]))
	idt += 1
average_displacement_error /= len(dpm_minus_predictions)
final_displacement_error /= len(dpm_minus_predictions)
print("DPM (-20) Average Displacement Error:", average_displacement_error)
print("DPM (-20) Final Displacement Error:", final_displacement_error)

#####################################################################
# plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
# matplotlib.use('Agg')
if not os.path.exists("images/"):
	os.mkdir("images/")
if not os.path.exists("zoomed_out/"):
	os.mkdir("zoomed_out/")
if not os.path.exists("bars/"):
	os.mkdir("bars/")
samples = range(100)
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
	plt.plot(dpm_plus_predictions[idx, 0, :], dpm_plus_predictions[idx, 1, :], '--co', label="Deep Pacejka (+20)")
	plt.plot(dpm_minus_predictions[idx, 0, :], dpm_minus_predictions[idx, 1, :], '--mo', label="Deep Pacejka (-20)")
	plt.legend(loc='upper center', ncol=2, frameon=False)
	plt.xlim(np.min(ddm_predictions[idx,0,:]) - 5.0, np.max(ddm_predictions[idx,0,:]) + 5.0)
	plt.ylim(np.min(ddm_predictions[idx,1,:]) - 5.0, np.max(ddm_predictions[idx,1,:]) + 5.0)
	plt.savefig('images/{:0>4}.png'.format(idx))
	plt.close()
	plt.figure(figsize=(12,8))
	plt.axis('equal')
	plt.plot(inner_bounds[:,0], inner_bounds[:,1],'k', lw=0.5, alpha=0.5)
	plt.plot(outer_bounds[:,0], outer_bounds[:,1],'k', lw=0.5, alpha=0.5)
	plt.plot(poses[idx,0], poses[idx,1], 'bo')
	plt.savefig('zoomed_out/{:0>4}.png'.format(idx))
	plt.xlabel('$x$ [m]')
	plt.ylabel('$y$ [m]')
	plt.close()

	fig, (ax1,ax2) = plt.subplots(1,2, figsize=(18,8))
	ax1.set_ylabel('ADE [m]')
	ax1.set_ylim([0.0, np.max(dpm_ades)])
	ax1.bar(["DDM", "DPM (GT)"], [np.mean(ddm_ades[idx-9:idx+1]), np.mean(dpm_ades[idx-9:idx+1])], color=['tab:green', 'tab:red'])
	ax2.set_ylabel('FDE [m]')
	ax2.set_ylim([0.0, np.max(dpm_fdes)])
	ax2.bar(["DDM", "DPM (GT)"], [np.mean(ddm_fdes[idx-9:idx+1]), np.mean(dpm_fdes[idx-9:idx+1])], color=['tab:green', 'tab:red'])
	plt.savefig('bars/{:0>4}.png'.format(idx))
	plt.close()


# Create a scatter plot with sorted data
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(poses[:len(ddm_dataset),0], poses[:len(ddm_dataset),1], c=ddm_ades, cmap='hot')
ade_scatter = ax[0,1].scatter(poses[:len(dpm_dataset),0], poses[:len(dpm_dataset),1], c=dpm_ades, cmap='hot')
ax[1,0].scatter(poses[:len(ddm_dataset),0], poses[:len(ddm_dataset),1], c=ddm_fdes, cmap='hot')
fde_scatter = ax[1,1].scatter(poses[:len(dpm_dataset),0], poses[:len(dpm_dataset),1], c=dpm_fdes, cmap='hot')
ax[0,0].set_ylabel('$y$ [m]')
ax[1,0].set_ylabel('$y$ [m]')
ax[1,0].set_xlabel('$x$ [m]')
ax[1,1].set_xlabel('$x$ [m]')
ax[0,0].set_title("Deep Dynamics")
ax[0,1].set_title("Deep Pacejka")


# # Add a colorbar
plt.colorbar(ade_scatter, label='ADE', ax=ax[0,:])
plt.colorbar(fde_scatter, label='FDE', ax=ax[1,:])

plt.show()