"""	Plot offline data.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

from bayes_race.tracks import ETHZMobil
from bayes_race.models import Dynamic
from bayes_race.params import ORCA
import torch
import yaml
from deep_dynamics.model.models import string_to_model, DeepDynamicsDataset
from deep_dynamics.tools.bayesrace_parser import write_dataset

#####################################################################
# settings

SAVE_RESULTS = False

Ts = 0.02
HORIZON = 15

#####################################################################
# load track

N_SAMPLES = 300
TRACK_NAME = 'ETHZMobil'
track = ETHZMobil(reference='optimal', longer=True)

#####################################################################
# load inputs used to simulate Dynamic model

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

param_file = "../cfgs/model/deep_dynamics.yaml"
state_dict = "../output/deep_dynamics/13layers_108neurons_4batch_0.000317lr_8horizon_11gru/epoch_235.pth"
dataset_file = "../data/DYN-NMPC-NOCONS-ETHZMobil.npz"

with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
ddm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
ddm.to(device)
ddm.eval()
ddm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, ddm.horizon, save=False)
samples = list(range(50, 300, 50))
ddm_predictions = np.zeros((len(samples), 6, HORIZON))
ddm_dataset = DeepDynamicsDataset(features, labels)
ddm_data_loader = torch.utils.data.DataLoader(ddm_dataset, batch_size=1, shuffle=False)
params = ORCA(control='pwm')
ddm_model = Dynamic(**params)
global_idx = 0
idt = 0
for inputs, labels, norm_inputs in ddm_data_loader:
	if global_idx in samples:
		if ddm.is_rnn:
			h = ddm.init_hidden(inputs.shape[0])
			h = h.data
		inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
		if ddm.is_rnn:
			_, h, ddm_output, _ = ddm(inputs, None, h)
		else:
			_, _, ddm_output, _ = ddm(inputs, None)
		# Simulate model
		ddm_output = ddm_output.cpu().detach().numpy()[0]
		idx = 0
		for param in ddm.sys_params:
			params[param] = ddm_output[idx]
			idx += 1
		ddm_model = Dynamic(**params)
		ddm_predictions[idt,:,0] = poses[global_idx, :]
		for idh in range(HORIZON-1):
			# Predict over horizon
			ddm_next, _ = ddm_model.sim_continuous(ddm_predictions[idt,:,idh], features[global_idx+idh, 0, 5:].reshape(-1,1), [0, Ts], np.zeros((8,1)))
			ddm_predictions[idt,:,idh+1] = ddm_next[:,-1]
		idt += 1
	global_idx += 1

	

param_file = "../cfgs/model/deep_pacejka.yaml"
state_dict = "../output/deep_pacejka/9layers_40neurons_2batch_0.000403lr_4horizon_7gru//epoch_357.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
dpm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
dpm.cuda()
dpm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, dpm.horizon, save=False)
dpm_predictions = np.zeros((len(samples), 6, HORIZON))
dpm_dataset = DeepDynamicsDataset(features, labels)
dpm_data_loader = torch.utils.data.DataLoader(dpm_dataset, batch_size=1, shuffle=False)
params = ORCA(control='pwm')
dpm_model = Dynamic(**params)
global_idx = 0
idt = 0
for inputs, labels, norm_inputs in dpm_data_loader:
	if global_idx in samples:
		if dpm.is_rnn:
			h = dpm.init_hidden(inputs.shape[0])
			h = h.data
		inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
		if dpm.is_rnn:
			_, h, dpm_output, _ = dpm(inputs, None, h)
		else:
			_, _, dpm_output, _ = dpm(inputs, None)
		# Simulate model
		dpm_output = dpm_output.cpu().detach().numpy()[0]
		idx = 0
		for param in dpm.sys_params:
			params[param] = dpm_output[idx]
			idx += 1
		dpm_model = Dynamic(**params)
		dpm_predictions[idt,:,0] = poses[global_idx, :]
		for idh in range(HORIZON-1):
			# Predict over horizon
			dpm_next, _ = dpm_model.sim_continuous(dpm_predictions[idt,:,idh], features[global_idx+idh, 0, 5:].reshape(-1,1), [0, Ts], np.zeros((8,1)))
			dpm_predictions[idt,:,idh+1] = dpm_next[:,-1]
		idt += 1
	global_idx += 1

#####################################################################
# plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
plt.figure(figsize=(6,4))
plt.axis('equal')
plt.plot(track.x_outer, track.y_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(track.x_inner, track.y_inner, 'k', lw=0.5, alpha=0.5)
plt.plot(poses[:300,0], poses[:300,1], 'b', lw=1, label='Ground Truth')
legend_initialized = False
for idx in range(len(samples)):
	if not legend_initialized:
		plt.plot(ddm_predictions[idx, 0, :], ddm_predictions[idx, 1, :], '--go', label="Deep Dynamics")
		plt.plot(dpm_predictions[idx, 0, :], dpm_predictions[idx, 1, :], '--ro', label="Deep Pacejka")
		legend_initialized = True
	else:
		plt.plot(ddm_predictions[idx, 0, :], ddm_predictions[idx, 1, :], '--go')
		plt.plot(dpm_predictions[idx, 0, :], dpm_predictions[idx, 1, :], '--ro')

plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)
plt.show()