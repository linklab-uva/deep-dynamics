"""	Nonlinear MPC using dynamic bicycle model.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'

import sys
import time as tm
import numpy as np
import casadi
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZMobil
from bayes_race.mpc.planner import ConstantSpeed
from bayes_race.mpc.nmpc import setupNLP
from bayes_race.pp import purePursuit

from deep_dynamics.model.models import DeepDynamicsModel, string_to_model
import yaml
import torch
import os

#####################################################################
# CHANGE THIS

SAVE_RESULTS = True
TRACK_CONS = False

#####################################################################
# default settings

SAMPLING_TIME = 0.02
HORIZON = 13
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])
LD = 0.2
KP = 0.6

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)

#####################################################################
# deep dynamics parameters

param_file = "../cfgs/model/deep_dynamics.yaml"
state_dict = "../output/deep_dynamics/16layers_436neurons_2batch_0.000144lr_5horizon_7gru/epoch_385.pth"
with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
ddm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
ddm.cuda()
ddm.load_state_dict(torch.load(state_dict))
with open(os.path.join(os.path.dirname(state_dict), "scaler.pkl"), "rb") as f:
	ddm_scaler = pickle.load(f)

#####################################################################

# load track

TRACK_NAME = 'ETHZMobil'
track = ETHZMobil(reference='optimal', longer=True)
SIM_TIME = 6.5

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states, n_steps+1])
ddm_states = np.zeros([3, n_steps+1])
ddm_forces = np.zeros([3, n_steps+1])
dstates = np.zeros([8, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
ddm_states[:,0] = x_init[3:]
data_x = [*x_init, 0.0, 0.0]
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

# dynamic plot
fig = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=0.8)
LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(xyproj[0], xyproj[1], 'g', marker='o', alpha=0.5, markersize=5, label="current position")
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5, label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5, label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnFfy, = ax2.plot(0, 0, label='Ffy')
LnFrx, = ax2.plot(0, 0, label='Frx')
LnFry, = ax2.plot(0, 0, label='Fry')
plt.xlim([0, SIM_TIME])
plt.ylim([-params['mass']*9.81, params['mass']*9.81])
plt.xlabel('time [s]')
plt.ylabel('force [N]')
plt.legend()
plt.ion()
plt.show()

# main simulation loop
for idt in range(n_steps-horizon):
		
	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
# planner based on BayesOpt
	xref, projidx = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)
	if idt > 15:
		ddm_data = np.vstack((dstates[3:, idt-ddm.horizon:idt], inputs[:,idt-ddm.horizon+1:idt+1])).T
		ddm_data_norm = torch.from_numpy(np.expand_dims(ddm_scaler.transform(ddm_data), axis=0)).float().cuda()
		ddm_data = torch.from_numpy(np.expand_dims(ddm_data, axis=0)).float().cuda()
		if ddm.is_rnn:
			h = ddm.init_hidden(ddm_data.shape[0])
			h = h.data
			_, _, ddm_output = ddm(ddm_data, ddm_data_norm, h)
		else:
			_, _, ddm_output = ddm(ddm_data, ddm_data_norm)
		ddm_output = ddm_output.cpu().detach().numpy()[0]
		idx = 0
		for param in ddm.sys_params:
			params[param] = ddm_output[idx]
			idx += 1
		dpm_model = Dynamic(**params)
		nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, dpm_model, track, track_cons=TRACK_CONS)

		start = tm.time()
		umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)
		end = tm.time()
		inputs[:,idt] = umpc[:,0]
	else:
		start = tm.time()
		upp = purePursuit(x0, LD, KP, track, params)
		end = tm.time()
		inputs[:,idt] = upp

	print("iter: {}, time: {:.2f}".format(idt, end-start))

	# update current position with numerical integration (exact model)
	x_next, data_x = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts], data_x)
	states[:,idt+1] = x_next[:,-1]
	dstates[:,idt+1] = data_x
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])

	if idt > 15:
		# forward sim to predict over the horizon
		hstates[:,0] = x0
		hstates2[:,0] = x0
		for idh in range(horizon):
			x_next, data_x = dpm_model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts], data_x)
			hstates[:,idh+1] = x_next[:,-1]
			hstates2[:,idh+1] = xmpc[:,idh+1]

		# update plot
		LnS.set_xdata(states[0,:idt+1])
		LnS.set_ydata(states[1,:idt+1])

		LnR.set_xdata(xref[0,1:])
		LnR.set_ydata(xref[1,1:])

		LnP.set_xdata(states[0,idt])
		LnP.set_ydata(states[1,idt])

		LnH.set_xdata(hstates[0])
		LnH.set_ydata(hstates[1])

		LnH2.set_xdata(hstates2[0])
		LnH2.set_ydata(hstates2[1])
		
		LnFfy.set_xdata(time[:idt+1])
		LnFfy.set_ydata(Ffy[:idt+1])

		LnFrx.set_xdata(time[:idt+1])
		LnFrx.set_ydata(Frx[:idt+1])

		LnFry.set_xdata(time[:idt+1])
		LnFry.set_ydata(Fry[:idt+1])
	else:
		ddm_states[:,idt+1] = x_next[3:,-1]
		ddm_forces[:,idt+1] = np.array([Ffy[idt+1], Frx[idt+1], Fry[idt+1]])
	if states[0,idt] > 1.2 and idt > 100:
		print("Lap Time:", Ts * idt)
		break
	plt.pause(Ts/100)
print("Average Speed:", np.mean(states[3,:idt]))
plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-NMPC-{}{}-{}.npz'.format(SUFFIX, TRACK_NAME, "DEEP-DYNAMICS"),
		time=time,
		states=states,
		dstates=dstates,
		inputs=inputs,
		ddm_states=ddm_states,
		ddm_forces=ddm_forces,
		forces=np.vstack([Ffy, Frx, Fry])
		)

#####################################################################
# plots

# plot speed
plt.figure()
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon], label='abs')
plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.grid(True)
plt.legend()

# plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('PWM duty cycle [-]')
plt.grid(True)

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('steering [rad]')
plt.grid(True)

# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('orientation [rad]')
plt.grid(True)

plt.show()