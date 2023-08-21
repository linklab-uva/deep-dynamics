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

#####################################################################
# settings

SAVE_RESULTS = False

SAMPLING_TIME = 0.02
HORIZON = 15

#####################################################################
# load track

N_SAMPLES = 300
TRACK_NAME = 'ETHZMobil'
track = ETHZMobil(reference='optimal', longer=True)

#####################################################################
# load inputs used to simulate Dynamic model

data = np.load('../data/DYN-NMPC-NOCONS-ETHZMobil-DEEP-DYNAMICS.npz'.format(TRACK_NAME))
time_dyn = data['time'][:N_SAMPLES+1]
states_dyn = data['states'][:,:N_SAMPLES+1]
inputs_dyn = data['inputs'][:,:N_SAMPLES]

ddm_states = data['ddm_states'][:,:N_SAMPLES+1]

data = np.load('../data/DYN-NMPC-NOCONS-ETHZMobil-DEEP-PACEJKA.npz'.format(TRACK_NAME))
time_gp = data['time'][:N_SAMPLES+1]
states_gp = data['states'][:,:N_SAMPLES+1]
inputs_gp = data['inputs'][:,:N_SAMPLES]
dpm_states = data['ddm_states'][:,:N_SAMPLES+1]

data = np.load('../data/DYN-NMPC-NOCONS-ETHZMobil.npz')
times = data['time'][:N_SAMPLES+1]
states_true = data['states'][:,:N_SAMPLES+1]
ddm_predictions = data['ddm_predictions'][:,:,:N_SAMPLES+1]
dpm_predictions = data['dpm_predictions'][:,:,:N_SAMPLES+1]

#####################################################################
# plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
plt.figure(figsize=(6,4))
plt.axis('equal')
plt.plot(-track.y_outer, track.x_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(-track.y_inner, track.x_inner, 'k', lw=0.5, alpha=0.5)
plt.plot(-states_dyn[1], states_dyn[0], 'g', lw=1, label='Deep Dynamics')
plt.plot(-states_gp[1], states_gp[0], 'r', lw=1, label='Deep Pacejka')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)
plt.title("MPC Lap Comparison", fontweight="bold")

INDEX = 0
plt.scatter(-states_gp[1,INDEX], states_gp[0,INDEX], color='r', marker='o', alpha=0.8, s=15)
plt.text(-states_gp[1,INDEX], states_gp[0,INDEX]+0.05, '0', color='g', fontsize=10, ha='center', va='bottom')
plt.scatter(-states_dyn[1,INDEX], states_dyn[0,INDEX], color='g', marker='o', alpha=0.8, s=15)
for INDEX in range(HORIZON+5,N_SAMPLES,HORIZON+5):
	plt.scatter(-states_gp[1,INDEX], states_gp[0,INDEX], color='r', marker='o', alpha=0.8, s=15)
	plt.text(-states_gp[1,INDEX]+0.05, states_gp[0,INDEX]+0.05, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	plt.scatter(-states_dyn[1,INDEX], states_dyn[0,INDEX], color='g', marker='o', alpha=0.8, s=15)
	plt.text(-states_dyn[1,INDEX]-0.05, states_dyn[0,INDEX]-0.05, "%.1f" % float(INDEX*SAMPLING_TIME), color='g', fontsize=18, ha='right', va='top')

filepath = 'track_mpc.png'
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.figure(figsize=(6,4.3))
gs = gridspec.GridSpec(2,1)

plt.subplot(gs[0,:])
plt.plot(time_dyn[:-1], inputs_dyn[1], 'g', lw=1, label='Deep Dynamics')
plt.plot(time_gp[:-1], inputs_gp[1], 'r', lw=1, label='Deep Pacejka')
plt.ylabel('steering $\delta$ [rad]')
plt.xlim([0, N_SAMPLES*SAMPLING_TIME])
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.31), frameon=False)

plt.subplot(gs[1,:])
plt.plot(time_gp[:-1], inputs_gp[0], 'r', lw=1, label='MPC uses $f_{\mathrm{corr}}$')
plt.plot(time_dyn[:-1], inputs_dyn[0], 'g', lw=1, label='MPC uses $f_{\mathrm{dyn}}$')
plt.ylabel('PWM $d$ [-]')
plt.xlabel('time [s]')
plt.xlim([0, N_SAMPLES*SAMPLING_TIME])

filepath = 'inputs_mpc.png'
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.figure(figsize=(6,4))
plt.axis('equal')
plt.plot(track.x_outer, track.y_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(track.x_inner, track.y_inner, 'k', lw=0.5, alpha=0.5)
plt.plot(states_true[0], states_true[1], 'b', lw=1, label='Ground Truth')
for idx in range(20, N_SAMPLES,40):
	if idx == 20:
		plt.plot(ddm_predictions[idx, 0, :], ddm_predictions[idx, 1, :], '--go', label='Deep Dynamics')
		plt.plot(dpm_predictions[idx, 0, :], dpm_predictions[idx, 1, :], '--ro', label='Deep Pacejka')
	else:
		plt.plot(ddm_predictions[idx, 0, :], ddm_predictions[idx, 1, :], '--go')
		plt.plot(dpm_predictions[idx, 0, :], dpm_predictions[idx, 1, :], '--ro')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)
plt.title("MPC Prediction Comparison", fontweight="bold")



plt.show()