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
plt.figure(figsize=(12,8))
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

plt.show()