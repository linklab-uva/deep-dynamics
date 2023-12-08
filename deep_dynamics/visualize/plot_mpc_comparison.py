import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import os
from bayes_race.tracks import ETHZMobil
from bayes_race.models import Dynamic

#####################################################################
# settings

SAVE_RESULTS = False

SAMPLING_TIME = 0.02
HORIZON = 15

#####################################################################
# load track

N_SAMPLES = 270
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

data = np.load('../data/DYN-NMPC-NOCONS-ETHZMobil-DEEP-PACEJKA-PLUS-20.npz'.format(TRACK_NAME))
time_plus = data['time'][:N_SAMPLES+1]
states_plus = data['states'][:,:N_SAMPLES+1]
inputs_plus = data['inputs'][:,:N_SAMPLES]
dpm_plus_states = data['ddm_states'][:,:N_SAMPLES+1]

data = np.load('../data/DYN-NMPC-NOCONS-ETHZMobil-DEEP-PACEJKA-MINUS-20.npz'.format(TRACK_NAME))
time_minus = data['time'][:N_SAMPLES+1]
states_minus = data['states'][:,:N_SAMPLES+1]
inputs_minus = data['inputs'][:,:N_SAMPLES]
dpm_minus_states = data['ddm_states'][:,:N_SAMPLES+1]

#####################################################################
# plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
plt.figure(figsize=(12,8))
plt.axis('equal')
matplotlib.rcParams['pdf.fonttype'] = 42
plt.plot(track.x_outer, track.y_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(track.x_inner, track.y_inner, 'k', lw=0.5, alpha=0.5)
plt.plot(states_dyn[0], states_dyn[1], 'g', lw=1, label='DDM (ours)')
plt.plot(states_gp[0], states_gp[1], 'r', lw=1, label='DPM (GT)')
plt.plot(states_plus[0], states_plus[1], 'b', lw=1, label='DPM (+20)')
plt.plot(states_minus[0], states_minus[1], 'm', lw=1, label='DPM (-20)')

plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.1), frameon=False)
# plt.title("MPC Lap Comparison", fontweight="bold")

INDEX = 0
plt.scatter(states_gp[0,INDEX], states_gp[1,INDEX], color='r', marker='o', alpha=0.8, s=15)
plt.text(states_gp[0,INDEX], states_gp[1,INDEX]+0.05, '0', color='g', fontsize=10, ha='center', va='bottom')
plt.scatter(states_dyn[0,INDEX], states_dyn[1,INDEX], color='g', marker='o', alpha=0.8, s=15)
for INDEX in range(HORIZON+5,N_SAMPLES,HORIZON+5):
	plt.scatter(states_gp[0,INDEX], states_gp[1,INDEX], color='r', marker='o', alpha=0.8, s=15)
	if INDEX == 2 * (HORIZON+5) or INDEX == 13 * (HORIZON+5):
		plt.text(states_gp[0,INDEX]+0.05, states_gp[1,INDEX]-0.1, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	elif INDEX == 6 * (HORIZON+5):
		plt.text(states_gp[0,INDEX]+0.05, states_gp[1,INDEX]+0.1, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	elif INDEX == 9 * (HORIZON+5):
		plt.text(states_gp[0,INDEX]-0.15, states_gp[1,INDEX]+0.05, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	else:
		plt.text(states_gp[0,INDEX]+0.05, states_gp[1,INDEX]+0.05, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	plt.scatter(states_plus[0,INDEX], states_plus[1,INDEX], color='b', marker='o', alpha=0.8, s=15)
	# plt.text(states_plus[0,INDEX]+0.05, states_plus[1,INDEX]+0.05, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	plt.scatter(states_minus[0,INDEX], states_minus[1,INDEX], color='m', marker='o', alpha=0.8, s=15)
	# plt.text(states_minus[0,INDEX]+0.05, states_minus[1,INDEX]+0.05, '%.1f' % float(INDEX*SAMPLING_TIME), color='r', fontsize=18)
	plt.scatter(states_dyn[0,INDEX], states_dyn[1,INDEX], color='g', marker='o', alpha=0.8, s=15)
	if INDEX == HORIZON+5 or INDEX == 9 * (HORIZON+5):
		plt.text(states_dyn[0,INDEX]+0.15, states_dyn[1,INDEX]-0.05, "%.1f" % float(INDEX*SAMPLING_TIME), color='g', fontsize=18, ha='right', va='top')
	elif INDEX == 2 * (HORIZON+5) or INDEX == 12 * (HORIZON+5):
		plt.text(states_dyn[0,INDEX]-0.05, states_dyn[1,INDEX]+0.1, "%.1f" % float(INDEX*SAMPLING_TIME), color='g', fontsize=18, ha='right', va='top')
	else:
		plt.text(states_dyn[0,INDEX]-0.05, states_dyn[1,INDEX]-0.05, "%.1f" % float(INDEX*SAMPLING_TIME), color='g', fontsize=18, ha='right', va='top')
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(12,8))
time = np.array(range(len(inputs_dyn[0])))*0.02
# plt.axis('equal')
matplotlib.rcParams['pdf.fonttype'] = 42
print(inputs_dyn.shape)
ax[0].plot(time, inputs_dyn[0,:], 'g', label='DDM (ours)')
ax[0].plot(time, inputs_gp[0,:], 'r', label='DPM (GT)')
ax[0].plot(time, inputs_plus[0,:], 'b', label='DPM (+20)')
ax[0].plot(time, inputs_minus[0,:], 'm', label='DPM (-20)')
ax[0].set_ylabel('Throttle (%)')
ax[0].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.2), frameon=False)
# ax[0].set_xlabel("Time (s)")

ax[1].plot(time, inputs_dyn[1,:], 'g')
ax[1].plot(time, inputs_gp[1,:], 'r')
ax[1].plot(time, inputs_plus[1,:], 'b')
ax[1].plot(time, inputs_minus[1,:], 'm')
ax[1].set_ylabel('Steering Angle (rad)')
ax[1].set_xlabel("Time (s)")


# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
plt.show()



#if not os.path.exists("mpc_images/"):
#	os.mkdir("mpc_images/")
#for INDEX in range(N_SAMPLES):
#	plt.figure(figsize=(12,8))
#	plt.axis('equal')
#	plt.plot(track.x_outer, track.y_outer, 'k', lw=0.5, alpha=0.5)
#	plt.plot(track.x_inner, track.y_inner, 'k', lw=0.5, alpha=0.5)
#	plt.plot(states_dyn[0, :INDEX+1], states_dyn[1, :INDEX+1], 'g', lw=1, label='DDM (ours)')
#	plt.plot(states_gp[0, :INDEX+1], states_gp[1, :INDEX+1], 'r', lw=1, label='DPM (GT)')
#	plt.plot(states_plus[0, :INDEX+1], states_plus[1, :INDEX+1], 'b', lw=1, label='DPM (+20)')
#	plt.plot(states_minus[0, :INDEX+1], states_minus[1, :INDEX+1], 'm', lw=1, label='DPM (-20)')
#	plt.plot(states_dyn[0, INDEX], states_dyn[1, INDEX], 'g', marker='o')
#	plt.plot(states_gp[0, INDEX], states_gp[1, INDEX], 'r', marker='o')
#	plt.plot(states_plus[0, INDEX], states_plus[1, INDEX], 'b', marker='o')
#	plt.plot(states_minus[0, INDEX], states_minus[1, INDEX], 'm', marker='o')
#	plt.xlabel('$x$ [m]')
	#plt.ylabel('$y$ [m]')
#	plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.1), frameon=False)
#	plt.savefig('mpc_images/{:0>4}.png'.format(INDEX))
#	plt.close()
