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
MAX_DDM_HORIZON = 10

#####################################################################
# load track

N_SAMPLES = 300
TRACK_NAME = 'ETHZMobil'
track = ETHZMobil(reference='optimal', longer=True)

#####################################################################
# load inputs used to simulate Dynamic model

data = np.load('../data/DYN-NMPC-NOCONS-ETHZMobil.npz')
time = data['time'][MAX_DDM_HORIZON:N_SAMPLES+1]
states = data['dstates'][:,MAX_DDM_HORIZON:N_SAMPLES+1]
inputs = data['inputs'][:,MAX_DDM_HORIZON:N_SAMPLES]
ddm_states = data['ddm_states'][:,MAX_DDM_HORIZON:N_SAMPLES+1]
dpm_states = data['dpm_states'][:,MAX_DDM_HORIZON:N_SAMPLES+1]
forces = data['forces'][:,MAX_DDM_HORIZON:N_SAMPLES+1]
ddm_forces = data['ddm_forces'][:,MAX_DDM_HORIZON:N_SAMPLES+1]
dpm_forces = data['dpm_forces'][:,MAX_DDM_HORIZON:N_SAMPLES+1]
print(forces.shape)
print(ddm_forces.shape)

#####################################################################
# plots

# Velocities
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 3)
ax[0,0].plot(time, states[3,:], 'b', label='Ground Truth')
ax[0,0].plot(time, ddm_states[0,:], '--g', label='Deep Dynamics')
ax[0,0].plot(time, dpm_states[0,:], '--r', label='Deep Pacejka')
ax[0,0].set(ylabel="Velocity", title="Longitudinal Velocity ($V_x$)")
ax[0,1].plot(time, states[4,:], 'b', label='Ground Truth')
ax[0,1].plot(time, ddm_states[1,:], '--g', label='Deep Dynamics')
ax[0,1].plot(time, dpm_states[1,:], '--r', label='Deep Pacejka')
ax[0,1].set(title="Lateral Velocity ($V_y$)")
ax[0,2].plot(time, states[5,:], 'b', label='Ground Truth')
ax[0,2].plot(time, ddm_states[2,:], '--g', label='Deep Dynamics')
ax[0,2].plot(time, dpm_states[2,:], '--r', label='Deep Pacejka')
ax[0,2].set(title="Yaw Rate ($V_\Theta$)")


## Residuals
ax[1,0].plot(time, np.abs(states[3,:] - ddm_states[0,:]), '--g', label='Deep Dynamics')
ax[1,0].plot(time, np.abs(states[3,:] - dpm_states[0,:]), '--r', label='Deep Pacejka')
ax[1,0].set(ylabel="Error", xlabel="Time (s)")
ax[1,1].plot(time, np.abs(states[4,:] - ddm_states[1,:]), '--g',  label='Deep Dynamics')
ax[1,1].plot(time, np.abs(states[4,:] - dpm_states[1,:]), '--r',  label='Deep Pacejka')
ax[1,1].set(xlabel="Time (s)")
ax[1,2].plot(time, np.abs(states[5,:] - ddm_states[2,:]), '--g',  label='Deep Dynamics')
ax[1,2].plot(time, np.abs(states[5,:] - dpm_states[2,:]), '--r', label='Deep Pacejka')
ax[1,2].set(xlabel="Time (s)")

handles, labels = ax[0,0].get_legend_handles_labels()
fig.suptitle("Comparison of Model Predictions vs. Time", fontweight="bold")
fig.legend(handles, labels, loc='upper right')

# Forces
# fig, ax = plt.subplots(2, 3)
# ax[0,0].plot(time, forces[0,:], 'b', label='Ground Truth')
# ax[0,0].plot(time, ddm_forces[0,:], '--g', label='Deep Dynamics')
# ax[0,0].plot(time, dpm_forces[0,:], '--r', label='Deep Pacejka')
# ax[0,0].set(ylabel="Force (N)")
# ax[0,0].legend()
# ax[0,1].plot(time, forces[1,:], 'b', label='Ground Truth')
# ax[0,1].plot(time, ddm_forces[1,:], '--g', label='Deep Dynamics')
# ax[0,1].plot(time, dpm_forces[1,:], '--r', label='Deep Pacejka')
# ax[0,1].legend()
# ax[0,2].plot(time, forces[2,:], 'b', label='Ground Truth')
# ax[0,2].plot(time, ddm_forces[2,:], '--g', label='Deep Dynamics')
# ax[0,2].plot(time, dpm_forces[2,:], '--r', label='Deep Pacejka')
# ax[0,2].legend()

# ## Residuals
# ax[1,0].plot(time, np.abs(forces[0,:] - ddm_forces[0,:]), '--g', label='Deep Dynamics')
# ax[1,0].plot(time, np.abs(forces[0,:] - dpm_forces[0,:]), '--r', label='Deep Pacejka')
# ax[1,0].set(ylabel="Error (N)")
# ax[1,0].legend()
# ax[1,1].plot(time, np.abs(forces[1,:] - ddm_forces[1,:]), '--g',  label='Deep Dynamics')
# ax[1,1].plot(time, np.abs(forces[1,:] - dpm_forces[1,:]), '--r',  label='Deep Pacejka')
# ax[1,1].legend()
# ax[1,2].plot(time, np.abs(forces[2,:] - ddm_forces[2,:]), '--g',  label='Deep Dynamics')
# ax[1,2].plot(time, np.abs(forces[2,:] - dpm_forces[2,:]), '--r', label='Deep Pacejka')
# ax[1,2].legend()

plt.show()