# If you're reading this, please don't scroll. It's not winning any efficiency awards, but it works :)
import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np

hyperparams = {
    "layers" : [4, 6, 8],
    "neurons" : [64, 128, 256],
    "batch_size": [4],
    "lr" : [0.0002],
    "horizon": [1, 2, 4, 8]
}

test_csv = "csvs/deep_pacejka_horizon.csv"
# Gather data
testing_data = []
with open(test_csv) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        data = [float(x) for x in row]
        testing_data.append(np.mean(data))
# Sort data by hyperparemter
layers_neurons = dict()
layers_horizon = dict()
neurons_horizon = dict()
count = 0
for layers in hyperparams["layers"]:
    for neurons in hyperparams["neurons"]:
        for batch_size in hyperparams["batch_size"]:
            for lr in hyperparams["lr"]:
                for horizon in hyperparams["horizon"]:
                    # Layers neurons pair
                    if layers in layers_neurons.keys():
                        if neurons in layers_neurons[layers].keys():
                            layers_neurons[layers][neurons].append(testing_data[count])
                        else:
                            layers_neurons[layers][neurons] = [testing_data[count]]
                    else:
                        layers_neurons[layers] = dict()
                        layers_neurons[layers][neurons] = [testing_data[count]]
                    # Layers horizon
                    if layers in layers_horizon.keys():
                        if horizon in layers_horizon[layers].keys():
                            layers_horizon[layers][horizon].append(testing_data[count])
                        else:
                            layers_horizon[layers][horizon] = [testing_data[count]]
                    else:
                        layers_horizon[layers] = dict()
                        layers_horizon[layers][horizon] = [testing_data[count]]
                    # Neurons horizon
                    if neurons in neurons_horizon.keys():
                        if horizon in neurons_horizon[neurons].keys():
                            neurons_horizon[neurons][horizon].append(testing_data[count])
                        else:
                            neurons_horizon[neurons][horizon] = [testing_data[count]]
                    else:
                        neurons_horizon[neurons] = dict()
                        neurons_horizon[neurons][horizon] = [testing_data[count]]
                    # # Neurons lr
                    # if neurons in neurons_lr.keys():
                    #     if lr in neurons_lr[neurons].keys():
                    #         neurons_lr[neurons][lr].append(testing_data[count])
                    #     else:
                    #         neurons_lr[neurons][lr] = [testing_data[count]]
                    # else:
                    #     neurons_lr[neurons] = dict()
                    #     neurons_lr[neurons][lr] = [testing_data[count]]
                    # # Batch lr
                    # if batch_size in batch_lr.keys():
                    #     if lr in batch_lr[batch_size].keys():
                    #         batch_lr[batch_size][lr].append(testing_data[count])
                    #     else:
                    #         batch_lr[batch_size][lr] = [testing_data[count]]
                    # else:
                    #     batch_lr[batch_size] = dict()
                    #     batch_lr[batch_size][lr] = [testing_data[count]]
                    count += 1

data_wrapper = [layers_neurons, layers_horizon, neurons_horizon]#, neurons_batch, neurons_lr, batch_lr]
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
count = 0
fig, ax = plt.subplots(2,2)
zmin = 1000.0
zmax = -1.0
xdata = []
ydata = []
zdata = []
for col in range(2):
    for row in range(2):
        if col > row:
            continue
        x = []
        y = []
        z = []
        for key1 in data_wrapper[count].keys():
            for key2 in data_wrapper[count][key1].keys():
                x.append(key1)
                y.append(key2)
                z.append(np.mean(data_wrapper[count][key1][key2]))
                if z[-1] > zmax:
                    zmax = z[-1]
                if z[-1] < zmin:
                    zmin = z[-1]
        xdata.append(x)
        ydata.append(y)
        zdata.append(z)
        count += 1
count = 0

for col in range(2):
    for row in range(2):
        if col > row:
            continue
        contour = ax[row,col].tricontourf(xdata[count], ydata[count], zdata[count], cmap = 'viridis_r', vmin=zmin, vmax=zmax)#, norm=matplotlib.colors.LogNorm())
        count += 1

print(zmin, zmax)
fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(contour, ax=ax.ravel().tolist())
# cbar.set_ticks([1.0e-3, 1.0e-1, 1.0e1, 1.0e3])
# cbar.set_ticklabels([1.0e-3, 1.0e-1, 1.0e1, 1.0e3])
plt.show()
