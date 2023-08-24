import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

Bf = 2.579
Cf = 1.2
Df = 0.192
Br = 3.3852
Cr = 1.2691
Dr = 0.1737

alpha = np.linspace(-0.3, 0.3, 1000)

Ffy = Df * np.sin(Cf * np.arctan(Bf * alpha))
Fry = Dr * np.sin(Cr * np.arctan(Br * alpha))

plt.plot(alpha, Fry, label="Pacejka Model")
plt.plot(alpha, Dr *Cr *Br * alpha, label="Linear Model")

plt.xlabel("Slip Angle (rad)")
plt.ylabel("Lateral Force (N)")
plt.legend()
plt.show()