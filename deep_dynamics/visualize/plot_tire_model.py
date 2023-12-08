import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

# Gt Coeffs
Bf_gt = 5.579
Cf_gt = 1.2
Df_gt = 0.192
Ef_gt = -0.083
Br_gt = 5.385
Cr_gt = 1.269
Dr_gt = 0.173
Er_gt = -0.019

# DDM Coeffs

Bf_ddm = 5.566
Cf_ddm = 1.203
Df_ddm = 0.192
Ef_ddm = -0.081
Br_ddm = 5.505
Cr_ddm = 1.237
Dr_ddm = 0.174
Er_ddm = -0.070

# Bf_ddm = 6.630
# Cf_ddm = 1.047
# Df_ddm = 3451.427
# Ef_ddm = -1.051
# Br_ddm = 15.426
# Cr_ddm = 0.777
# Dr_ddm = 5335.26
# Er_ddm = -0.733

# DPM Coeffs
# Bf_dpm = -4.010
# Cf_dpm = -0.616
# Df_dpm = 0.499
# Ef_dpm = 1.554
# Br_dpm = -1.897
# Cr_dpm = -0.790
# Dr_dpm = 0.763
# Er_dpm = 8.542

# Bf_dpm = 5.763
# Cf_dpm = -28.338
# Df_dpm = -1249.052
# Ef_dpm = 27.035
# Br_dpm = 0.095
# Cr_dpm = 9.587
# Dr_dpm = 863.780
# Er_dpm = 12.189

# Experimental
Bf_dpm = 10.988
Cf_dpm = 1.290
Df_dpm = 11651.23
Ef_dpm = -5.858
Br_dpm = 0.095
Cr_dpm = 9.587
Dr_dpm = 863.780
Er_dpm = 12.189


alpha = np.linspace(-0.5, 0.5, 1000)

# fig, ax = plt.subplots(1,2, figsize=(24,8))
# Ffy_gt = Df_gt * np.sin(Cf_gt * np.arctan(Bf_gt * alpha - Ef_gt * (Bf_gt * alpha - np.arctan(Bf_gt * alpha))))
# Ffy_ddm = Df_ddm * np.sin(Cf_ddm * np.arctan(Bf_ddm * alpha - Ef_ddm * (Bf_ddm * alpha - np.arctan(Bf_ddm * alpha))))
# Ffy_dpm = Df_dpm * np.sin(Cf_dpm * np.arctan(Bf_dpm * alpha - Ef_dpm * (Bf_dpm * alpha - np.arctan(Bf_dpm * alpha))))
# ax[0].plot(alpha, Ffy_gt, 'b', lw=3, label="Ground Truth")
# ax[0].plot(alpha, Ffy_ddm, '--g', lw=3,  label="Deep Dynamics")
# ax[0].plot(alpha, Ffy_dpm, '--r', lw=3, label="Deep Pacejka")
# ax[0].set_xlabel("Side Slip Angle (rad)")
# ax[0].set_ylabel("Front Lateral Force (N)")
# ax[0].grid()
# Fry_gt = Dr_gt * np.sin(Cr_gt * np.arctan(Br_gt * alpha - Er_gt * (Br_gt * alpha - np.arctan(Br_gt * alpha))))
# Fry_ddm = Dr_ddm * np.sin(Cr_ddm * np.arctan(Br_ddm * alpha - Er_ddm * (Br_ddm * alpha - np.arctan(Br_ddm * alpha))))
# Fry_dpm = Dr_dpm * np.sin(Cr_dpm * np.arctan(Br_dpm * alpha - Er_dpm * (Br_dpm * alpha - np.arctan(Br_dpm * alpha))))
# ax[1].plot(alpha, Fry_gt, 'b', lw=3)
# ax[1].plot(alpha, Fry_ddm, '--g', lw=3)
# ax[1].plot(alpha, Fry_dpm, '--r', lw=3)
# ax[1].set_xlabel("Side Slip Angle (rad)")
# ax[1].set_ylabel("Rear Lateral Force (N)")
# ax[1].grid()
# handles, labels = ax[0].get_legend_handles_labels()
# # fig.suptitle("Comparison of Model Predictions vs. Time", fontweight="bold")
# fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.0), frameon=False)


# plt.show()
# plt.ylim([-510, 510])
# plt.xlim([-0.2, 0.2])
# plt.plot(alpha, Df *Cf *Bf * alpha, label="Linear Model")

plt.figure(figsize=(12,8))
Ffy_dpm = Df_dpm * np.sin(Cf_dpm * np.arctan(Bf_dpm * alpha - Ef_dpm * (Bf_dpm * alpha - np.arctan(Bf_dpm * alpha))))
plt.plot(alpha, Ffy_dpm)
plt.xlabel("Side Slip Angle (rad)")
plt.ylabel("Front Lateral Force (N)")
plt.grid()
plt.title("B = {0}, C={1}, D={2}, E={3}".format(Bf_dpm, Cf_dpm, Df_dpm, Ef_dpm))
plt.show()