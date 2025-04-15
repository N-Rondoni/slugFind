import numpy as np
import matplotlib.pyplot as plt

mpcCors = np.load("data/allScores.npy")
#print(np.shape(mpcCors))

stmCors = np.load("data/allScoresSTM.npy")
#print(np.shape(stmCors))

oasisCors = np.load("data/allScoresOasis.npy")
#print(np.shape(oasisCors))

print("Mean oasis difference, mpc - oasis:")
print(np.mean(mpcCors - oasisCors))



print("Mean stm dif:, mpc - stm")
print(np.mean(mpcCors - stmCors))


print("standard deviations, std of [mpc, stm, oasis]:", np.std(mpcCors), np.std(stmCors), np.std(oasisCors))
print("means of [mpc, stm, oasis]:", np.mean(mpcCors), np.mean(stmCors), np.mean(oasisCors))
print("medians of [mpc, stm, oasis]:", np.median(mpcCors), np.median(stmCors), np.median(oasisCors))


# Try as one figure, subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(r"Comparisons with State of the Art", fontsize=20)

# Subplot 1: MPC vs STM
axs[0].scatter(mpcCors, stmCors, color='green', alpha=0.7)
x_vals1 = np.linspace(min(min(mpcCors), min(stmCors)), max(max(mpcCors), max(stmCors)), 100)
axs[0].plot(x_vals1, x_vals1, color='coral', linestyle='--', label='y = x')
axs[0].set_xlabel(r"MPC Correlation Coeff. ", fontsize=14)
axs[0].set_ylabel(r"STM Correlation Coeff.", fontsize=14)
axs[0].set_title(r"MPC vs STM", fontsize=16)
#axs[0].legend()
axs[0].grid(True)

# Subplot 2: MPC vs Oasis
axs[1].scatter(mpcCors, oasisCors, color='magenta', alpha=0.7)
x_vals2 = np.linspace(min(min(mpcCors), min(oasisCors)), max(max(mpcCors), max(oasisCors)), 100)
axs[1].plot(x_vals2, x_vals2, color='coral', linestyle='--', label='y = x')
axs[1].set_xlabel(r"MPC Correlation Coeff.", fontsize=14)
axs[1].set_ylabel(r"Oasis Correlation Coeff.", fontsize=14)
axs[1].set_title(r"MPC vs Oasis", fontsize=16)
#axs[1].legend()
axs[1].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit main title)


# load in VPDs for an alternative plot
#plt.figure(2)
allVPDs = np.load("data/allVPDs.npy")
allVPDsSTM = np.load("data/STM_allVPDs.npy")
allVPDsOasis = np.load("data/Oasis_allVPDs.npy")

viewWindow = 2500
index_allVPDs = (allVPDs <= viewWindow) #makes viewing reasonable
print("before viewing window", np.shape(allVPDs))
allVPDs = allVPDs[index_allVPDs]
allVPDsSTM = allVPDsSTM[index_allVPDs]
allVPDsOasis = allVPDsOasis[index_allVPDs]
print("after viewing window", np.shape(allVPDs))
print(allVPDsOasis)

# Try as one figure, subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(r"Victor-Purpura Comparisons with State of the Art", fontsize=20)


# Subplot 1: MPC vs STM, VPDs
axs[0].scatter(allVPDs, allVPDsSTM, color='green', alpha=0.7)
x_vals1 = np.linspace(min(min(allVPDs), min(allVPDsSTM)), max(max(allVPDs), max(allVPDsSTM)), 100)
x_vals2 = np.linspace(0, viewWindow, 100)
axs[0].plot(x_vals1, x_vals1, color='coral', linestyle='--', label='y = x')
axs[0].set_xlabel(r"MPC Victor-Purpura Distance", fontsize=14)
axs[0].set_ylabel(r"STM Victor-Purpura Distance", fontsize=14)
axs[0].set_title(r"MPC vs STM", fontsize=16)
#axs[0].legend()
axs[0].grid(True)

# Subplot 2: MPC vs Oasis, VPDs
axs[1].scatter(allVPDs, allVPDsOasis, color='magenta', alpha=0.7)
x_vals2 = np.linspace(min(min(allVPDs), min(allVPDsSTM)), max(max(allVPDs), max(allVPDsOasis)), 100)
x_vals2 = np.linspace(0, viewWindow, 100)
axs[1].plot(x_vals2, x_vals2, color='coral', linestyle='--', label='y = x')
axs[1].set_xlabel(r"MPC Victor-Purpura Distance", fontsize=14)
axs[1].set_ylabel(r"Oasis Victor-Purpura Distance", fontsize=14)
axs[1].set_title(r"MPC vs Oasis", fontsize=16)
#axs[1].legend()
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit main title
plt.show()



plt.show()
