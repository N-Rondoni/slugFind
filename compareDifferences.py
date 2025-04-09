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


plt.figure(1)
plt.scatter(mpcCors, stmCors)
x_vals = np.linspace(min(min(mpcCors), min(stmCors)), max(max(mpcCors), max(stmCors)), 100)
plt.plot(x_vals, x_vals, color='red', linestyle='--', label="y = x")
plt.xlabel(r"MPC score", fontsize = 14)
plt.ylabel(r"STM score", fontsize = 14)


plt.figure(2)
plt.scatter(mpcCors, oasisCors)
x_vals = np.linspace(min(min(mpcCors), min(oasisCors)), max(max(mpcCors), max(oasisCors)), 100)
plt.plot(x_vals, x_vals, color='red', linestyle='--', label="y = x")
plt.xlabel(r"MPC score", fontsize = 14)
plt.ylabel(r"Oasis score", fontsize = 14)


plt.show()
