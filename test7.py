import numpy as np

file1 = np.loadtxt('saved_sim_5.txt', skiprows=1, dtype=int)
fx = file1[:, 0]
fy = file1[:, 1]
tx = file1[:, 2]
ty = file1[:, 3]
result = np.zeros(20)

for i in range (100):
    distance = abs(fx[i] - tx[i]) + abs(fy[i] - ty[i])
    result[distance] += 1
result = np.around(result, 2)
print(result)

prob_sensor = np.array([0.329, 0.288, 0.11, 0.065, 0.033, 0.02, 0.014, 0.011, 0.01, 0.01,
                        0.009, 0.01, 0.009, 0.009, 0.009, 0.008, 0.008, 0.008, 0.007, 0.006,
                        0.005, 0.005, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001,
                        0.001, 0, 0, 0, 0, 0, 0, 0])