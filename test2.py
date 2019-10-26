import numpy as np

file1 = np.loadtxt('saved_sim_2.txt', skiprows=1, dtype=int)
sx = file1[:,2]
sy = file1[:,3]
loc1 = loc2 = loc3 = loc4 = loc5 = loc6 = loc7 = loc8 = loc9 = 0

for i in range(99):
    tx = sx[i]
    ty = sy[i]
    fx = sx[i+1]
    fy = sy[i+1]

    if np.logical_and(fx - tx == -1, fy - ty == -1): loc1 = loc1 + 1
    elif np.logical_and(fx - tx == -1, fy - ty == 0): loc2 = loc2 + 1
    elif np.logical_and(fx - tx == -1, fy - ty == 1): loc3 = loc3 + 1
    elif np.logical_and(fx - tx == 0, fy - ty == -1): loc4 = loc4 + 1
    elif np.logical_and(fx - tx == 0, fy - ty == 0): loc5 = loc5 + 1
    elif np.logical_and(fx - tx == 0, fy - ty == 1): loc6 = loc6 + 1
    elif np.logical_and(fx - tx == 1, fy - ty == -1): loc7 = loc7 + 1
    elif np.logical_and(fx - tx == 1, fy - ty == 0): loc8 = loc8 + 1
    elif np.logical_and(fx - tx == 1, fy - ty == 1): loc9 = loc9 + 1

print("toloc1", loc1)
print("toloc2", loc2)
print("toloc3", loc3)
print("toloc4", loc4)
print("toloc5", loc5)
print("toloc6", loc6)
print("toloc7", loc7)
print("toloc8", loc8)
print("toloc9", loc9)
