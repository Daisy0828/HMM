import numpy as np

file1 = np.loadtxt('saved_sim_2.txt', skiprows=1, dtype=int)
sx = file1[:,2]
sy = file1[:,3]
loc = 0
loc_1 = 0
sum = 0
sum1 = 0
result = np.zeros((9,9))
loc1 = loc2 = loc3 = loc4 = loc5 = loc6 = loc7 = loc8 = loc9 = 0

for i in range(98):
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

    if np.logical_and(fx - tx == -1, fy - ty == -1):
        loc = 1
    elif np.logical_and(fx - tx == -1, fy - ty == 0):
        loc = 2
    elif np.logical_and(fx - tx == -1, fy - ty == 1):
        loc = 3
    elif np.logical_and(fx - tx == 0, fy - ty == -1):
        loc = 4
    elif np.logical_and(fx - tx == 0, fy - ty == 0):
        loc = 5
    elif np.logical_and(fx - tx == 0, fy - ty == 1):
        loc = 6
    elif np.logical_and(fx - tx == 1, fy - ty == -1):
        loc = 7
    elif np.logical_and(fx - tx == 1, fy - ty == 0):
        loc = 8
    elif np.logical_and(fx - tx == 1, fy - ty == 1):
        loc = 9

    tx = sx[i + 1]
    ty = sy[i + 1]
    fx = sx[i + 2]
    fy = sy[i + 2]

    if np.logical_and(fx - tx == -1, fy - ty == -1):
        loc_1 = 1
    elif np.logical_and(fx - tx == -1, fy - ty == 0):
        loc_1 = 2
    elif np.logical_and(fx - tx == -1, fy - ty == 1):
        loc_1 = 3
    elif np.logical_and(fx - tx == 0, fy - ty == -1):
        loc_1 = 4
    elif np.logical_and(fx - tx == 0, fy - ty == 0):
        loc_1 = 5
    elif np.logical_and(fx - tx == 0, fy - ty == 1):
        loc_1 = 6
    elif np.logical_and(fx - tx == 1, fy - ty == -1):
        loc_1 = 7
    elif np.logical_and(fx - tx == 1, fy - ty == 0):
        loc_1 = 8
    elif np.logical_and(fx - tx == 1, fy - ty == 1):
        loc_1 = 9

    result[loc -1][loc_1 - 1] = result[loc -1][loc_1 - 1] + 1
    if (loc_1 == loc and loc != 5):
        sum = sum + 1
    elif(loc_1 == loc and loc == 5):
        sum1 = sum1 + 1

print(sum)
print(sum1)
print(result)
