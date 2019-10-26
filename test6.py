import numpy as np

file1 = np.loadtxt('saved_sim_5.txt', skiprows=1, dtype=int)
sx = file1[:,2]
sy = file1[:,3]
loc = loc_1 = 0
loc1 = loc2 = loc3 = loc4 = loc5 = loc6 = loc7 = loc8 = loc9 = 0
x_0_y_0 = np.zeros((9,9))
x_0_y_19 = np.zeros((9,9))
x_19_y_0 = np.zeros((9,9))
x_19_y_19 = np.zeros((9,9))
x_0 = np.zeros((9,9))
x_19 = np.zeros((9,9))
y_0 = np.zeros((9,9))
y_19 = np.zeros((9,9))

for i in range(1):
    tx = sx[i]
    ty = sy[i]
    fx = sx[i+1]
    fy = sy[i+1]

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

    if tx == 0 and ty == 0:
        x_0_y_0[loc -1][loc_1-1] += 1
    elif tx == 0 and ty == 19:
        x_0_y_19[loc -1][loc_1-1] += 1
    elif tx == 19 and ty == 0:
        x_19_y_0[loc -1][loc_1 - 1] += 1
    elif tx == 19 and ty == 19:
        x_19_y_19[loc -1][loc_1-1] += 1
    elif tx == 0:
        x_0[loc -1][loc_1-1] += 1
    elif tx == 19:
        x_19[loc -1][loc_1 - 1] += 1
    elif ty == 0:
        y_0[loc -1][loc_1 - 1] += 1
    elif ty == 19:
        y_19[loc -1][loc_1 - 1] += 1

print("x_0_y_0", x_0_y_0)
print("x_0_y_19", x_0_y_19)
print("x_19_y_0", x_19_y_0)
print("x_19_y_19", x_19_y_19)
print("x_0", x_0)
print("x_19", x_19)
print("y_0", y_0)
print("y_19", y_19)

arr1 = x_0_y_0
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("x_0_y_0\n", arr1)

arr1 = x_0_y_19
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("x_0_y_19\n", arr1)

arr1 = x_19_y_0
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("x_19_y_0\n", arr1)

arr1 = x_19_y_19
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("x_19_y_19\n", arr1)

arr1 = x_0
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("x_0\n", arr1)

arr1 = x_19
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("x_19\n", arr1)

arr1 = y_0
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("y_0\n", arr1)

arr1 = y_19
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
print("y_19\n", arr1)
