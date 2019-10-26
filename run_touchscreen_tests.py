from simulation_testing_manager import SimulationTestingManager
from touchscreen import touchscreenHMM
import numpy as np
np.set_printoptions(suppress=True)

# You can test with any size (smaller size will be easier to see and follow)
# However, we will be running all graded tests using 20x20 screens
student_solution = touchscreenHMM(20, 20)

testing_manager = SimulationTestingManager()

"""
SimulationTestingManager's run_simulation:

FORMAT: name of flag - (type) description. Default = value

width - (int) specify the width of the simluation. Default = 20
height - (int) specify the height of the simluation. Default = 20
frames - (int) specify number of frames to run for simulation. Default = 50
visualization - (bool) specify whether to visualize simulation. Default = True
evaluate - (bool) specify whether to evaluate student solution. Default = False
           If True, the student's touchscreenHMM will automatically be intialized within this testing manager
save (bool) - specify whether to save the current run. Default = True
file_name (string) - specify a filename to save the simulation run to. Default = 'saved_sim.txt'
                     Only used when save flag is true.
frame_length (float) - specify the number of seconds each frame is shown for (lower for faster simulation). Default = 0.5
"""

# Basic simulation run, same functionality as run_visual_simulation.py
#testing_manager.run_simulation(
#    width=20, height=20, frames=100, visualization=True, evaluate=False
#)

# This will evaluate your solution and print your score without visualization
testing_manager.run_simulation(frames=100, visualization=False, evaluate=True, save=False)

# This will visualize the simulation and display your distribution at the same time.
# Then it will evaluate your solution and print your score.
#testing_manager.run_simulation(
#    frames=100, visualization=True, evaluate=True, frame_length=1.0
#)

# Will save a simulation to saved_sim.txt. Specify the file_name flag to customize the file name.
# Be careful, running this with an existing file_name will overwrite the last saved run in that file.
"""
avg_same = 0
avg_adj = 0
avg_loc1 = 0
avg_loc2 = 0
avg_loc3 = 0
avg_loc4 = 0
avg_loc6 = 0
avg_loc7 = 0
avg_loc8 = 0
avg_loc9 = 0
avg_lay2 = 0
avg_lay3 = 0
avg_lay4 = 0
avg_lay5 = 0
avg_lay6 = 0
avg_lay7 = 0
avg_lay8 = 0
avg_lay9 = 0
avg_lay10 = 0
avg_lay11 = 0
avg_lay12 = 0
avg_lay13 = 0
avg_lay14 = 0
avg_lay15 = 0
avg_lay16 = 0
avg_lay17 = 0
avg_lay18 = 0
avg_lay19 = 0
avg_lay20 = 0


n = 1000
for i in range(n):
    testing_manager.run_simulation(frames=100, save=True, visualization=False, file_name='saved_sim.txt')
    file1 = np.loadtxt('saved_sim.txt', skiprows=1, dtype=int)
    fx = file1[:, 0]
    fy = file1[:, 1]
    tx = file1[:, 2]
    ty = file1[:, 3]

    same = np.logical_and(fx - tx == 0, fy - ty == 0)
    num_same = np.sum(same)
    avg_same = avg_same + num_same

    adj_x = np.logical_and(fx - tx >= -1, fx - tx <= 1)
    adj_y = np.logical_and(fy - ty >= -1, fy - ty <= 1)
    adjacent = np.logical_and(adj_x, adj_y)
    adj_num = np.sum(adjacent)
    avg_adj = avg_adj + adj_num

    loc1 = np.logical_and(fx - tx == -1, fy - ty == -1)
    num_loc1 = np.sum(loc1)
    avg_loc1 = avg_loc1 + num_loc1
    loc2 = np.logical_and(fx - tx == -1, fy - ty == 0)
    num_loc2 = np.sum(loc2)
    avg_loc2 = avg_loc2 + num_loc2
    loc3 = np.logical_and(fx - tx == -1, fy - ty == 1)
    num_loc3 = np.sum(loc3)
    avg_loc3 = avg_loc3 + num_loc3
    loc4 = np.logical_and(fx - tx == 0, fy - ty == -1)
    num_loc4 = np.sum(loc4)
    avg_loc4 = avg_loc4 + num_loc4
    loc6 = np.logical_and(fx - tx == 0, fy - ty == 1)
    num_loc6 = np.sum(loc6)
    avg_loc6 = avg_loc6 + num_loc6
    loc7 = np.logical_and(fx - tx == 1, fy - ty == -1)
    num_loc7 = np.sum(loc7)
    avg_loc7 = avg_loc7 + num_loc7
    loc8 = np.logical_and(fx - tx == 1, fy - ty == 0)
    num_loc8 = np.sum(loc8)
    avg_loc8 = avg_loc8 + num_loc8
    loc9 = np.logical_and(fx - tx == 1, fy - ty == 1)
    num_loc9 = np.sum(loc9)
    avg_loc9 = avg_loc9 + num_loc9

    layer2_x = np.logical_and(np.absolute(fx - tx) == 2, np.absolute(fy - ty) <= 2)
    layer2_y = np.logical_and(np.absolute(fx - tx) <= 2, np.absolute(fy - ty) == 2)
    layer2 = np.logical_or(layer2_x, layer2_y)
    num_layer2 = np.sum(layer2)
    avg_lay2 = avg_lay2 + num_layer2

    layer3_x = np.logical_and(np.absolute(fx - tx) == 3, np.absolute(fy - ty) <= 3)
    layer3_y = np.logical_and(np.absolute(fx - tx) <= 3, np.absolute(fy - ty) == 3)
    layer3 = np.logical_or(layer3_x, layer3_y)
    num_layer3 = np.sum(layer3)
    avg_lay3 = avg_lay3 + num_layer3

    layer4_x = np.logical_and(np.absolute(fx - tx) == 4, np.absolute(fy - ty) <= 4)
    layer4_y = np.logical_and(np.absolute(fx - tx) <= 4, np.absolute(fy - ty) == 4)
    layer4 = np.logical_or(layer4_x, layer4_y)
    num_layer4 = np.sum(layer4)
    avg_lay4 = avg_lay4 + num_layer4

    layer5_x = np.logical_and(np.absolute(fx - tx) == 5, np.absolute(fy - ty) <= 5)
    layer5_y = np.logical_and(np.absolute(fx - tx) <= 5, np.absolute(fy - ty) == 5)
    layer5 = np.logical_or(layer5_x, layer5_y)
    num_layer5 = np.sum(layer5)
    avg_lay5 = avg_lay5 + num_layer5

    layer6_x = np.logical_and(np.absolute(fx - tx) == 6, np.absolute(fy - ty) <= 6)
    layer6_y = np.logical_and(np.absolute(fx - tx) <= 6, np.absolute(fy - ty) == 6)
    layer6 = np.logical_or(layer6_x, layer6_y)
    num_layer6 = np.sum(layer6)
    avg_lay6 = avg_lay6 + num_layer6

    layer7 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 7, np.absolute(fy - ty) <= 7),
                           np.logical_and(np.absolute(fx - tx) <= 7, np.absolute(fy - ty) == 7))
    num_layer7 = np.sum(layer7)
    avg_lay7 = avg_lay7 + num_layer7
    layer8 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 8, np.absolute(fy - ty) <= 8),
                           np.logical_and(np.absolute(fx - tx) <= 8, np.absolute(fy - ty) == 8))
    num_layer8 = np.sum(layer8)
    avg_lay8 = avg_lay8 + num_layer8
    layer9 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 9, np.absolute(fy - ty) <= 9),
                           np.logical_and(np.absolute(fx - tx) <= 9, np.absolute(fy - ty) == 9))
    num_layer9 = np.sum(layer9)
    avg_lay9 = avg_lay9 + num_layer9
    layer10 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 10, np.absolute(fy - ty) <= 10),
                            np.logical_and(np.absolute(fx - tx) <= 10, np.absolute(fy - ty) == 10))
    num_layer10 = np.sum(layer10)
    avg_lay10 = avg_lay10 + num_layer10
    layer11 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 11, np.absolute(fy - ty) <= 11),
                            np.logical_and(np.absolute(fx - tx) <= 11, np.absolute(fy - ty) == 11))
    num_layer11 = np.sum(layer11)
    avg_lay11 = avg_lay11 + num_layer11
    layer12 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 12, np.absolute(fy - ty) <= 12),
                            np.logical_and(np.absolute(fx - tx) <= 12, np.absolute(fy - ty) == 12))
    num_layer12 = np.sum(layer12)
    avg_lay12 = avg_lay12 + num_layer12
    layer13 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 13, np.absolute(fy - ty) <= 13),
                            np.logical_and(np.absolute(fx - tx) <= 13, np.absolute(fy - ty) == 13))
    num_layer13 = np.sum(layer13)
    avg_lay13 = avg_lay13 + num_layer13
    layer14 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 14, np.absolute(fy - ty) <= 14),
                            np.logical_and(np.absolute(fx - tx) <= 14, np.absolute(fy - ty) == 14))
    num_layer14 = np.sum(layer14)
    avg_lay14 = avg_lay14 + num_layer14
    layer15 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 15, np.absolute(fy - ty) <= 15),
                            np.logical_and(np.absolute(fx - tx) <= 15, np.absolute(fy - ty) == 15))
    num_layer15 = np.sum(layer15)
    avg_lay15 = avg_lay15 + num_layer15
    layer16 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 16, np.absolute(fy - ty) <= 16),
                            np.logical_and(np.absolute(fx - tx) <= 16, np.absolute(fy - ty) == 16))
    num_layer16 = np.sum(layer16)
    avg_lay16 = avg_lay16 + num_layer16
    layer17 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 17, np.absolute(fy - ty) <= 17),
                            np.logical_and(np.absolute(fx - tx) <= 17, np.absolute(fy - ty) == 17))
    num_layer17 = np.sum(layer17)
    avg_lay17 = avg_lay17 + num_layer17
    layer18 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 18, np.absolute(fy - ty) <= 18),
                            np.logical_and(np.absolute(fx - tx) <= 18, np.absolute(fy - ty) == 18))
    num_layer18 = np.sum(layer18)
    avg_lay18 = avg_lay18 + num_layer18
    layer19 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 19, np.absolute(fy - ty) <= 19),
                            np.logical_and(np.absolute(fx - tx) <= 19, np.absolute(fy - ty) == 19))
    num_layer19 = np.sum(layer19)
    avg_lay19 = avg_lay19 + num_layer19
    layer20 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 20, np.absolute(fy - ty) <= 20),
                            np.logical_and(np.absolute(fx - tx) <= 20, np.absolute(fy - ty) == 20))
    num_layer20 = np.sum(layer20)
    avg_lay20 = avg_lay20 + num_layer20
sum = avg_adj + avg_lay2 + avg_lay3 + avg_lay4 + avg_lay5 + avg_lay6 + avg_lay7 + avg_lay8 + avg_lay9 + avg_lay10 + avg_lay11 + avg_lay12 + avg_lay13 + avg_lay14 + avg_lay15 + avg_lay16 + avg_lay17 + avg_lay18 + avg_lay19 + avg_lay20
print(sum)
avg_same = avg_same / n
avg_adj = avg_adj / n
avg_loc1 = avg_loc1 / n
avg_loc2 = avg_loc2 / n
avg_loc3 = avg_loc3 / n
avg_loc4 = avg_loc4 / n
avg_loc6 = avg_loc6 / n
avg_loc7 = avg_loc7 / n
avg_loc8 = avg_loc8 / n
avg_loc9 = avg_loc9 / n
avg_lay2 = avg_lay2 / n
avg_lay3 = avg_lay3 / n
avg_lay4 = avg_lay4 / n
avg_lay5 = avg_lay5 / n
avg_lay6 = avg_lay6 / n
avg_lay7 = avg_lay7 / n
avg_lay8 = avg_lay8 / n
avg_lay9 = avg_lay9 / n
avg_lay10 = avg_lay10 / n
avg_lay11 = avg_lay11 / n
avg_lay12 = avg_lay12 / n
avg_lay13 = avg_lay13 / n
avg_lay14 = avg_lay14 / n
avg_lay15 = avg_lay15 / n
avg_lay16 = avg_lay16 / n
avg_lay17 = avg_lay17 / n
avg_lay18 = avg_lay18 / n
avg_lay19 = avg_lay19 / n
avg_lay20 = avg_lay20 / n


print("avg_same", avg_same)
print("avg_adj", avg_adj)
print("avg_loc1", avg_loc1)
print("avg_loc2", avg_loc2)
print("avg_loc3", avg_loc3)
print("avg_loc4", avg_loc4)
print("avg_loc6", avg_loc6)
print("avg_loc7", avg_loc7)
print("avg_loc8", avg_loc8)
print("avg_loc9", avg_loc9)
print("avg_lay2", avg_lay2)
print("avg_lay3", avg_lay3)
print("avg_lay4", avg_lay4)
print("avg_lay5", avg_lay5)
print("avg_lay6", avg_lay6)
print("avg_lay7", avg_lay7)
print("avg_lay8", avg_lay8)
print("avg_lay9", avg_lay9)
print("avg_lay10", avg_lay10)
print("avg_lay11", avg_lay11)
print("avg_lay12", avg_lay12)
print("avg_lay13", avg_lay13)
print("avg_lay14", avg_lay14)
print("avg_lay15", avg_lay15)
print("avg_lay16", avg_lay16)
print("avg_lay17", avg_lay17)
print("avg_lay18", avg_lay18)
print("avg_lay19", avg_lay19)
print("avg_lay20", avg_lay20)

n = 500
loc1 = loc2 = loc3 = loc4 = loc5 = loc6 = loc7 = loc8 = loc9 = 0
for i in range(n):
    testing_manager.run_simulation(frames=100, save=True, visualization=False, file_name='saved_sim.txt')
    file1 = np.loadtxt('saved_sim.txt', skiprows=1, dtype=int)
    sx = file1[:, 2]
    sy = file1[:, 3]


    for i in range(99):
        tx = sx[i]
        ty = sy[i]
        fx = sx[i + 1]
        fy = sy[i + 1]

        if np.logical_and(fx - tx == -1, fy - ty == -1):
            loc1 = loc1 + 1
        elif np.logical_and(fx - tx == -1, fy - ty == 0):
            loc2 = loc2 + 1
        elif np.logical_and(fx - tx == -1, fy - ty == 1):
            loc3 = loc3 + 1
        elif np.logical_and(fx - tx == 0, fy - ty == -1):
            loc4 = loc4 + 1
        elif np.logical_and(fx - tx == 0, fy - ty == 0):
            loc5 = loc5 + 1
        elif np.logical_and(fx - tx == 0, fy - ty == 1):
            loc6 = loc6 + 1
        elif np.logical_and(fx - tx == 1, fy - ty == -1):
            loc7 = loc7 + 1
        elif np.logical_and(fx - tx == 1, fy - ty == 0):
            loc8 = loc8 + 1
        elif np.logical_and(fx - tx == 1, fy - ty == 1):
            loc9 = loc9 + 1

print("toloc1", loc1/n)
print("toloc2", loc2/n)
print("toloc3", loc3/n)
print("toloc4", loc4/n)
print("toloc5", loc5/n)
print("toloc6", loc6/n)
print("toloc7", loc7/n)
print("toloc8", loc8/n)
print("toloc9", loc9/n)

n = 1000
sum = 0
sum1 = 0
result = np.zeros((9,9))
loc = 0
loc_1 = 0
loc1 = loc2 = loc3 = loc4 = loc5 = loc6 = loc7 = loc8 = loc9 = 0

for i in range(n):
    testing_manager.run_simulation(frames=100, save=True, visualization=False, file_name='saved_sim.txt')
    file1 = np.loadtxt('saved_sim.txt', skiprows=1, dtype=int)
    sx = file1[:, 2]
    sy = file1[:, 3]



    for i in range(98):
        tx = sx[i]
        ty = sy[i]
        fx = sx[i + 1]
        fy = sy[i + 1]



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
            loc1 = loc1 + 1
        elif np.logical_and(fx - tx == -1, fy - ty == 0):
            loc2 = loc2 + 1
        elif np.logical_and(fx - tx == -1, fy - ty == 1):
            loc3 = loc3 + 1
        elif np.logical_and(fx - tx == 0, fy - ty == -1):
            loc4 = loc4 + 1
        elif np.logical_and(fx - tx == 0, fy - ty == 0):
            loc5 = loc5 + 1
        elif np.logical_and(fx - tx == 0, fy - ty == 1):
            loc6 = loc6 + 1
        elif np.logical_and(fx - tx == 1, fy - ty == -1):
            loc7 = loc7 + 1
        elif np.logical_and(fx - tx == 1, fy - ty == 0):
            loc8 = loc8 + 1
        elif np.logical_and(fx - tx == 1, fy - ty == 1):
            loc9 = loc9 + 1

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

        result[loc - 1][loc_1 - 1] = result[loc - 1][loc_1 - 1] + 1
print(result / n)
arr1 = result
arr2 = np.sum(arr1, axis=1)
arr3 = np.sum(arr1, axis=0)
arr2 = np.reshape(arr2, (9,1))
arr1 = np.true_divide(arr1,arr2)
#print(arr1)
#print(arr2)
print("toloc1", loc1)
print("toloc2", loc2)
print("toloc3", loc3)
print("toloc4", loc4)
print("toloc5", loc5)
print("toloc6", loc6)
print("toloc7", loc7)
print("toloc8", loc8)
print("toloc9", loc9)
print(arr3)
print(arr1)

n = 10000
loc = 0
loc_1 = 0
loc1 = loc2 = loc3 = loc4 = loc5 = loc6 = loc7 = loc8 = loc9 = 0
x_0_y_0 = np.zeros((9,9))
x_0_y_19 = np.zeros((9,9))
x_19_y_0 = np.zeros((9,9))
x_19_y_19 = np.zeros((9,9))
x_0 = np.zeros((9,9))
x_19 = np.zeros((9,9))
y_0 = np.zeros((9,9))
y_19 = np.zeros((9,9))


for i in range(n):
    testing_manager.run_simulation(frames=100, save=True, visualization=False, file_name='saved_sim.txt')
    file1 = np.loadtxt('saved_sim.txt', skiprows=1, dtype=int)
    sx = file1[:, 2]
    sy = file1[:, 3]
    for i in range(98):
        tx = sx[i]
        ty = sy[i]
        fx = sx[i + 1]
        fy = sy[i + 1]

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
            x_0_y_0[loc - 1][loc_1 - 1] += 1
        elif tx == 0 and ty == 19:
            x_0_y_19[loc - 1][loc_1 - 1] += 1
        elif tx == 19 and ty == 0:
            x_19_y_0[loc - 1][loc_1 - 1] += 1
        elif tx == 19 and ty == 19:
            x_19_y_19[loc - 1][loc_1 - 1] += 1
        elif tx == 0:
            x_0[loc - 1][loc_1 - 1] += 1
        elif tx == 19:
            x_19[loc - 1][loc_1 - 1] += 1
        elif ty == 0:
            y_0[loc - 1][loc_1 - 1] += 1
        elif ty == 19:
            y_19[loc - 1][loc_1 - 1] += 1

print("x_0_y_0\n", x_0_y_0)
print(np.sum(x_0_y_0, axis=1))
print(np.sum(x_0_y_0, axis=0))
print("x_0_y_19\n", x_0_y_19)
print(np.sum(x_0_y_19, axis=1))
print(np.sum(x_0_y_19, axis=0))
print("x_19_y_0\n", x_19_y_0)
print(np.sum(x_19_y_0, axis=1))
print(np.sum(x_19_y_0, axis=0))
print("x_19_y_19\n", x_19_y_19)
print(np.sum(x_19_y_19, axis=1))
print(np.sum(x_19_y_19, axis=0))
print("x_0\n", x_0)
print(np.sum(x_0, axis=1))
print(np.sum(x_0, axis=0))
print("x_19\n", x_19)
print(np.sum(x_19, axis=1))
print(np.sum(x_19, axis=0))
print("y_0\n", y_0)
print(np.sum(y_0, axis=1))
print(np.sum(y_0, axis=0))
print("y_19\n", y_19)
print(np.sum(y_19, axis=1))
print(np.sum(y_19, axis=0))

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


n = 1000
result = np.zeros(40)
for i in range(n):
    testing_manager.run_simulation(frames=100, save=True, visualization=False, file_name='saved_sim.txt')
    file1 = np.loadtxt('saved_sim.txt', skiprows=1, dtype=int)
    fx = file1[:, 0]
    fy = file1[:, 1]
    tx = file1[:, 2]
    ty = file1[:, 3]


    for i in range(100):
        distance = abs(fx[i] - tx[i]) + abs(fy[i] - ty[i])
        result[distance] += 1
result = result / 100000
result = np.around(result,3)
print(result)
print(np.sum(result))





SimulationTestingManager's run_saved_simulation:

To be used when running a simulation that was previously saved to disk.

FORMAT: name of flag - (type) description. Default = value

visualization - (bool) specify whether to visualize simulation. Default = True
evaluate - (bool) specify whether to evaluate student solution. Default = False
           If True, a student_hmm must be passed in
student_hmm - (touchscreenHMM)  if evaluate flag is true, must pass in an instance of touchscreenHMM.
frame_length (float) - specify the number of seconds each frame is shown for (lower for faster simulation). Default = 0.5
"""

# Save a run to simple_text.txt
#testing_manager.run_simulation(
#    width=20, height=20, frames=10, save=True, file_name="simple_text.txt"
#)

# In order to visualize with the saved data in simple_text.txt:
#testing_manager.run_saved_simulation(file_name="simple_text.txt", frame_length=0.1)

# Test against the saved data without visualizing
#testing_manager.run_saved_simulation(
#    file_name="simple_text.txt",
#    visualization=False,
#    evaluate=True,
#    student_hmm=student_solution,
#)
