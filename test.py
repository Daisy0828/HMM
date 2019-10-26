import numpy as np

file1 = np.loadtxt('saved_sim_8.txt', skiprows=1, dtype=int)
print(file1.shape)
fx = file1[:,0]
fy = file1[:,1]
tx = file1[:,2]
ty = file1[:,3]

same = np.logical_and(fx - tx == 0, fy - ty == 0)
num_same = np.sum(same)
print("same:", num_same)

adj_x = np.logical_and(fx - tx >= -1, fx - tx <= 1)
adj_y = np.logical_and(fy - ty >= -1, fy - ty <= 1)
adjacent = np.logical_and(adj_x, adj_y)
adj_num = np.sum(adjacent)
print("adj:",adj_num)

loc1 = np.logical_and(fx - tx == -1, fy - ty == -1)
num_loc1 = np.sum(adjacent)
#print("loc1:",num_loc1)
loc2 = np.logical_and(fx - tx == -1, fy - ty == 0)
num_loc2 = np.sum(adjacent)
#print("loc2:",num_loc2)
loc3 = np.logical_and(fx - tx == -1, fy - ty == 1)
num_loc3 = np.sum(adjacent)
#print("loc3:",num_loc3)
loc4 = np.logical_and(fx - tx == 0, fy - ty == -1)
num_loc4 = np.sum(adjacent)
#print("loc4:",num_loc4)
loc6 = np.logical_and(fx - tx == 0, fy - ty == 1)
num_loc6 = np.sum(adjacent)
#print("loc6:",num_loc6)
loc7 = np.logical_and(fx - tx == 1, fy - ty == -1)
num_loc7 = np.sum(adjacent)
#print("loc7:",num_loc7)
loc8 = np.logical_and(fx - tx == 1, fy - ty == 0)
num_loc8 = np.sum(adjacent)
#print("loc8:",num_loc8)
loc9 = np.logical_and(fx - tx == 1, fy - ty == 1)
num_loc9 = np.sum(adjacent)
#print("loc9:",num_loc9)


layer2_x = np.logical_and(np.absolute(fx - tx) == 2, np.absolute(fy - ty) <= 2)
layer2_y = np.logical_and(np.absolute(fx - tx) <= 2, np.absolute(fy - ty) == 2)
layer2 = np.logical_or(layer2_x, layer2_y)
num_layer2 = np.sum(layer2)
print("layer2:",num_layer2)

layer3_x = np.logical_and(np.absolute(fx - tx) == 3, np.absolute(fy - ty) <= 3)
layer3_y = np.logical_and(np.absolute(fx - tx) <= 3, np.absolute(fy - ty) == 3)
layer3 = np.logical_or(layer3_x, layer3_y)
num_layer3 = np.sum(layer3)
print("layer3:",num_layer3)

layer4_x = np.logical_and(np.absolute(fx - tx) == 4, np.absolute(fy - ty) <= 4)
layer4_y = np.logical_and(np.absolute(fx - tx) <= 4, np.absolute(fy - ty) == 4)
layer4 = np.logical_or(layer4_x, layer4_y)
num_layer4 = np.sum(layer4)
print("layer4:",num_layer4)

layer5_x = np.logical_and(np.absolute(fx - tx) == 5, np.absolute(fy - ty) <= 5)
layer5_y = np.logical_and(np.absolute(fx - tx) <= 5, np.absolute(fy - ty) == 5)
layer5 = np.logical_or(layer5_x, layer5_y)
num_layer5 = np.sum(layer5)
print("layer5:",num_layer5)

layer6_x = np.logical_and(np.absolute(fx - tx) == 6, np.absolute(fy - ty) <= 6)
layer6_y = np.logical_and(np.absolute(fx - tx) <= 6, np.absolute(fy - ty) == 6)
layer6 = np.logical_or(layer6_x, layer6_y)
num_layer6 = np.sum(layer6)
print("layer6:",num_layer6)

layer7 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 7, np.absolute(fy - ty) <= 7), np.logical_and(np.absolute(fx - tx) <= 7, np.absolute(fy - ty) == 7))
print("layer7:",np.sum(layer7))
layer8 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 8, np.absolute(fy - ty) <= 8), np.logical_and(np.absolute(fx - tx) <= 8, np.absolute(fy - ty) == 8))
print("layer8:",np.sum(layer8))
layer9 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 9, np.absolute(fy - ty) <= 9), np.logical_and(np.absolute(fx - tx) <= 9, np.absolute(fy - ty) == 9))
print("layer9:",np.sum(layer9))
layer10 = np.logical_or(np.logical_and(np.absolute(fx - tx) == 10, np.absolute(fy - ty) <= 10), np.logical_and(np.absolute(fx - tx) <= 10, np.absolute(fy - ty) == 10))
print("layer10:",np.sum(layer10))

out_layer10 = np.logical_or(np.absolute(fx - tx) > 10, np.absolute(fy - ty) > 10)
num_out_layer10 = np.sum(out_layer10)
print("out_layer10:",num_out_layer10)