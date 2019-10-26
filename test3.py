import numpy as np
arr = np.array([])
arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#arr = np.append(arr, arr1)
#print(arr)
#result = np.where(arr1 == np.max(arr1))
#result = list(zip(result[0], result[1]))
#print(result)
arr2 = np.sum(arr1, axis=1)
arr2 = np.reshape(arr2, (3,1))
arr1 = np.true_divide(arr1,arr2)
print(arr2)
print(arr1)

prob = np.zeros((39,39))
switcher = {
    0: 0.001/152,
    1: 0.003/144,
    2: 0.005/136,
    3: 0.006/128,
    4: 0.007/120,
    5: 0.009/112,
    6: 0.01/104,
    7: 0.01/96,
    8: 0.012/88,
    9: 0.012/80,
    10: 0.013/72,
    11: 0.013/64,
    12: 0.013/56,
    13: 0.013/48,
    14: 0.015/40,
    15: 0.021/32,
    16: 0.041/24,
    17: 0.102/16,
}
for i in range(18):
    num = switcher.get(i)
    prob[i,i:39-i] = prob[38-i,i:39-i] = [num] * (39 - 2 * i)
    prob[i:39-i,i] = prob[i:39-i,38-i] = [num] * (39 - 2 * i)
prob[18, 18] = prob[20, 20] = prob[18, 20] = prob[20, 18] = 0.018
prob[18, 19] = prob[20, 19] = prob[19, 20] = prob[19, 18] = 0.072
prob[19,19] = 0.334
tx = 1
ty = 2
normalize = prob[19 - tx : 38- tx, 19 - ty : 38- ty]
s = np.sum(normalize)
print(s)
print(normalize[0,0]/s)

x_0_y_0 = np.zeros((9,9))
x_0_y_0[0][4] = 0.294
x_0_y_0[0][5] = 0.23
x_0_y_0[0][7] = 0.207
x_0_y_0[0][8] = 0.269
x_0_y_0[1][4] = 0.333
x_0_y_0[1][5] = 0.167
x_0_y_0[1][7] = 0.301
x_0_y_0[1][8] = 0.199
x_0_y_0[3][4] = 0.32
x_0_y_0[3][5] = 0.32
x_0_y_0[3][7] = 0.169
x_0_y_0[3][8] = 0.191
x_0_y_0[4][4] = 0.228
x_0_y_0[4][5] = 0.246
x_0_y_0[4][7] = 0.333
x_0_y_0[4][8] = 0.193

x_0_y_19 = np.zeros((9,9))
x_0_y_19[1][3] = 0.202
x_0_y_19[1][4] = 0.297
x_0_y_19[1][6] = 0.181
x_0_y_19[1][7] = 0.32
x_0_y_19[2][3] = 0.175
x_0_y_19[2][4] = 0.325
x_0_y_19[2][6] = 0.343
x_0_y_19[2][7] = 0.157
x_0_y_19[4][3] = 0.28
x_0_y_19[4][4] = 0.2
x_0_y_19[4][6] = 0.22
x_0_y_19[4][7] = 0.3
x_0_y_19[5][3] = 0.299
x_0_y_19[5][4] = 0.316
x_0_y_19[5][6] = 0.196
x_0_y_19[5][7] = 0.189

x_19_y_0 = np.zeros((9,9))
x_19_y_0[3][1] = 0.181
x_19_y_0[3][2] = 0.192
x_19_y_0[3][4] = 0.302
x_19_y_0[3][5] = 0.325
x_19_y_0[4][1] = 0.265
x_19_y_0[4][2] = 0.201
x_19_y_0[4][4] = 0.258
x_19_y_0[4][5] = 0.276
x_19_y_0[6][1] = 0.19
x_19_y_0[6][2] = 0.297
x_19_y_0[6][4] = 0.306
x_19_y_0[6][5] = 0.207
x_19_y_0[7][1] = 0.323
x_19_y_0[7][2] = 0.222
x_19_y_0[7][4] = 0.269
x_19_y_0[7][5] = 0.186

x_19_y_19 = np.zeros((9,9))
x_19_y_19[4][0] = 0.213
x_19_y_19[4][1] = 0.294
x_19_y_19[4][3] = 0.302
x_19_y_19[4][4] = 0.191
x_19_y_19[5][0] = 0.151
x_19_y_19[5][1] = 0.189
x_19_y_19[5][3] = 0.319
x_19_y_19[5][4] = 0.341
x_19_y_19[7][0] = 0.188
x_19_y_19[7][1] = 0.331
x_19_y_19[7][3] = 0.193
x_19_y_19[7][4] = 0.288
x_19_y_19[8][0] = 0.295
x_19_y_19[8][1] = 0.234
x_19_y_19[8][3] = 0.227
x_19_y_19[8][4] = 0.244

x_0 = np.zeros((9,9))
x_0[0] = [0, 0, 0, 0.297, 0.095, 0.083, 0.434, 0.043, 0.048]
x_0[1] = [0, 0, 0, 0.090, 0.298, 0.093, 0.044, 0.428, 0.047]
x_0[2] = [0, 0, 0, 0.095, 0.095, 0.300, 0.042, 0.043, 0.425]
x_0[3] = [0, 0, 0, 0.444, 0.093, 0.087, 0.288, 0.045, 0.043]
x_0[4] = [0, 0, 0, 0.136, 0.148, 0.132, 0.069, 0.448, 0.067]
x_0[5] = [0, 0, 0, 0.087, 0.089, 0.450, 0.049, 0.044, 0.281]

x_19 = np.zeros((9,9))
x_19[3] = [0.277, 0.043, 0.044, 0.456, 0.094, 0.086, 0, 0, 0]
x_19[4] = [0.069, 0.445, 0.068, 0.146, 0.127, 0.145, 0, 0, 0]
x_19[5] = [0.046, 0.040, 0.289, 0.085, 0.088, 0.452, 0, 0, 0]
x_19[6] = [0.436, 0.043, 0.044, 0.295, 0.093, 0.089, 0, 0, 0]
x_19[7] = [0.042, 0.439, 0.041, 0.091, 0.295, 0.092, 0, 0, 0]
x_19[8] = [0.040, 0.043, 0.431, 0.091, 0.092, 0.303, 0, 0, 0]

y_0 = np.zeros((9,9))
y_0[0] = [0, 0.302, 0.432, 0, 0.085, 0.041, 0, 0.090, 0.050]
y_0[1] = [0, 0.452, 0.282, 0, 0.088, 0.047, 0, 0.087, 0.044]
y_0[3] = [0, 0.088, 0.048, 0, 0.296, 0.437, 0, 0.086, 0.045]
y_0[4] = [0, 0.138, 0.069, 0, 0.141, 0.450, 0, 0.133, 0.069]
y_0[6] = [0, 0.093, 0.045, 0, 0.087, 0.045, 0, 0.304, 0.426]
y_0[7] = [0, 0.094, 0.047, 0, 0.085, 0.045, 0, 0.447, 0.282]

y_19 = np.zeros((9,9))
y_19[1] = [0.292, 0.450, 0, 0.049, 0.089, 0, 0.042, 0.078, 0]
y_19[2] = [0.437, 0.306, 0, 0.042, 0.086, 0, 0.044, 0.085, 0]
y_19[4] = [0.070, 0.137, 0, 0.446, 0.140, 0, 0.069, 0.138, 0]
y_19[5] = [0.047, 0.088, 0, 0.428, 0.301, 0, 0.045, 0.091, 0]
y_19[7] = [0.043, 0.082, 0, 0.043, 0.093, 0, 0.294, 0.446, 0]
y_19[8] = [0.042, 0.088, 0, 0.046, 0.085, 0, 0.439, 0.298, 0]

prob_sensor = np.zeros((39, 39))
switcher = {
            0: 0.001 / 152,
            1: 0.003 / 144,
            2: 0.005 / 136,
            3: 0.006 / 128,
            4: 0.007 / 120,
            5: 0.009 / 112,
            6: 0.01 / 104,
            7: 0.01 / 96,
            8: 0.012 / 88,
            9: 0.012 / 80,
            10: 0.013 / 72,
            11: 0.013 / 64,
            12: 0.013 / 56,
            13: 0.013 / 48,
            14: 0.015 / 40,
            15: 0.021 / 32,
            16: 0.041 / 24,
            17: 0.102 / 16,
        }
for i in range(18):
    num = switcher.get(i)
    prob_sensor[i, i:39 - i] = prob_sensor[38 - i, i:39 - i] = [num] * (39 - 2 * i)
    prob_sensor[i:39 - i, i] = prob_sensor[i:39 - i, 38 - i] = [num] * (39 - 2 * i)
prob_sensor[18, 18] = prob_sensor[20, 20] = prob_sensor[18, 20] = prob_sensor[20, 18] = 0.018
prob_sensor[18, 19] = prob_sensor[20, 19] = prob_sensor[19, 20] = prob_sensor[19, 18] = 0.072
prob_sensor[19, 19] = 0.334
print(np.sum(prob_sensor))
