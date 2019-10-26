# Implement part 2 here!
import numpy as np

class touchscreenHMM:

    # You may add instance variables, but you may not create a
    # custom initializer; touchscreenHMMs will be initialized
    # with no arguments.

    def __init__(self, width=20, height=20):
        """
        Feel free to initialize things in here!
        """
        self.width = width
        self.height = height
        # Write your code here!
        self.max = []
        self.obs_frame = np.array([])
        self.time = 0
        self.num_states = width * height
        self.prev = [1 / self.num_states] * self.num_states
        self.direction = 0

        x_0_y_0 = np.zeros((9, 9))
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

        x_0_y_19 = np.zeros((9, 9))
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

        x_19_y_0 = np.zeros((9, 9))
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

        x_19_y_19 = np.zeros((9, 9))
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

        x_0 = np.zeros((9, 9))
        x_0[0] = [0, 0, 0, 0.297, 0.095, 0.083, 0.434, 0.043, 0.048]
        x_0[1] = [0, 0, 0, 0.090, 0.298, 0.093, 0.044, 0.428, 0.047]
        x_0[2] = [0, 0, 0, 0.095, 0.095, 0.300, 0.042, 0.043, 0.425]
        x_0[3] = [0, 0, 0, 0.444, 0.093, 0.087, 0.288, 0.045, 0.043]
        x_0[4] = [0, 0, 0, 0.136, 0.148, 0.132, 0.069, 0.448, 0.067]
        x_0[5] = [0, 0, 0, 0.087, 0.089, 0.450, 0.049, 0.044, 0.281]

        x_19 = np.zeros((9, 9))
        x_19[3] = [0.277, 0.043, 0.044, 0.456, 0.094, 0.086, 0, 0, 0]
        x_19[4] = [0.069, 0.445, 0.068, 0.146, 0.127, 0.145, 0, 0, 0]
        x_19[5] = [0.046, 0.040, 0.289, 0.085, 0.088, 0.452, 0, 0, 0]
        x_19[6] = [0.436, 0.043, 0.044, 0.295, 0.093, 0.089, 0, 0, 0]
        x_19[7] = [0.042, 0.439, 0.041, 0.091, 0.295, 0.092, 0, 0, 0]
        x_19[8] = [0.040, 0.043, 0.431, 0.091, 0.092, 0.303, 0, 0, 0]

        y_0 = np.zeros((9, 9))
        y_0[0] = [0, 0.302, 0.432, 0, 0.085, 0.041, 0, 0.090, 0.050]
        y_0[1] = [0, 0.452, 0.282, 0, 0.088, 0.047, 0, 0.087, 0.044]
        y_0[3] = [0, 0.088, 0.048, 0, 0.296, 0.437, 0, 0.086, 0.045]
        y_0[4] = [0, 0.138, 0.069, 0, 0.141, 0.450, 0, 0.133, 0.069]
        y_0[6] = [0, 0.093, 0.045, 0, 0.087, 0.045, 0, 0.304, 0.426]
        y_0[7] = [0, 0.094, 0.047, 0, 0.085, 0.045, 0, 0.447, 0.282]

        y_19 = np.zeros((9, 9))
        y_19[1] = [0.292, 0.450, 0, 0.049, 0.089, 0, 0.042, 0.078, 0]
        y_19[2] = [0.437, 0.306, 0, 0.042, 0.086, 0, 0.044, 0.085, 0]
        y_19[4] = [0.070, 0.137, 0, 0.446, 0.140, 0, 0.069, 0.138, 0]
        y_19[5] = [0.047, 0.088, 0, 0.428, 0.301, 0, 0.045, 0.091, 0]
        y_19[7] = [0.043, 0.082, 0, 0.043, 0.093, 0, 0.294, 0.446, 0]
        y_19[8] = [0.042, 0.088, 0, 0.046, 0.085, 0, 0.439, 0.298, 0]

        self.x_0_y_0 = x_0_y_0
        self.x_0_y_19 = x_0_y_19
        self.x_19_y_0 = x_19_y_0
        self.x_19_y_19 = x_19_y_19
        self.x_0 = x_0
        self.x_19 = x_19
        self.y_0 = y_0
        self.y_19 = y_19

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
        self.prob_sensor = prob_sensor

        self.prob_trans = np.array([[0.58042988, 0.05570675, 0.06118581, 0.05418878, 0.04868278, 0.04650912,
                          0.06044928, 0.04717379, 0.04567379],
                         [0.05435038, 0.58687343, 0.05480302, 0.04447611, 0.06284158, 0.04424979,
                          0.04134116, 0.07015088, 0.04091366],
                         [0.06144233, 0.05428691, 0.57727677, 0.04728565, 0.05082256, 0.05499429,
                          0.04592531, 0.04603414, 0.06193206],
                         [0.05443377, 0.04463017, 0.04193063, 0.58844129, 0.06266611, 0.06936064,
                          0.05277058, 0.04349352, 0.0422733],
                         [0.08646994, 0.13325982, 0.08775366, 0.13428021, 0.12392818, 0.13156465,
                          0.08513685, 0.13098863, 0.08661806],
                         [0.04198292, 0.04422067, 0.05388142, 0.06851031, 0.06267378, 0.58890475,
                          0.04234196, 0.04402027, 0.05346392],
                         [0.06186122, 0.04753786, 0.04556851, 0.05518732, 0.05087481, 0.04751051,
                          0.57640795, 0.05484086, 0.06021098],
                         [0.04206244, 0.06798581, 0.04144882, 0.04426475, 0.06525394, 0.0441975,
                          0.05436007, 0.58677269, 0.05365399],
                         [0.04477693, 0.0468716, 0.06078165, 0.04622778, 0.05010881, 0.0547062,
                          0.06213275, 0.05509612, 0.57929815]])
        pass

    def sensor_model(self, observation, state):
        """
        Feel free to change the parameters of this function as you see fit.
        You may even delete this function! It is only here to point you
        in the right direction.

        This is the sensor model to get the probability of getting an observation from a state
        :param observation: A 2D numpy array filled with 0s, and a single 1 denoting
                            a touch location.
        :param state: A 2D numpy array filled with 0s, and a single 1 denoting
                        a touch location.
        :return: The probability of observing that observation from that given state, a number
        """
        # Write your code here!

        fx = np.nonzero(observation)[0][0]
        fy = np.nonzero(observation)[1][0]
        tx = np.nonzero(state)[0][0]
        ty = np.nonzero(state)[1][0]

        normalize = self.prob_sensor[19 - tx:39 - tx, 19 - ty:39 - ty]
        return normalize[fx,fy] / np.sum(normalize)



    def transition_model(self, cur_state, prev_state):
        """
        Feel free to change the parameters of this function as you see fit.
        You may even delete this function! It is only here to point you
        in the right direction.

        This will be your transition model to go from the old state to the new state
        :param old_state: A 2D numpy array filled with 0s, and a single 1 denoting
                            a touch location.
        :param new_state: A 2D numpy array filled with 0s, and a single 1 denoting
                            a touch location.
        :return: The probability of transitioning from the old state to the new state, a number
        """
        # Write your code here!
        fx = np.nonzero(cur_state)[0][0]
        fy = np.nonzero(cur_state)[1][0]
        tx = np.nonzero(prev_state)[0][0]
        ty = np.nonzero(prev_state)[1][0]
        x = fx - tx
        y = fy - ty
        cur_direction = 0

        if x >= 2 or x <= -2 or y >= 2 or y <= -2:
            return 0

        if np.logical_and(fx - tx == -1, fy - ty == -1):  # loc1
            cur_direction = 1
        elif np.logical_and(fx - tx == -1, fy - ty == 0):  # loc2
            cur_direction = 2
        elif np.logical_and(fx - tx == -1, fy - ty == 1):  # loc3
            cur_direction = 3
        elif np.logical_and(fx - tx == 0, fy - ty == -1):  # loc4
            cur_direction = 4
        elif np.logical_and(fx - tx == 0, fy - ty == 0):  # loc5
            cur_direction = 5
        elif np.logical_and(fx - tx == 0, fy - ty == 1):  # loc6
            cur_direction = 6
        elif np.logical_and(fx - tx == 1, fy - ty == -1):  # loc7
            cur_direction = 7
        elif np.logical_and(fx - tx == 1, fy - ty == 0):  # loc8
            cur_direction = 8
        elif np.logical_and(fx - tx == 1, fy - ty == 1):  # loc9
            cur_direction = 9

        if self.direction != 0:
            if tx == 0 and ty == 0:
                return self.x_0_y_0[self.direction-1][cur_direction-1]
            elif tx == 0 and ty == 19:
                return self.x_0_y_19[self.direction-1][cur_direction-1]
            elif tx == 19 and ty == 0:
                return self.x_19_y_0[self.direction-1][cur_direction-1]
            elif tx == 19 and ty == 19:
                return self.x_19_y_19[self.direction-1][cur_direction-1]
            elif tx == 0:
                return self.x_0[self.direction-1][cur_direction-1]
            elif tx == 19:
                return self.x_19[self.direction-1][cur_direction-1]
            elif ty == 0:
                return self.y_0[self.direction-1][cur_direction-1]
            elif ty == 19:
                return self.y_19[self.direction - 1][cur_direction - 1]
            else:
                return self.prob_trans[self.direction - 1][cur_direction - 1]

        else:
            if np.logical_and(fx - tx == -1, fy - ty == -1):  # loc1
                return 0.117
            elif np.logical_and(fx - tx == -1, fy - ty == 0):  # loc2
                return 0.117
            elif np.logical_and(fx - tx == -1, fy - ty == 1):  # loc3
                return 0.117
            elif np.logical_and(fx - tx == 0, fy - ty == -1):  # loc4
                return 0.117
            elif np.logical_and(fx - tx == 0, fy - ty == 0):  # loc5
                return 0.064
            elif np.logical_and(fx - tx == 0, fy - ty == 1):  # loc6
                return 0.117
            elif np.logical_and(fx - tx == 1, fy - ty == -1):  # loc7
                return 0.117
            elif np.logical_and(fx - tx == 1, fy - ty == 0):  # loc8
                return 0.117
            elif np.logical_and(fx - tx == 1, fy - ty == 1):  # loc9
                return 0.117
            else:
                return 0


    def filter_noisy_data(self, frame):
        """
        This is the function we will be calling during grading, passing in a noisy simualation. It should return the
        distribution where you think the actual position of the finger is in the same format that it is passed in as.

        DO NOT CHANGE THE FUNCTION PARAMETERS

        :param frame: A noisy frame to run your HMM on. This is a 2D numpy array filled with 0s, and a single 1 denoting
                    a touch location.
        :return: A 2D numpy array with the probabilities of the actual finger location.
        """
        # Write your code here!
        self.obs_frame = frame
        # add direction

        if self.time == 0:
            x = np.nonzero(frame)[0][0]
            y = np.nonzero(frame)[1][0]
            first = np.zeros([self.height, self.width])
            first[x][y] = 1
            first_loc = []
            first_loc.append(x)
            first_loc.append(y)
            self.max = first_loc
            self.prev = first
            result = first
        else:
            result = np.zeros(self.num_states)
            prev = np.reshape(self.prev, (1, -1))

            for i in range(self.num_states):
                cur_state = np.zeros(self.num_states)
                cur_state[i] = 1
                cur_state = np.reshape(cur_state,(self.height, self.width))
                transition_arr = np.zeros(self.num_states)
                for j in range(self.num_states):
                    prev_state = np.zeros(self.num_states)
                    prev_state[j] = 1
                    prev_state = np.reshape(prev_state, (self.height, self.width))
                    transition_arr[j] = self.transition_model(cur_state, prev_state)
                if np.sum(transition_arr) != 0:
                    transition_arr = transition_arr / np.sum(transition_arr)
                    result[i] = np.sum(prev * transition_arr) * self.sensor_model(self.obs_frame, cur_state)

            result = result / np.sum(result)
            result = np.reshape(result, (self.height, self.width))
            self.prev = result

            max_state = np.where(result == np.max(result))
            max_state = list(zip(max_state[0], max_state[1]))
            tx = self.max[0]
            ty = self.max[1]
            for ele in max_state:
                fx = ele[0]
                fy = ele[1]
                if np.logical_or(np.logical_and(np.absolute(fx - tx) == 1, np.absolute(fy - ty) <= 1),
                                 np.logical_and(np.absolute(fx - tx) <= 1, np.absolute(fy - ty) == 1)):
                    self.max = ele
                    break
            fx = self.max[0]
            fy = self.max[1]
            if np.logical_and(fx - tx == -1, fy - ty == -1):  # loc1
                self.direction = 1
            elif np.logical_and(fx - tx == -1, fy - ty == 0):  # loc2
                self.direction = 2
            elif np.logical_and(fx - tx == -1, fy - ty == 1):  # loc3
                self.direction = 3
            elif np.logical_and(fx - tx == 0, fy - ty == -1):  # loc4
                self.direction = 4
            elif np.logical_and(fx - tx == 0, fy - ty == 0):  # loc5
                self.direction = 5
            elif np.logical_and(fx - tx == 0, fy - ty == 1):  # loc6
                self.direction = 6
            elif np.logical_and(fx - tx == 1, fy - ty == -1):  # loc7
                self.direction = 7
            elif np.logical_and(fx - tx == 1, fy - ty == 0):  # loc8
                self.direction = 8
            elif np.logical_and(fx - tx == 1, fy - ty == 1):  # loc9
                self.direction = 9


        self.time = self.time + 1


        return result


