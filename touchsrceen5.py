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


        prob = np.zeros((39, 39))
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
            prob[i, i:39 - i] = prob[38 - i, i:39 - i] = [num] * (39 - 2 * i)
            prob[i:39 - i, i] = prob[i:39 - i, 38 - i] = [num] * (39 - 2 * i)
        prob[18, 18] = prob[20, 20] = prob[18, 20] = prob[20, 18] = 0.018
        prob[18, 19] = prob[20, 19] = prob[19, 20] = prob[19, 18] = 0.072
        prob[19, 19] = 0.334

        normalize = prob[19 - tx : 39- tx, 19 - ty : 39- ty]
        return normalize[fx,fy] / np.sum(normalize)



    def transition_model(self, cur_state):
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
        transition_arr = np.zeros(self.num_states)
        transition_arr = np.reshape(np.array(transition_arr), (self.height, self.width))
        tx = np.nonzero(cur_state)[0][0]
        ty = np.nonzero(cur_state)[1][0]



        prob = np.array([[0.58042988, 0.05570675, 0.06118581, 0.05418878, 0.04868278, 0.04650912,
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

        transition_arr[tx - 0][ty - 0] = prob[self.direction - 1][4]
        if self.direction != 0:
            if tx == 0:
                transition_arr[tx + 1][ty - 0] = prob[self.direction - 1][1]
                if ty != 0 and ty != 19:
                    transition_arr[tx + 1][ty - 1] = prob[self.direction - 1][2]
                    transition_arr[tx + 1][ty + 1] = prob[self.direction - 1][0]
                elif ty == 0:
                    transition_arr[tx + 1][ty + 1] = prob[self.direction - 1][0]
                else:
                    transition_arr[tx + 1][ty - 1] = prob[self.direction - 1][2]
            elif tx == 19:
                transition_arr[tx - 1][ty - 0] = prob[self.direction - 1][7]
                if ty != 0 and ty != 19:
                    transition_arr[tx - 1][ty - 1] = prob[self.direction - 1][8]
                    transition_arr[tx - 1][ty + 1] = prob[self.direction - 1][6]
                elif ty == 0:
                    transition_arr[tx - 1][ty + 1] = prob[self.direction - 1][6]
                else:
                    transition_arr[tx - 1][ty - 1] = prob[self.direction - 1][8]
            else:
                transition_arr[tx + 1][ty - 0] = prob[self.direction - 1][1]
                transition_arr[tx - 1][ty - 0] = prob[self.direction - 1][7]
                if ty != 0 and ty != 19:
                    transition_arr[tx - 1][ty - 1] = prob[self.direction - 1][8]
                    transition_arr[tx - 1][ty + 1] = prob[self.direction - 1][6]
                    transition_arr[tx + 1][ty - 1] = prob[self.direction - 1][2]
                    transition_arr[tx + 1][ty + 1] = prob[self.direction - 1][0]
                elif ty == 0:
                    transition_arr[tx - 1][ty + 1] = prob[self.direction - 1][6]
                    transition_arr[tx + 1][ty + 1] = prob[self.direction - 1][0]
                else:
                    transition_arr[tx - 1][ty - 1] = prob[self.direction - 1][8]
                    transition_arr[tx + 1][ty - 1] = prob[self.direction - 1][2]
            if ty == 0:
                transition_arr[tx - 0][ty + 1] = prob[self.direction - 1][3]
            elif ty == 19:
                transition_arr[tx - 0][ty - 1] = prob[self.direction - 1][5]
            else:
                transition_arr[tx - 0][ty + 1] = prob[self.direction - 1][3]
                transition_arr[tx - 0][ty - 1] = prob[self.direction - 1][5]
        else:
            transition_arr[tx - 0][ty - 0] = 0.064
            if tx == 0:
                transition_arr[tx + 1][ty - 0] = 0.117
                if ty != 0 and ty != 19:
                    transition_arr[tx + 1][ty - 1] = 0.117
                    transition_arr[tx + 1][ty + 1] = 0.117
                elif ty == 0:
                    transition_arr[tx + 1][ty + 1] = 0.117
                else:
                    transition_arr[tx + 1][ty - 1] = 0.117
            elif tx == 19:
                transition_arr[tx - 1][ty - 0] = 0.117
                if ty != 0 and ty != 19:
                    transition_arr[tx - 1][ty - 1] = 0.117
                    transition_arr[tx - 1][ty + 1] = 0.117
                elif ty == 0:
                    transition_arr[tx - 1][ty + 1] = 0.117
                else:
                    transition_arr[tx - 1][ty - 1] = 0.117
            else:
                transition_arr[tx + 1][ty - 0] = 0.117
                transition_arr[tx - 1][ty - 0] = 0.117
                if ty != 0 and ty != 19:
                    transition_arr[tx - 1][ty - 1] = 0.117
                    transition_arr[tx - 1][ty + 1] = 0.117
                    transition_arr[tx + 1][ty - 1] = 0.117
                    transition_arr[tx + 1][ty + 1] = 0.117
                elif ty == 0:
                    transition_arr[tx - 1][ty + 1] = 0.117
                    transition_arr[tx + 1][ty + 1] = 0.117
                else:
                    transition_arr[tx - 1][ty - 1] = 0.117
                    transition_arr[tx + 1][ty - 1] = 0.117
            if ty == 0:
                transition_arr[tx - 0][ty + 1] = 0.117
            elif ty == 19:
                transition_arr[tx - 0][ty - 1] = 0.117
            else:
                transition_arr[tx - 0][ty + 1] = 0.117
                transition_arr[tx - 0][ty - 1] = 0.117
        transition_arr = np.reshape(transition_arr, (1, -1))
        transition_arr = transition_arr / np.sum(transition_arr)
        #print(transition_arr)
        return transition_arr

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
            result = np.array([])
            prev = np.reshape(self.prev, (1, -1))

            for i in range(self.num_states):
                cur_state = np.zeros(self.num_states)
                cur_state[i] = 1
                cur_state = np.reshape(cur_state,(self.height, self.width))
                transition_arr = self.transition_model(cur_state)
                #print(self.sensor_model(self.obs_frame, cur_state))
                result = np.append(result,
                                   np.sum(prev * transition_arr) * self.sensor_model(self.obs_frame, cur_state))

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


