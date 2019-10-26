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
        self.old_frame = np.array([])
        self.obs_frame = np.array([])
        self.time = 0
        self.num_states = width * height
        self.prev = [1 / self.num_states] * self.num_states

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

        if np.logical_and(fx - tx == -1, fy - ty == -1):#loc1
            return 0.018
        elif np.logical_and(fx - tx == -1, fy - ty == 0):#loc2
            return 0.072
        elif np.logical_and(fx - tx == -1, fy - ty == 1):#loc3
            return 0.018
        elif np.logical_and(fx - tx == 0, fy - ty == -1):#loc4
            return 0.072
        elif np.logical_and(fx - tx == 0, fy - ty == 0):#loc5
            return 0.334
        elif np.logical_and(fx - tx == 0, fy - ty == 1):#loc6
            return 0.072
        elif np.logical_and(fx - tx == 1, fy - ty == -1):#loc7
            return 0.018
        elif np.logical_and(fx - tx == 1, fy - ty == 0):#loc8
            return 0.072
        elif np.logical_and(fx - tx == 1, fy - ty == 1):#loc9
            return 0.018
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 2, np.absolute(fy - ty) <= 2), np.logical_and(np.absolute(fx - tx) <= 2, np.absolute(fy - ty) == 2)):
            return 0.102
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 3, np.absolute(fy - ty) <= 3), np.logical_and(np.absolute(fx - tx) <= 3, np.absolute(fy - ty) == 3)):
            return 0.041
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 4, np.absolute(fy - ty) <= 4), np.logical_and(np.absolute(fx - tx) <= 4, np.absolute(fy - ty) == 4)):
            return 0.021
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 5, np.absolute(fy - ty) <= 5), np.logical_and(np.absolute(fx - tx) <= 5, np.absolute(fy - ty) == 5)):
            return 0.015
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 6, np.absolute(fy - ty) <= 6), np.logical_and(np.absolute(fx - tx) <= 6, np.absolute(fy - ty) == 6)):
            return 0.013
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 7, np.absolute(fy - ty) <= 7), np.logical_and(np.absolute(fx - tx) <= 7, np.absolute(fy - ty) == 7)):
            return 0.013
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 8, np.absolute(fy - ty) <= 8), np.logical_and(np.absolute(fx - tx) <= 8, np.absolute(fy - ty) == 8)):
            return 0.013
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 9, np.absolute(fy - ty) <= 9), np.logical_and(np.absolute(fx - tx) <= 9, np.absolute(fy - ty) == 9)):
            return 0.013
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 10, np.absolute(fy - ty) <= 10), np.logical_and(np.absolute(fx - tx) <= 10, np.absolute(fy - ty) == 10)):
            return 0.012
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 11, np.absolute(fy - ty) <= 11), np.logical_and(np.absolute(fx - tx) <= 11, np.absolute(fy - ty) == 11)):
            return 0.012
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 12, np.absolute(fy - ty) <= 12), np.logical_and(np.absolute(fx - tx) <= 12, np.absolute(fy - ty) == 12)):
            return 0.01
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 13, np.absolute(fy - ty) <= 13), np.logical_and(np.absolute(fx - tx) <= 13, np.absolute(fy - ty) == 13)):
            return 0.01
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 14, np.absolute(fy - ty) <= 14), np.logical_and(np.absolute(fx - tx) <= 14, np.absolute(fy - ty) == 14)):
            return 0.009
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 15, np.absolute(fy - ty) <= 15), np.logical_and(np.absolute(fx - tx) <= 15, np.absolute(fy - ty) == 15)):
            return 0.007
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 16, np.absolute(fy - ty) <= 16), np.logical_and(np.absolute(fx - tx) <= 16, np.absolute(fy - ty) == 16)):
            return 0.006
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 17, np.absolute(fy - ty) <= 17), np.logical_and(np.absolute(fx - tx) <= 17, np.absolute(fy - ty) == 17)):
            return 0.005
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 18, np.absolute(fy - ty) <= 18), np.logical_and(np.absolute(fx - tx) <= 18, np.absolute(fy - ty) == 18)):
            return 0.003
        elif np.logical_or(np.logical_and(np.absolute(fx - tx) == 19, np.absolute(fy - ty) <= 19), np.logical_and(np.absolute(fx - tx) <= 19, np.absolute(fy - ty) == 19)):
            return 0.001



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
        self.old_frame = self.obs_frame
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
            #self.max = first_loc
            self.prev = first
            result = first
        else:
            result = np.array([])
            prev = np.reshape(self.prev, (1, -1))

            for i in range(self.num_states):
                cur_state = np.zeros(self.num_states)
                cur_state[i] = 1
                cur_state = np.reshape(cur_state, (self.height, self.width))
                transition_arr = self.transition_model(cur_state)
                # print(self.sensor_model(self.obs_frame, cur_state))
                result = np.append(result,
                                   np.sum(prev * transition_arr) * self.sensor_model(self.obs_frame, cur_state))

            result = result / np.sum(result)
            result = np.reshape(result, (self.height, self.width))
            self.prev = result
        self.time = self.time + 1
        return result


