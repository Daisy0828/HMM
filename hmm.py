# Implement your HMM for Part 1 here!
import numpy as np

class HMM:

    # You may add instance variables, but you may not change the
    # initializer; HMMs will be initialized with the given __init__
    # function when grading.
    observationList = []

    def __init__(self, sensor_model, transition_model, num_states):
        """
        Inputs:
        - sensor_model: the sensor model of the HMM.
          This is a function that takes in an observation E
          (represented as a string 'A', 'B', ...) and a state S
          (reprensented as a natural number 0, 1, ...) and
          outputs the probability of observing E in state S.

        - transition_model: the transition model of the HMM.
          This is a function that takes in two states, s and s',
          and outputs the probability of transitioning from
          state s to state s'.

        - num_states: this is the number of hidden states in the HMM, an integer
        """
        # Initialize your HMM here!
        self.sensor_model = sensor_model
        self.transition_model = transition_model
        self.num_states = num_states

    def tell(self, observation):
        """
        Takes in an observation and records it.
        You will need to keep track of the current timestep and increment
        it for each observation.

        Input:
        - observation: The observation at the current timestep, a string

        Output:
        - None
        """
        # Write your code here!
        self.observationList.append(observation)

    def ask(self, time):
        """
        Takes in a timestep that is greater than or equal to
        the current timestep and outputs a probability distribution
        (represented as a list) over states for that timestep.
        The index of the probability is the state it corresponds to.

        Input:
        - time: the timestep to get the observation distribution for, an integer

        Output:
        - a probability distribution over the hidden state for the given timestep, a list of numbers
        """
        # Write your code here!

        if time == 0:
            return [1 / self.num_states] * self.num_states
        elif time <= len(self.observationList):
            #print("time = obs")
            result = np.array([])
            prev = np.array(self.ask(time - 1))
            for i in range(self.num_states):
                transition_arr = np.array([])
                for j in range(len(prev)):
                    transition_arr = np.append(transition_arr, self.transition_model(j, i))
                result = np.append(result, np.sum(prev * transition_arr) * self.sensor_model(self.observationList[time-1], i))
            result = result / np.sum(result)
            return result.tolist()
        else:
            #print("time > obs")
            result = np.array([])
            prev = np.array(self.ask(time - 1))
            for i in range(self.num_states):
                transition_arr = np.array([])
                for j in range(self.num_states):
                    transition_arr = np.append(transition_arr, self.transition_model(j, i))
                result = np.append(result, np.sum(prev * transition_arr))
            return result.tolist()