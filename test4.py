from touchscreen import touchscreenHMM
import numpy as np

student_solution = touchscreenHMM(20, 20)
start_frame = np.zeros(400)
start_frame[0] = 1
start_frame = np.reshape(start_frame, (20,20))
student_solution.filter_noisy_data(start_frame)
print(student_solution.prev)
#print(student_solution.max)
#print(student_solution.direction)

start_frame = np.zeros(400)
start_frame[1] = 1
start_frame = np.reshape(start_frame, (20,20))
student_solution.filter_noisy_data(start_frame)
print(student_solution.prev)
#print(student_solution.max)
#print(student_solution.direction)

start_frame = np.zeros(400)
start_frame[1] = 1
start_frame = np.reshape(start_frame, (20,20))
student_solution.filter_noisy_data(start_frame)
print(student_solution.prev)
#print(student_solution.max)
#print(student_solution.direction)

start_frame = np.zeros(400)
start_frame[21] = 1
start_frame = np.reshape(start_frame, (20,20))
student_solution.filter_noisy_data(start_frame)
print(student_solution.prev)
#print(student_solution.max)
#print(student_solution.direction)
