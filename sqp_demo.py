from systems import *
from sqp import *
from constraints import *
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

eps = 1e-3
num_state = 4
num_input = 2
horizon = 80

ntimesteps = horizon
target_states = np.zeros((ntimesteps, 4))
noisy_targets = np.zeros((ntimesteps, 4))
xmax = np.zeros((ntimesteps, 4))
xmin = np.zeros((ntimesteps, 4))
noisy_targets[0, 2] = 1
target_states[0, 2] = 1
ref_vel = np.ones(ntimesteps)
# ref_vel = np.zeros(ntimesteps)
dt = 0.2
curv = 0.1
a = 1.5
v_max = 11

system = Car()
system.set_dt(dt)
system.set_cost(np.diag([50.0, 50.0, 10.0, 1.0]), np.diag([300.0, 1000.0]))
system.Q_f = system.Q*horizon/10  # *50
system.set_control_limit(np.array([[-1, 1], [-0.2, 0.2]]))
# system.control_limited = False
init_inputs = np.zeros((ntimesteps - 1, num_input))


# for i in range(40, ntimesteps):
#     if ref_vel[i - 1] > v_max:
#         a = 0
#     ref_vel[i] = ref_vel[i - 1] + a*dt

# for i in range(1, ntimesteps):
#     target_states[i, 0] = target_states[i-1, 0] + np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
#     target_states[i, 1] = target_states[i-1, 1] + np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
#     target_states[i, 2] = ref_vel[i]
#     target_states[i, 3] = target_states[i-1, 3] + curv*dt
#     noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 0.5)
#     noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 0.5)
#     noisy_targets[i, 2] = ref_vel[i]
#     noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.1)

# corner inputs case

for i in range(1, 40):
    target_states[i, 0] = target_states[i-1, 0] + dt*ref_vel[i - 1]
    target_states[i, 1] = target_states[i-1, 1]
    target_states[i, 2] = ref_vel[i]
    target_states[i, 3] = 0
    noisy_targets[i, 0] = target_states[i, 0]
    noisy_targets[i, 1] = target_states[i, 1]
    noisy_targets[i, 2] = target_states[i, 3]
    noisy_targets[i, 3] = target_states[i, 3]

for i in range(40, 80):
    target_states[i, 0] = target_states[i-1, 0]
    target_states[i, 1] = target_states[i-1, 1] + dt*ref_vel[i - 1]
    target_states[i, 2] = ref_vel[i]
    target_states[i, 3] = np.pi/2
    noisy_targets[i, 0] = target_states[i, 0]
    noisy_targets[i, 1] = target_states[i, 1]
    noisy_targets[i, 2] = target_states[i, 3]
    noisy_targets[i, 3] = target_states[i, 3]

for i in range(1, ntimesteps):
    init_inputs[i - 1, 0] = (noisy_targets[i, 2] - noisy_targets[i - 1, 2])/dt
    init_inputs[i - 1, 1] = (noisy_targets[i, 3] - noisy_targets[i - 1, 3])/dt

# State constraints
for i in range(ntimesteps):
    xmax[i, 0] = target_states[i, 0] + 1
    xmax[i, 1] = target_states[i, 1] + 1
    xmax[i, 2] = target_states[i, 2] + 2
    xmax[i, 3] = target_states[i, 3] + 1
    xmin[i, 0] = target_states[i, 0] - 1
    xmin[i, 1] = target_states[i, 1] - 1
    xmin[i, 2] = target_states[i, 2] - 2
    xmin[i, 3] = target_states[i, 3] - 1


start = time.time()
mpc_optimizer = iterative_MPC_optimizer(system, noisy_targets, dt)
mpc_optimizer.set_bounds(xmax, xmin)
mpc_optimizer.set_init_inputs(init_inputs)
# mpc_optimizer.init_input_fixed = True
mpc_optimizer()

print(init_inputs[0, 1])
print(mpc_optimizer.inputs[0, 1])

# print(states[:, 0])
end = time.time()
print("Computation time: ", end - start)
mpc_optimizer.plot()