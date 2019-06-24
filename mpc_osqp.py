import osqp
import random
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.linalg import block_diag
from systems import *
from matplotlib import pyplot as plt
# from scipy import linalg
num_state = 4
num_input = 2
horizon = 80

ntimesteps = horizon
target_states = np.zeros((ntimesteps, 4))
noisy_targets = np.zeros((ntimesteps, 4))
ref_vel = np.zeros(ntimesteps)
dt = 0.2
curv = 0.1
a = 1.5
v_max = 11

car_system = Car()
car_system.set_dt(dt)
car_system.set_cost(
    np.diag([50.0, 50.0, 1000.0, 0.0]), np.diag([30.0, 1000.0]))
car_system.set_control_limit(np.array([[-1.5, 1.5], [-0.3, 0.3]]))
init_inputs = np.zeros((ntimesteps - 1, car_system.control_size))

Q = sparse.diags([50.0, 50.0, 1000.0, 0.0])
QN = Q*100
R = sparse.diags([300.0, 10000.0])

for i in range(40, ntimesteps):
    if ref_vel[i - 1] > v_max:
        a = 0
    ref_vel[i] = ref_vel[i - 1] + a*dt
for i in range(1, ntimesteps):
    target_states[i, 0] = target_states[i-1, 0] + \
        np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
    target_states[i, 1] = target_states[i-1, 1] + \
        np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
    target_states[i, 2] = ref_vel[i]
    target_states[i, 3] = target_states[i-1, 3] + curv*dt
    noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 10.0)
    noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 10.0)
    noisy_targets[i, 2] = target_states[i, 2]
    noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 1.0)

for i in range(1, ntimesteps):
    init_inputs[i - 1, 0] = (noisy_targets[i, 2] - noisy_targets[i - 1, 2])/dt
    init_inputs[i - 1, 1] = (noisy_targets[i, 3] - noisy_targets[i - 1, 3])/dt

num_sim = 10

sim_states = noisy_targets
sim_inputs = init_inputs
# print(noisy_targets.shape)
# res_x = []
# res_y = []
states = []
for i in range(num_sim):

    Ad = []
    for i in range(ntimesteps - 1):
        Ai = car_system.compute_df_dx(sim_states[i, :], sim_inputs[i, :])
        Ad.append(Ai)
    aux = np.empty((0, num_state), int)
    Ax_offset = block_diag(aux.T, *Ad, aux)
    Ax_offset = sparse.csr_matrix(Ax_offset)

    Ax = sparse.kron(sparse.eye(ntimesteps),-sparse.eye(num_state)) + Ax_offset
    Bd = sparse.csc_matrix([
        [0, 0],
        [0, 0],
        [0.2, 0],
        [0, 0.2]
    ])
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, ntimesteps - 1)), sparse.eye(ntimesteps - 1)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    P = sparse.block_diag([sparse.kron(sparse.eye(ntimesteps - 1), Q), QN,
                        sparse.kron(sparse.eye(ntimesteps - 1), R)]).tocsc()
    # print("P: ")
    # print(P.shape)
    q = -Q.dot(noisy_targets[0, :])
    for i in range(1, ntimesteps - 1):
        q = np.hstack([q, -Q.dot(noisy_targets[i, :])])
    q = np.hstack([q, -QN.dot(noisy_targets[-1, :]), np.zeros((ntimesteps - 1)*num_input)])
    # print("q: ")
    # print(q.shape)
    umin = np.array([-1.5, -0.3])
    umax = np.array([1.5, 0.3])
    xmin = np.array([-np.inf, -np.inf, 0, -np.pi/2])

    xmax = np.array([np.inf, np.inf, 10, np.pi/2])
    x0 = noisy_targets[0, :]
    leq = np.hstack([-x0, np.zeros((ntimesteps - 1)*num_state)])
    # print("leq: ")
    # print(leq.shape)
    ueq = leq
    Aineq = sparse.eye(ntimesteps*num_state + (ntimesteps - 1)*num_input)
    lineq = np.hstack([np.kron(np.ones(ntimesteps), xmin), np.kron(np.ones(ntimesteps - 1), umin)])
    uineq = np.hstack([np.kron(np.ones(ntimesteps), xmax), np.kron(np.ones(ntimesteps - 1), umax)])

    A = sparse.vstack([Aeq, Aineq]).tocsc()
    # print("A: ")
    # print(A.shape)
    l = np.hstack([leq, lineq])
    # print("l: ")
    # print(l.shape)
    u = np.hstack([ueq, uineq])
    # print("u: ")
    # print(u.shape)

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
    res = prob.solve()
    # obj_val = prob.obj_val()
    print(res.info.obj_val)
    # print(len(res.x))
    states = res.x[0: ntimesteps*num_state]
    inputs = res.x[ntimesteps*num_state:]
    inputs = np.reshape(inputs, (-1, 2))
    states = np.reshape(states, (-1, 4))
    # print(states[10, :])
    # print(states.shape)
    sim_states = states
    sim_inputs = inputs
    # print(states)
    # print(inputs[0, :])
    # res_x = []
    # res_y = []
    # for state in states:
    #     res_x.append(state[0])
    #     res_y.append(state[1])

# print(states[:, 0])

plt.figure(figsize=(8*1.1, 6*1.1))
plt.title('MPC: 2D, x and y.  ')
plt.axis('equal')
plt.plot(noisy_targets[:, 0], noisy_targets[:, 1], '--r', label='Target', linewidth=2)
plt.plot(states[:, 0], states[:, 1], '-+b', label='MPC', linewidth=1.0)
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.figure(figsize=(8*1.1, 6*1.1))
plt.title('iLQR: state vs. time.  ')
plt.plot(states[:, 2], '-b', linewidth=1.0, label='speed')
plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
plt.ylabel('speed')
plt.show()
# for i in range(len(res.x)):


# Simulate in closed loop
# nsim = 15
# for i in range(nsim):
#     # Solve
#     res = prob.solve()

#     # Check solver status
#     if res.info.status != 'solved':
#         raise ValueError('OSQP did not solve the problem!')

#     # Apply first control input to the plant
#     ctrl = res.x[-N*nu:-(N-1)*nu]
#     x0 = Ad.dot(x0) + Bd.dot(ctrl)

#     # Update initial state
#     l[:nx] = -x0
#     u[:nx] = -x0
#     prob.update(l=l, u=u)
