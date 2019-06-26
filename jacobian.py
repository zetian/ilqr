import osqp
import time
import random
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.linalg import block_diag
from systems import *
from matplotlib import pyplot as plt


class iterative_MPC_optimizer:
    def __init__(self, sys, target_states, dt):
        self.target_states = target_states
        self.horizon = self.target_states.shape[0]
        self.dt = dt
        self.converge = False
        self.system = sys
        self.n_states = sys.state_size
        self.m_inputs = sys.control_size
        self.Q = sys.Q
        self.R = sys.R
        self.Qf = sys.Q_f
        self.maxIter = 10
        self.min_cost = 0.0
        self.LM_parameter = 0.0
        # self.states = np.zeros((self.n_states, self.horizon))
        # self.inputs = np.zeros((self.m_inputs, self.horizon - 1))

        self.states = np.zeros((self.horizon, self.n_states))
        self.inputs = np.zeros((self.horizon - 1, self.m_inputs))
        # print(self.states.shape)
    # def cost(self):
    #     states_diff = self.states - self.target_states
    #     cost = 0.0
    #     for i in range(self.horizon - 1):
    #         state = np.reshape(states_diff[:, i], (-1, 1))
    #         control = np.reshape(self.inputs[:, i], (-1, 1))
    #         cost += np.dot(np.dot(state.T, self.Q), state) + \
    #             np.dot(np.dot(control.T, self.R), control)
    #     state = np.reshape(states_diff[:, -1], (-1, 1))
    #     cost += np.dot(np.dot(state.T, self.Qf), state)
    #     return cost

    def cost(self):
        states_diff = self.states - self.target_states
        cost = 0.0
        for i in range(self.horizon - 1):
            state = np.reshape(states_diff[i, :], (-1, 1))
            control = np.reshape(self.inputs[i, :], (-1, 1))
            cost += np.dot(np.dot(state.T, self.Q), state) + \
                np.dot(np.dot(control.T, self.R), control)
        state = np.reshape(states_diff[-1, :], (-1, 1))
        cost += np.dot(np.dot(state.T, self.Qf), state)
        return cost
    
    def __call__(self):
        # umin = np.array([-1.5, -0.15]) - self.inputs[0, :]
        # umax = np.array([1.5, 0.15]) - self.inputs[0, :]
        # xmin = np.array([-np.inf, -np.inf, -10, -np.inf])
        # xmax = np.array([np.inf, np.inf, 10, np.inf])
        # lineq = np.hstack([np.kron(np.ones(self.horizon), xmin), np.kron(np.ones(self.horizon - 1), umin)])
        # uineq = np.hstack([np.kron(np.ones(self.horizon), xmax), np.kron(np.ones(self.horizon - 1), umax)])
        # P = sparse.block_diag([sparse.kron(sparse.eye(self.horizon - 1), self.Q), self.Qf,
        #                     sparse.kron(sparse.eye(self.horizon - 1), self.R)]).tocsc()
        # q = -self.Q.dot(self.target_states[0, :])
        # for i in range(1, self.horizon - 1):
        #     q = np.hstack([q, -self.Q.dot(self.target_states[i, :])])
        # q = np.hstack([q, -self.Qf.dot(self.target_states[-1, :]), np.zeros((self.horizon - 1)*self.m_inputs)])
        # x0 = self.target_states[0, :]
        # leq = np.hstack([-x0, np.zeros((self.horizon - 1)*self.n_states)])
        # ueq = leq
        # Aineq = sparse.eye(self.horizon*self.n_states + (self.horizon - 1)*self.m_inputs)
        # l = np.hstack([leq, lineq])
        # u = np.hstack([ueq, uineq])
        self.states = self.target_states
        # sim_inputs = self.inputs
        self.min_cost = self.cost()
        # print("init cost: ")
        # print(self.min_cost)
        last_cost = 0
        for iter in range(self.maxIter):

            P = sparse.block_diag([sparse.kron(sparse.eye(self.horizon - 1), self.Q), self.Qf,
                                sparse.kron(sparse.eye(self.horizon - 1), self.R)]).tocsc()
            q = -self.Q.dot(self.target_states[0, :])
            for i in range(1, self.horizon - 1):
                q = np.hstack([q, -self.Q.dot(self.target_states[i, :])])
            q = np.hstack([q, -self.Qf.dot(self.target_states[-1, :]), np.zeros((self.horizon - 1)*self.m_inputs)])
            # x0 = self.target_states[0, :]
            # leq = np.hstack([-x0, np.zeros((self.horizon - 1)*self.n_states)])
            # ueq = leq
            
            Ad = []
            for i in range(self.horizon - 1):
                Ai = system.compute_df_dx(self.states[i, :], self.inputs[i, :])
                Ad.append(Ai)
            # aux = np.empty((0, self.n_states), int)
            Ax = sparse.csr_matrix(block_diag(*Ad))
            off_set = np.zeros(((self.horizon - 1)*self.n_states, self.n_states))
            # print(off_set)
            Ax = sparse.hstack([Ax, off_set])
            # print(Ax)
            # Ax_offset = sparse.csr_matrix(Ax_offset)
            # print(-np.eye(self.n_states*(self.horizon - 1)))
            Ax_offset = np.hstack([off_set, -np.eye(self.n_states*(self.horizon - 1))])

            # print(Ax_offset.shape)
            # print(Ax_offset)
            Ax = Ax_offset + Ax
            # print(Ax.shape)
            # Bd = sparse.csc_matrix([
            #     [0, 0],
            #     [0, 0],
            #     [0.2, 0],
            #     [0, 0.2]
            # ])
            Bd = np.array([
                [0, 0],
                [0, 0],
                [0.2, 0],
                [0, 0.2]
            ])
            Bu = sparse.kron(sparse.eye(self.horizon - 1), sparse.csr_matrix(Bd))
            # print(Bu.shape)
            # Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.horizon - 1)), sparse.eye(self.horizon - 1)]), Bd)
            
            Aeq = sparse.hstack([Ax, Bu])
            # print(np.zeros((4, Aeq.shape[1])))
            init = np.zeros((4, Aeq.shape[1]))
            init[0, 0] = 1
            init[1, 1] = 1
            init[2, 2] = 1
            init[3, 3] = 1
            Aeq = sparse.vstack([init, Aeq])
            # Aeq[0, 0] = 1
            # Aeq[1, 1] = 1
            # Aeq[2, 2] = 1
            # Aeq[3, 3] = 1
            # print(Aeq)
            Aineq = sparse.eye(self.horizon*self.n_states + (self.horizon - 1)*self.m_inputs)
            A = sparse.vstack([Aeq, Aineq]).tocsc()
            # print(Ad[0])
            # print(self.states[1, :])
            # print(np.dot(Ad[0], self.states[0, :]))
            # print("~~")
            # print(self.inputs[0, :])
            # Bd = np.array([
            #     [0, 0],
            #     [0, 0],
            #     [0.2, 0],
            #     [0, 0.2]
            # ])
            # print(Bd)
            # print(np.dot(Bd, self.inputs[0, :]))
            # d = self.states[1, :] - np.dot(Ad[0], self.states[0, :]) - np.dot(Bd, self.inputs[0, :])
            # print(d)
            leq = []
            leq.extend(self.target_states[0, :])
            for i in range(0, self.horizon - 1):
                d = -self.states[i + 1, :] + np.dot(Ad[i], self.states[i, :]) + np.dot(Bd, self.inputs[i, :])
                # print(d)
                leq.extend(d)
            # print(leq)
            # print("leq: ", len(leq))

            # x0 = self.target_states[0, :]

            # leq = np.hstack([-x0, np.zeros((self.horizon - 1)*self.n_states)])
            # print(leq)
            ueq = leq
            umin = np.array([-1.5, -0.2])# - self.inputs[0, :]
            umax = np.array([1.5, 0.2])# - self.inputs[0, :]
            xmin = np.array([-np.inf, -np.inf, -10, -np.inf])
            xmax = np.array([np.inf, np.inf, 10, np.inf])
            lineq = np.hstack([np.kron(np.ones(self.horizon), xmin), np.kron(np.ones(self.horizon - 1), umin)])
            uineq = np.hstack([np.kron(np.ones(self.horizon), xmax), np.kron(np.ones(self.horizon - 1), umax)])
            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])
            prob = osqp.OSQP()

            # # Setup workspace
            prob.setup(P, q, A, l, u, warm_start=False, verbose=True)
            res = prob.solve()
            # obj_val = prob.obj_val()
            # print(res.info.obj_val)
            # if abs(1 - last_cost/res.info.obj_val) < eps:
            #     break
            last_cost = res.info.obj_val
            
            # print(len(res.x))
            states = res.x[0: self.horizon*self.n_states]
            inputs = res.x[self.horizon*self.n_states:]
            self.states = np.reshape(states, (-1, 4))
            self.inputs = np.reshape(inputs, (-1, 2))
            # cost = self.cost()
            # print(cost)



eps = 1e-3
num_state = 4
num_input = 2
horizon = 30

ntimesteps = horizon
target_states = np.zeros((ntimesteps, 4))
noisy_targets = np.zeros((ntimesteps, 4))
noisy_targets[0, 2] = 1
target_states[0, 2] = 1
ref_vel = np.ones(ntimesteps)
dt = 0.2
curv = 0.1
a = 1.5
v_max = 11

system = Car()
system.set_dt(dt)
system.set_cost(np.diag([50.0, 50.0, 10.0, 1.0]), np.diag([30.0, 100000.0]))
system.set_control_limit(np.array([[-1.5, 1.5], [-0.3, 0.3]]))
init_inputs = np.zeros((ntimesteps - 1, num_input))





Q = sparse.diags([50.0, 50.0, 1000.0, 0.0])
QN = Q*horizon/2
R = sparse.diags([30.0, 5000.0])

# for i in range(40, ntimesteps):
#     if ref_vel[i - 1] > v_max:
#         a = 0
#     ref_vel[i] = ref_vel[i - 1] + a*dt
for i in range(1, ntimesteps):
    target_states[i, 0] = target_states[i-1, 0] + \
        np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
    target_states[i, 1] = target_states[i-1, 1] + \
        np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
    target_states[i, 2] = ref_vel[i]
    target_states[i, 3] = target_states[i-1, 3] + curv*dt
    noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 0.05)
    noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 0.05)
    noisy_targets[i, 2] = ref_vel[i]
    noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.01)

# print(noisy_targets)

for i in range(1, ntimesteps):
    init_inputs[i - 1, 0] = (noisy_targets[i, 2] - noisy_targets[i - 1, 2])/dt
    init_inputs[i - 1, 1] = (noisy_targets[i, 3] - noisy_targets[i - 1, 3])/dt

# print(init_inputs)
# print("noisy_targets")
# print(noisy_targets.shape)
mpc_optimizer= iterative_MPC_optimizer(system, noisy_targets, dt)
mpc_optimizer.inputs = init_inputs
mpc_optimizer()


# print(states[:, 0])
# end = time.time()
# print(end - start)
# print(cnt)
plt.figure(figsize=(8*1.1, 6*1.1))
plt.title('MPC: 2D, x and y.  ')
plt.axis('equal')
plt.plot(noisy_targets[:, 0], noisy_targets[:, 1], '--r', label='Target', linewidth=2)
plt.plot(mpc_optimizer.states[:, 0], mpc_optimizer.states[:, 1], '-+b', label='MPC', linewidth=1.0)
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
# plt.figure(figsize=(8*1.1, 6*1.1))
# plt.title('iLQR: state vs. time.  ')
# plt.plot(states[:, 2], '-b', linewidth=1.0, label='speed')
# plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
# plt.ylabel('speed')
plt.show()




