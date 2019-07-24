import osqp
import time
import random
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.linalg import block_diag
from systems import *
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


def dh_dx(x, ref_x):
    dh_dx = np.array([2*(x[0] - ref_x[0]), 2*(x[1] - ref_x[1]), 0.0, 0.0])
    return dh_dx


def get_h(x, ref_x, r):
    h = (x[0] - ref_x[0])**2 + (x[1] - ref_x[1])**2 - r**2
    # return 0
    return h

def get_uieq(x, ref_x, r):
    C = dh_dx(x, ref_x)
    return C[0]*x[0] + C[1]*x[1] - get_h(x, ref_x, r)

class sequential_QP_optimizer:
    def __init__(self, sys, constraint, target_states, dt):
        self.target_states = target_states
        self.horizon = self.target_states.shape[0]
        self.dt = dt
        self.converge = False
        self.system = sys
        self.constraint = constraint
        self.n_states = sys.state_size
        self.m_inputs = sys.control_size
        self.Q = sys.Q
        self.R = sys.R
        self.Qf = sys.Q_f
        self.maxIter = 20
        self.min_cost = 0.0
        self.LM_parameter = 0.0
        self.eps = 1e-3
        self.states = np.zeros((self.horizon, self.n_states))
        self.inputs = np.zeros((self.horizon - 1, self.m_inputs))
        self.u0 = self.inputs[0, :]
        self.init_input_fixed = False
        self.x0 = self.target_states[0, :]
        self.xmax = np.full((self.horizon, self.n_states), np.inf)
        self.xmin = np.full((self.horizon, self.n_states), -np.inf)
        self.raduis = 0.7


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
        return cost[0, 0]

    def sim(self, x0, inputs):
        states = np.zeros((self.horizon, self.n_states))
        states[0, :] = x0
        for i in range(self.horizon - 1):
            states[i + 1, :] = self.system.model_f(states[i], inputs[i])
        return states

    def set_bounds(self, xmax, xmin):
        self.xmax = xmax
        self.xmin = xmin

    def set_init_inputs(self, init_inputs):
        self.inputs = init_inputs
        self.u0 = init_inputs[0, :]

    # def plot(self):
    #     plt.figure(figsize=(8*1.1, 6*1.1))
    #     currentAxis = plt.gca()
    #     plt.title('MPC: 2D, x and y.  ')
    #     plt.axis('equal')
    #     plt.plot(noisy_targets[:, 0], noisy_targets[:, 1],
    #              '--r', label='Target', linewidth=2)
    #     plt.plot(self.states[:, 0], self.states[:, 1],
    #              '-+k', label='MPC', linewidth=1.0)
    #     plt.xlabel('x (meters)')
    #     plt.ylabel('y (meters)')
    #     for i in range(self.horizon):
    #         x = self.target_states[i, 0]
    #         y = self.target_states[i, 1]
    #         currentAxis.add_patch(Circle((x, y), radius=self.raduis, alpha=1))
    #     plt.figure(figsize=(8*1.1, 6*1.1))
    #     plt.title('iLQR: state vs. time.  ')
    #     plt.plot(self.states[:, 2], '-b', linewidth=1.0, label='speed')
    #     plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
    #     plt.ylabel('speed')
    #     plt.figure(figsize=(8*1.1, 6*1.1))
    #     plt.title('iLQR: input vs. time.  ')
    #     plt.plot(self.inputs[:, 0], '-b', linewidth=1.0, label='turning rate')
    #     plt.ylabel('inputs')
    #     plt.show()

    # def update_A(self):


    def __call__(self):
        P = sparse.block_diag([sparse.kron(sparse.eye(self.horizon - 1), self.Q), self.Qf,
                               sparse.kron(sparse.eye(self.horizon - 1), self.R)]).tocsc()
        q = -self.Q.dot(self.target_states[0, :])
        for i in range(1, self.horizon - 1):
            q = np.hstack([q, -self.Q.dot(self.target_states[i, :])])
        q = np.hstack([q, -self.Qf.dot(self.target_states[-1, :]),
                       np.zeros((self.horizon - 1)*self.m_inputs)])

        umin = np.ones(self.m_inputs)
        umax = np.ones(self.m_inputs)
        if self.system.control_limited:
            for i in range(self.m_inputs):
                umin[i] = self.system.control_limit[i, 0]
                umax[i] = self.system.control_limit[i, 1]
        else:
            umin = -umin*np.inf
            umax = umax*np.inf

        # xmin = self.xmin.ravel()
        # xmax = self.xmax.ravel()
        # lineq = np.hstack([xmin, np.kron(np.ones(self.horizon - 1), umin)])
        # uineq = np.hstack([xmax, np.kron(np.ones(self.horizon - 1), umax)])

        self.states = self.target_states
        self.min_cost = np.inf
        for iter in range(self.maxIter):
            Ad = []
            Bd = []
            for i in range(self.horizon - 1):
                Ai = self.system.compute_df_dx(self.states[i, :], self.inputs[i, :])
                Bi = self.system.compute_df_du(self.states[i, :], self.inputs[i, :])
                Ad.append(Ai)
                Bd.append(Bi)
            # print(Ad)
            Bu = sparse.csr_matrix(block_diag(*Bd))
            Ax = sparse.csr_matrix(block_diag(*Ad))
            # print(Ax)
            off_set = np.zeros(
                ((self.horizon - 1)*self.n_states, self.n_states))
            Ax = sparse.hstack([Ax, off_set])
            Ax_offset = np.hstack(
                [off_set, -np.eye(self.n_states*(self.horizon - 1))])
            Ax = Ax_offset + Ax
            # Bd = np.array([
            #     [0, 0],
            #     [0, 0],
            #     [self.dt, 0],
            #     [0, self.dt]
            # ])
            # Bu = sparse.kron(sparse.eye(self.horizon - 1), sparse.csr_matrix(Bd))
            Aeq = sparse.hstack([Ax, Bu])
            init_x = np.zeros((self.n_states, Aeq.shape[1]))
            init_u = np.zeros((self.m_inputs, Aeq.shape[1]))
            for i in range(self.n_states):
                init_x[i, i] = 1
            for i in range(self.m_inputs):
                init_u[i, self.horizon*self.n_states + i] = 1
            if self.init_input_fixed:
                Aeq = sparse.vstack([init_x, init_u, Aeq])
            else:
                Aeq = sparse.vstack([init_x, Aeq])
            # Cd = []
            # for i in range(self.horizon):
            #     Ci = dh_dx(self.states[i, :], self.target_states[i, :])
            #     # print(Ci)
            #     Cd.append(Ci)
            # Cu = block_diag(*Cd)
            Cu = self.constraint.get_linear_constraint(self.states)
            # print("Cu")
            # print(Cu)
            Du = np.eye((self.horizon - 1)*self.m_inputs)
            # print(Du)
            Aineq = sparse.csr_matrix(block_diag(Cu, Du))
            A = sparse.vstack([Aeq, Aineq]).tocsc()

            leq = []
            leq.extend(self.x0)
            if self.init_input_fixed:
                leq.extend(self.u0)
            for i in range(0, self.horizon - 1):
                d = -self.states[i + 1, :] + \
                    np.dot(Ad[i], self.states[i, :]) + \
                    np.dot(Bd[i], self.inputs[i, :])
                leq.extend(d)
            ueq = leq

            xmin, xmax = self.constraint.get_bounds(self.states)
            # print(xmin)
            # print(xmax)
            # xmax = np.array([])
            # xmin = -np.ones(self.horizon)*np.inf
            # for i in range(0, self.horizon):
            #     upper = get_uieq(self.states[i, :], self.target_states[i, :], self.raduis)
            #     xmax = np.append(xmax, upper)
            uineq = np.hstack([xmax, np.kron(np.ones(self.horizon - 1), umax)])
            lineq = np.hstack([xmin, np.kron(np.ones(self.horizon - 1), umin)])

            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])
            prob = osqp.OSQP()

            # # Setup workspace
            prob.setup(P, q, A, l, u, warm_start=False, verbose=False)
            res = prob.solve()
            # states = res.x[0: self.horizon*self.n_states]
            inputs = res.x[self.horizon*self.n_states:]
            # self.states = np.reshape(states, (-1, 4))
            self.inputs = np.reshape(inputs, (-1, 2))
            self.states = self.sim(self.x0, self.inputs)
            cost = self.cost()
            print("cost: ", cost)
            # or cost > self.min_cost:
            if abs(1 - cost/self.min_cost) < self.eps:
                break
            else:
                self.min_cost = cost


# eps = 1e-3
# num_state = 4
# num_input = 2
# horizon = 80

# ntimesteps = horizon
# target_states = np.zeros((ntimesteps, 4))
# noisy_targets = np.zeros((ntimesteps, 4))
# xmax = np.zeros((ntimesteps, 4))
# xmin = np.zeros((ntimesteps, 4))
# noisy_targets[0, 2] = 1
# target_states[0, 2] = 1
# ref_vel = np.ones(ntimesteps)
# # ref_vel = np.zeros(ntimesteps)
# dt = 0.2
# curv = 0.1
# a = 1.5
# v_max = 11

# system = Car()
# system.set_dt(dt)
# system.set_cost(np.diag([50.0, 50.0, 10.0, 1.0]), np.diag([300.0, 1000.0]))
# system.Q_f = system.Q*horizon/10  # *50
# system.set_control_limit(np.array([[-1, 1], [-0.2, 0.2]]))
# # system.control_limited = False
# init_inputs = np.zeros((ntimesteps - 1, num_input))


# # for i in range(40, ntimesteps):
# #     if ref_vel[i - 1] > v_max:
# #         a = 0
# #     ref_vel[i] = ref_vel[i - 1] + a*dt

# # for i in range(1, ntimesteps):
# #     target_states[i, 0] = target_states[i-1, 0] + np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
# #     target_states[i, 1] = target_states[i-1, 1] + np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
# #     target_states[i, 2] = ref_vel[i]
# #     target_states[i, 3] = target_states[i-1, 3] + curv*dt
# #     noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 0.5)
# #     noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 0.5)
# #     noisy_targets[i, 2] = ref_vel[i]
# #     noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.1)

# # corner inputs case

# for i in range(1, 40):
#     target_states[i, 0] = target_states[i-1, 0] + dt*ref_vel[i - 1]
#     target_states[i, 1] = target_states[i-1, 1]
#     target_states[i, 2] = ref_vel[i]
#     target_states[i, 3] = 0
#     noisy_targets[i, 0] = target_states[i, 0]
#     noisy_targets[i, 1] = target_states[i, 1]
#     noisy_targets[i, 2] = target_states[i, 3]
#     noisy_targets[i, 3] = target_states[i, 3]

# for i in range(40, 80):
#     target_states[i, 0] = target_states[i-1, 0]
#     target_states[i, 1] = target_states[i-1, 1] + dt*ref_vel[i - 1]
#     target_states[i, 2] = ref_vel[i]
#     target_states[i, 3] = np.pi/2
#     noisy_targets[i, 0] = target_states[i, 0]
#     noisy_targets[i, 1] = target_states[i, 1]
#     noisy_targets[i, 2] = target_states[i, 3]
#     noisy_targets[i, 3] = target_states[i, 3]

# for i in range(1, ntimesteps):
#     init_inputs[i - 1, 0] = (noisy_targets[i, 2] - noisy_targets[i - 1, 2])/dt
#     init_inputs[i - 1, 1] = (noisy_targets[i, 3] - noisy_targets[i - 1, 3])/dt

# # State constraints
# for i in range(ntimesteps):
#     xmax[i, 0] = target_states[i, 0] + 1
#     xmax[i, 1] = target_states[i, 1] + 1
#     xmax[i, 2] = target_states[i, 2] + 2
#     xmax[i, 3] = target_states[i, 3] + 1
#     xmin[i, 0] = target_states[i, 0] - 1
#     xmin[i, 1] = target_states[i, 1] - 1
#     xmin[i, 2] = target_states[i, 2] - 2
#     xmin[i, 3] = target_states[i, 3] - 1


# start = time.time()
# mpc_optimizer = iterative_MPC_optimizer(system, noisy_targets, dt)
# mpc_optimizer.set_bounds(xmax, xmin)
# mpc_optimizer.set_init_inputs(init_inputs)
# # mpc_optimizer.init_input_fixed = True
# mpc_optimizer()

# print(init_inputs[0, 1])
# print(mpc_optimizer.inputs[0, 1])

# # print(states[:, 0])
# end = time.time()
# print("Computation time: ", end - start)
# mpc_optimizer.plot()
