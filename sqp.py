import osqp
import time
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.linalg import block_diag
from systems import *


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
        self.eps = 1e-6
        self.states = np.zeros((self.horizon, self.n_states))
        self.inputs = np.zeros((self.horizon - 1, self.m_inputs))
        self.u0 = self.inputs[0, :]
        self.init_input_fixed = False
        self.x0 = self.target_states[0, :]
        # self.xmax = np.full((self.horizon, self.n_states), np.inf)
        # self.xmin = np.full((self.horizon, self.n_states), -np.inf)
        # self.raduis = 0.7

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

    def set_init_inputs(self, init_inputs):
        self.inputs = init_inputs
        self.u0 = init_inputs[0, :]

    def plot(self):
        plt.figure(figsize=(8*1.1, 6*1.1))
        # currentAxis = plt.gca()
        plt.title('SQP: 2D, x and y.  ')
        plt.axis('equal')
        plt.plot(self.target_states[:, 0], self.target_states[:, 1],
                 '--r', label='Target', linewidth=2)
        plt.plot(self.states[:, 0], self.states[:, 1],
                 '-+k', label='MPC', linewidth=1.0)
        plt.show()

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

        self.states = self.target_states
        self.min_cost = np.inf

        converged = False

        for iter in range(self.maxIter):
            if converged:
                break
            Ad = []
            Bd = []
            for i in range(self.horizon - 1):
                Ai = self.system.compute_df_dx(
                    self.states[i, :], self.inputs[i, :])
                Bi = self.system.compute_df_du(
                    self.states[i, :], self.inputs[i, :])
                Ad.append(Ai)
                Bd.append(Bi)
            Bu = sparse.csr_matrix(block_diag(*Bd))
            Ax = sparse.csr_matrix(block_diag(*Ad))
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
            Cu = self.constraint.get_linear_constraint(self.states)
            Du = np.eye((self.horizon - 1)*self.m_inputs)
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
            uineq = np.hstack([xmax, np.kron(np.ones(self.horizon - 1), umax)])
            lineq = np.hstack([xmin, np.kron(np.ones(self.horizon - 1), umin)])

            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])

            prob = osqp.OSQP()
            prob.setup(P, q, A, l, u, warm_start=False, verbose=False)
            res = prob.solve()
            prev_states = np.copy(self.states)
            prev_inputs = np.copy(self.inputs)
            solved_inputs = np.reshape(
                res.x[self.horizon*self.n_states:], (-1, 2))
            d_u = solved_inputs - self.inputs
            alpha = 1.0
            while True:
                cost = self.cost()
                if alpha < 1e-3 or abs(1 - cost/self.min_cost) < self.eps:
                    converged = True
                    break
                self.inputs = prev_inputs + alpha*d_u
                self.states = self.sim(self.x0, self.inputs)
                if (cost < self.min_cost):
                    self.min_cost = cost
                    break
                else:
                    alpha /= 2.0
                    self.states = np.copy(prev_states)
                    self.inputs = np.copy(prev_inputs)
