import numpy as np
import scipy.sparse as sparse
from scipy.linalg import block_diag
import osqp

import random
from matplotlib import pyplot as plt
import timeit
from systems import *

"iterative LQR with Quadratic cost"


class iterative_LQR:
    """
    iterative LQR can be used as a controller/trajectory optimizer.
    Reference:
    Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization
    https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
    Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems
    https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf
    cost function: x'Qx + u'Ru
    """

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
        self.maxIter = 100
        self.min_cost = 0.0
        self.LM_parameter = 0.0
        self.states = np.zeros(
            (self.horizon, self.n_states))
        self.inputs = np.zeros(
            (self.horizon - 1, self.m_inputs))

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

    def forward_pass(self):
        prev_states = np.copy(self.states)
        prev_inputs = np.copy(self.inputs)
        alpha = 1.0
        cnt = 50
        while (cnt >= 0):
            cnt -= 1
            for i in range(0, self.horizon - 1):
                self.inputs[i, :] = self.inputs[i, :] + alpha*np.reshape(self.k[i, :, :], (-1,)) + np.reshape(
                    np.dot(self.K[i, :, :], np.reshape(self.states[i, :] - prev_states[i, :], (-1, 1))), (-1,))
                if self.system.control_limited:
                    for j in range(self.m_inputs):
                        self.inputs[i, j] = min(max(
                            self.inputs[i, j], self.system.control_lower_limit[j]), self.system.control_upper_limit[j])
                self.states[i + 1, :] = self.system.model_f(
                    self.states[i, :], self.inputs[i, :])
            cost = self.cost()
            if cost < self.min_cost:
                self.min_cost = cost
                break
            else:
                self.states = np.copy(prev_states)
                self.inputs = np.copy(prev_inputs)
                if alpha < 1e-4:
                    self.converge = True
                    break
                alpha /= 2.0

    def backward_pass(self):
        self.k = np.zeros((self.horizon - 1, self.m_inputs, 1))
        self.K = np.zeros((self.horizon - 1, self.m_inputs, self.n_states))
        Vx = 2.0 *\
            np.dot(
                self.Qf, self.states[-1, :] - self.target_states[-1, :])
        Vxx = 2.0*self.Qf
        dl_dxdx = 2.0*self.Q
        dl_dudu = 2.0*self.R
        dl_dudx = np.zeros((self.m_inputs, self.n_states))
        for i in range(self.horizon - 2, -1, -1):
            u = self.inputs[i, :]
            x = self.states[i, :]
            df_du = self.system.compute_df_du(x, u)
            df_dx = self.system.compute_df_dx(x, u)
            dl_dx = 2.0*np.dot(self.Q, x - self.target_states[i, :])
            dl_du = 2.0*np.dot(self.R, u)
            Qx = dl_dx + np.dot(df_dx.T, Vx)
            Qu = dl_du + np.dot(df_du.T, Vx)
            Vxx_augmented = Vxx + self.LM_parameter * np.eye(self.n_states)
            Qxx = dl_dxdx + np.dot(np.dot(df_dx.T, Vxx_augmented), df_dx)
            Quu = dl_dudu + np.dot(np.dot(df_du.T, Vxx_augmented), df_du)
            Qux = dl_dudx + np.dot(np.dot(df_du.T, Vxx_augmented), df_dx)
            Quu_inv = np.linalg.inv(Quu)

            P = sparse.csc_matrix(Quu)
            q = 2*Qu
            A = sparse.csc_matrix(np.eye(self.m_inputs))
            l = self.system.control_lower_limit - self.inputs[i, :]
            u = self.system.control_upper_limit - self.inputs[i, :]
            prob = osqp.OSQP()

            # Setup workspace and change alpha parameter
            prob.setup(P, q, A, l, u, verbose=False)

            # Solve problem
            res = prob.solve()

            # print(res.x)
            k = res.x;
            
            # k = -np.dot(Quu_inv, Qu)
            # print(k)
            # K = -np.dot(Quu_inv, Qux)
            if (abs(res.y[1]) > 0.0001):
                K = np.zeros((self.m_inputs, self.n_states))
            else:
                K = -np.dot(Quu_inv, Qux)
            # K = -np.dot(Quu_inv, Qux)
            # print(K)
            self.k[i, :, :] = np.reshape(k, (-1, 1), order='F')
            self.K[i, :, :] = K
            Vx = Qx + np.dot(K.T, Qu)
            Vxx = Qxx + np.dot(K.T, Qux)

    def __call__(self):
        self.min_cost = self.cost()
        for iter in range(self.maxIter):
            # print(iter)
            print(self.cost())
            if (self.converge):
                break
            self.backward_pass()
            self.forward_pass()
        return self.states



def example_acc():
    horizon = 120
    target_states = np.zeros((horizon, 4))
    noisy_targets = np.zeros((horizon, 4))
    ref_vel = np.zeros(horizon)
    dt = 0.2
    curv = 0.1
    a = 1.5
    v_max = 11
    car_system = Car()
    car_system.set_dt(dt)
    car_system.set_cost(
        np.diag([50.0, 50.0, 1000.0, 0.0]), np.diag([3000.0, 1000.0]))
    car_system.set_control_limit([-1.5, -0.3], [1.5, 0.3])
    car_system.Q_f = car_system.Q*2
    init_inputs = np.zeros((horizon - 1, car_system.control_size))

    for i in range(40, horizon):
        if ref_vel[i - 1] > v_max:
            a = 0
        ref_vel[i] = ref_vel[i - 1] + a*dt
    for i in range(1, horizon):
        target_states[i, 0] = target_states[i-1, 0] + \
            np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
        target_states[i, 1] = target_states[i-1, 1] + \
            np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
        target_states[i, 2] = ref_vel[i]
        target_states[i, 3] = target_states[i-1, 3] + curv*dt
        noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 3)
        noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 3)
        noisy_targets[i, 2] = ref_vel[i]
        noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.5)

    for i in range(1, horizon):
        init_inputs[i - 1, 0] = (noisy_targets[i, 2] -
                                 noisy_targets[i - 1, 2])/dt
        init_inputs[i - 1, 1] = (noisy_targets[i, 3] -
                                 noisy_targets[i - 1, 3])/dt

    optimizer = iterative_LQR(
        car_system, noisy_targets, dt)
    optimizer.inputs = init_inputs

    start_time = timeit.default_timer()
    optimizer()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)

    jerks = np.zeros(horizon)
    for i in range(1, horizon - 1):
        jerks[i] = (optimizer.inputs[i, 0] - optimizer.inputs[i - 1, 0])/dt

    plt.figure
    plt.title('jerks')
    plt.plot(jerks, '--r', label='jerks', linewidth=2)

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(noisy_targets[:, 0],
             noisy_targets[:, 1], '--r', label='Target', linewidth=2)
    plt.plot(optimizer.states[:, 0], optimizer.states[:, 1],
             '-+k', label='iLQR', linewidth=1.0)
    plt.legend(loc='upper left')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: state vs. time.  ')
    plt.plot(optimizer.states[:, 2], '-b', linewidth=1.0, label='speed')
    plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
    plt.ylabel('speed')
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: inputs vs. time.  ')
    plt.plot(optimizer.inputs[:, 0], '-r',
             linewidth=1.0, label='Acceleration')
    plt.plot(optimizer.inputs[:, 1], '-b',
             linewidth=1.0, label='turning rate')
    plt.ylabel('acceleration and turning rate input')
    plt.show()

if __name__ == '__main__':
    # example_dubins()
    # example_jerk()
    example_acc()
