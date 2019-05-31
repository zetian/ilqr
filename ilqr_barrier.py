import numpy as np
import random
from matplotlib import pyplot as plt
import timeit
from systems import *
from ilqr import iterative_LQR_quadratic_cost
from constraints import CircleConstraintForCar

"iterative LQR with Quadratic cost"


class iterative_LQR_quadratic_cost:
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
        self.horizon = self.target_states.shape[1]
        self.dt = dt
        self.converge = False
        self.system = sys
        self.n_states = sys.state_size
        self.m_inputs = sys.control_size
        self.Q = sys.Q
        self.R = sys.R
        self.Qf = sys.Q_f
        self.maxIter = 30
        self.min_cost = 0.0
        self.LM_parameter = 0.0
        self.states = np.zeros(
            (self.n_states, self.horizon))
        self.inputs = np.zeros(
            (self.m_inputs, self.horizon - 1))
        self.obs_list = []
        self.obstacle_weight = 100
        self.obstacle_weight_2 = 9

    def set_obstacles(self, obs):
        self.obs_list = obs

    def cost(self):
        states_diff = self.states - self.target_states
        cost = 0.0
        for i in range(self.horizon - 1):
            state = np.reshape(states_diff[:,i], (-1,1))
            control = np.reshape(self.inputs[:,i], (-1,1) )
            cost += np.dot(np.dot(state.T, self.Q), state) + np.dot(np.dot(control.T, self.R), control)
            for obs in self.obs_list:
                # if (self.state_sequence[0, i] - obs[0])**2 + (self.state_sequence[1, i] - obs[1])**2 - obs[2]**2 > 0:
                #     print(np.exp (-( (self.state_sequence[0, i] - obs[0])**2 + (self.state_sequence[1, i] - obs[1])**2 - obs[2]**2 )*self.obstacle_weight))
            
                cost += self.obstacle_weight*np.exp(self.obstacle_weight_2*(obs[2]**2 - (self.states[0, i] - obs[0])**2 - (self.states[1, i] - obs[1])**2))

        state = np.reshape(states_diff[:,-1], (-1,1))
        cost += np.dot(np.dot(state.T, self.Qf), state)
        for obs in self.obs_list:
                # cost += self.obs_w*np.exp (-( (self.state_sequence[0, -1] - obs[0])**2 + (self.state_sequence[1, -1] - obs[1])**2 - obs[2]**2 )*self.obstacle_weight)
                cost += self.obstacle_weight*np.exp(self.obstacle_weight_2*(obs[2]**2 - (self.states[0, -1] - obs[0])**2 - (self.states[1, -1] - obs[1])**2))
        return cost

    def compute_dl_dx(self, x, xr):
        assert (x.shape[0] == self.n_states), "state dimension inconsistent with setup."
        dl_dx = 2.0*np.dot(self.Q, x - xr)
        # print(dl_dx)
        # a = np.array([1, 3, 4, 5])
        # dl_dx_obs = np.array([[0.0], [0.0], [0.0], [0.0]])
        dl_dx_obs = np.zeros(4, dtype = 'float')
        for obs in self.obs_list:
            dl_dx_obs += np.array([-2.0*(x[0] - obs[0]), 
                                  -2.0*(x[1] - obs[1]),
                                  0.0,
                                  0.0])*np.exp(self.obstacle_weight*(obs[2]**2 - (x[0] - obs[0])**2 - (x[1] - obs[1])**2))*self.obstacle_weight
        # print(dl_dx_obs)
        dl_dx += dl_dx_obs
        return dl_dx

    def compute_dl_dxdx(self, x, xr):
        # assert (x.shape == (self.n_states,1)), "state dimension inconsistent with setup."
        dl_dxdx = 2.0* self.Q
        dl_dxdx_obs = np.zeros(shape = (4, 4), dtype = 'float')
        for obs in self.obs_list:
            # dl_dxdx_obs += np.array([[-2*self.obstacle_weight + 4*self.obstacle_weight**2*(x[0] - obs[0])**2, 4*self.obstacle_weight**2*(x[1] - obs[1])*(x[0] - obs[0]), 0.0, 0.0],
            #                       [4*self.obstacle_weight**2*(x[1] - obs[1])*(x[0] - obs[0]), -2*self.obstacle_weight + 4*self.obstacle_weight**2*self.obstacle_weight*(x[1] - obs[1])**2, 0.0, 0.0],
            #                       [0.0, 0.0, 0.0, 0.0],
            #                       [0.0, 0.0, 0.0, 0.0]])*np.exp(self.obstacle_weight*(obs[2]**2 - (x[0] - obs[0])**2 - (x[1] - obs[1])**2))
            dl_dxdx_obs += np.array([[4*(x[0] - obs[0])**2, 4*(x[1] - obs[1])*(x[0] - obs[0]), 0.0, 0.0],
                                  [4*(x[1] - obs[1])*(x[0] - obs[0]), 4*(x[1] - obs[1])**2, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0]])*np.exp(self.obstacle_weight*(obs[2]**2 - (x[0] - obs[0])**2 - (x[1] - obs[1])**2))*self.obstacle_weight**2
        
        dl_dxdx += dl_dxdx_obs
        return dl_dxdx

    def forward_pass(self):
        prev_states = np.copy(self.states)
        prev_inputs = np.copy(self.inputs)
        # prev_cost = self.cost()
        alpha = 1.0
        cnt = 50
        while (cnt >= 0):
            cnt -= 1
            for i in range(0, self.horizon - 1):
            # for i in range(0, 2):
                self.inputs[:, i] = self.inputs[:, i] + alpha*np.reshape(self.k[i, :, :], (-1,)) + np.reshape(
                    np.dot(self.K[i, :, :], np.reshape(self.states[:, i] - prev_states[:, i], (-1, 1))), (-1,))
                # print("self.inputs[:, i] before constraints: ", self.inputs[:, i])
                if self.system.control_limited:
                    for j in range(self.m_inputs):
                        self.inputs[j, i] = min(max(
                            self.inputs[j, i], self.system.control_limit[j, 0]), self.system.control_limit[j, 1])
                # print("self.inputs[:, i]: ", self.inputs[:, i])
                self.states[:, i + 1] = self.system.model_f(
                    self.states[:, i], self.inputs[:, i])
                # print("self.states[:, i]: ", self.states[:, i])
                # print("self.states[:, i + 1]: ", self.states[:, i + 1])
            cost = self.cost()
            if cost < self.min_cost:
                self.min_cost = cost
                # print('cost decreased after this pass. learning_rate: ', alpha)
                break
            elif alpha < 1e-4:
                self.converge = True
                # print(
                #     'learning_rate below threshold. Unable to reduce cost. learning_rate: ', alpha)
                break
            else:
                alpha /= 2.0
                self.states = np.copy(prev_states)
                self.inputs = np.copy(prev_inputs)

    def backward_pass(self):
        self.k = np.zeros((self.horizon - 1, self.m_inputs, 1))
        self.K = np.zeros((self.horizon - 1, self.m_inputs, self.n_states))
        Vx = 2.0 *\
            np.dot(
                self.Qf, self.states[:, -1] - self.target_states[:, -1])
        Vxx = 2.0*self.Qf
        # dl_dxdx = 2.0*self.Q
        dl_dudu = 2.0*self.R
        # print(Vxx)
        dl_dudx = np.zeros((self.m_inputs, self.n_states))
        for i in range(self.horizon - 2, -1, -1):
        # for i in range(0, -1, -1):
            u = self.inputs[:, i]
            x = self.states[:, i]
            df_du = self.system.compute_df_du(x, u)
            # print("df_du", df_du)
            df_dx = self.system.compute_df_dx(x, u)
            dl_dx = self.compute_dl_dx(x, self.target_states[:,i] )
            # dl_dx = 2.0*np.dot(self.Q, x - self.target_states[:, i])
            dl_du = 2.0*np.dot(self.R, u)
            Qx = dl_dx + np.dot(df_dx.T, Vx)
            # print("Qx", Qx)
            Qu = dl_du + np.dot(df_du.T, Vx)
            # print("Qx", Qu)
            dl_dxdx = self.compute_dl_dxdx(x, self.target_states[:,i] )
            Vxx_augmented = Vxx + self.LM_parameter * np.eye(self.n_states)
            Qxx = dl_dxdx + np.dot(np.dot(df_dx.T, Vxx_augmented), df_dx)
            # print("Qxx", Qxx)
            Quu = dl_dudu + np.dot(np.dot(df_du.T, Vxx_augmented), df_du)
            # print("Quu", Quu)
            Qux = dl_dudx + np.dot(np.dot(df_du.T, Vxx_augmented), df_dx)
            Quu_inv = np.linalg.inv(Quu)
            k = -np.dot(Quu_inv, Qu)
            K = -np.dot(Quu_inv, Qux)
            self.k[i, :, :] = np.reshape(k, (-1, 1), order='F')
            self.K[i, :, :] = K
            Vx = Qx + np.dot(K.T, Qu)
            Vxx = Qxx + np.dot(K.T, Qux)
            # print("Vxx", Vxx)

    def __call__(self):
        # print(self.inputs)
        self.min_cost = self.cost()
        # print("init cost: ", self.min_cost)
        for iter in range(self.maxIter):
            if (self.converge):
                break
            self.backward_pass()
            self.forward_pass()
        # print(self.min_cost)
        return self.states


if __name__ == '__main__':
    obs_list = []
    obs_1 = [5, 40, 9]
    obs_list.append(obs_1)
    ntimesteps = 100
    target_states = np.zeros((4, ntimesteps))
    noisy_targets = np.zeros((4, ntimesteps))
    ref_vel = np.zeros(ntimesteps)
    dt = 0.2
    curv = 0.1
    a = 1.5
    v_max = 11
    
    car_system = Car()
    car_system.set_dt(dt)
    car_system.set_cost(
        np.diag([50.0, 50.0, 1000.0, 0.0]), np.diag([3000.0, 1000.0]))
    car_system.set_control_limit(np.array([[-1.5, 1.5], [-0.3, 0.3]]))
    init_inputs = np.zeros((car_system.control_size, ntimesteps - 1))

    for i in range(40, ntimesteps):
        if ref_vel[i - 1] > v_max:
            a = 0
        ref_vel[i] = ref_vel[i - 1] + a*dt
    for i in range(1, ntimesteps):
        target_states[0, i] = target_states[0, i-1] + \
            np.cos(target_states[3, i-1])*dt*ref_vel[i - 1]
        target_states[1, i] = target_states[1, i-1] + \
            np.sin(target_states[3, i-1])*dt*ref_vel[i - 1]
        target_states[2, i] = ref_vel[i]
        target_states[3, i] = target_states[3, i-1] + curv*dt
        noisy_targets[0, i] = target_states[0, i] + random.uniform(0, 5.0)
        noisy_targets[1, i] = target_states[1, i] + random.uniform(0, 5.0)
        noisy_targets[2, i] = target_states[2, i]
        noisy_targets[3, i] = target_states[3, i] + random.uniform(0, 1.0)
    
    for i in range(1, ntimesteps):
        init_inputs[0, i - 1] = (noisy_targets[2, i] - noisy_targets[2, i - 1])/dt
        init_inputs[1, i - 1] = (noisy_targets[3, i] - noisy_targets[3, i - 1])/dt

    myiLQR = iterative_LQR_quadratic_cost(
        car_system, noisy_targets, dt)
    myiLQR.set_obstacles(obs_list)
    myiLQR.inputs = init_inputs

    start_time = timeit.default_timer()
    myiLQR()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)

    jerks = np.zeros(ntimesteps)
    for i in range(1, ntimesteps - 1):
        jerks[i] = (myiLQR.inputs[0, i] - myiLQR.inputs[0, i - 1])/dt
    
    plt.figure
    plt.title('jerks')
    plt.plot(jerks, '--r', label='jerks', linewidth=2)

    plt.figure(figsize=(8*1.1, 6*1.1))
    ax = plt.gca()
    plt.title('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(noisy_targets[0, :],
             noisy_targets[1, :], '--g', label='Target', linewidth=2)
    plt.plot(myiLQR.states[0, :], myiLQR.states[1, :],
             '-+b', label='iLQR', linewidth=1.0)
    for obs in obs_list:
        # print(obs[0], obs[1])
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='r')
        ax.add_artist(circle)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.legend()
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: state vs. time.  ')
    plt.plot(myiLQR.states[2, :], '-b', linewidth=1.0, label='speed')
    plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
    plt.ylabel('speed')
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: inputs vs. time.  ')
    plt.plot(myiLQR.inputs[0, :], '-r',
             linewidth=1.0, label='Acceleration')
    plt.plot(myiLQR.inputs[1, :], '-b',
             linewidth=1.0, label='turning rate')
    plt.ylabel('acceleration and turning rate input')
    plt.legend()
    plt.show()