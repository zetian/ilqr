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
        self.maxIter = 1
        self.min_cost = 0.0
        self.LM_parameter = 0.0
        self.eps = 1e-3
        self.states = np.zeros((self.horizon, self.n_states))
        self.inputs = np.zeros((self.horizon - 1, self.m_inputs))
        self.xmax = np.full((self.horizon, self.n_states), np.inf)
        self.xmin = np.full((self.horizon, self.n_states), -np.inf)

    def cost(self):
        states_diff = self.states - self.target_states
        cost = 0.0
        for i in range(self.horizon - 1):
            state = np.reshape(states_diff[i, :], (-1, 1))
            control = np.reshape(self.inputs[i, :], (-1, 1))
            cost += np.dot(np.dot(state.T, self.Q), state) + np.dot(np.dot(control.T, self.R), control)
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
    
    def plot(self):
        plt.figure(figsize=(8*1.1, 6*1.1))
        currentAxis = plt.gca()
        plt.title('MPC: 2D, x and y.  ')
        plt.axis('equal')
        plt.plot(noisy_targets[:, 0], noisy_targets[:, 1], '--r', label='Target', linewidth=2)
        plt.plot(self.states[:, 0], self.states[:, 1], '-+k', label='MPC', linewidth=1.0)
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        for i in range(self.horizon):
            x = self.xmin[i, 0]
            size_x = abs(self.xmax[i, 0] - self.xmin[i, 0])
            y = self.xmin[i, 1]
            size_y = abs(self.xmax[i, 1] - self.xmin[i, 1])
            currentAxis.add_patch(Rectangle((x, y), size_x, size_y, alpha=1))
        # plt.figure(figsize=(8*1.1, 6*1.1))
        # plt.title('iLQR: state vs. time.  ')
        # plt.plot(self.states[:, 2], '-b', linewidth=1.0, label='speed')
        # plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
        # plt.ylabel('speed')
        # plt.figure(figsize=(8*1.1, 6*1.1))
        # plt.title('iLQR: input vs. time.  ')
        # plt.plot(self.inputs[:, 1], '-b', linewidth=1.0, label='turning rate')
        # plt.ylabel('inputs')
        plt.show()
    
    def __call__(self):
        P = sparse.block_diag([sparse.kron(sparse.eye(self.horizon - 1), self.Q), self.Qf,
                            sparse.kron(sparse.eye(self.horizon - 1), self.R)]).tocsc()
        q = -self.Q.dot(self.target_states[0, :])
        for i in range(1, self.horizon - 1):
            q = np.hstack([q, -self.Q.dot(self.target_states[i, :])])
        q = np.hstack([q, -self.Qf.dot(self.target_states[-1, :]), np.zeros((self.horizon - 1)*self.m_inputs)])

        umin = np.ones(self.m_inputs)
        umax = np.ones(self.m_inputs)
        if self.system.control_limited:
            for i in range(self.m_inputs):
                umin[i] = self.system.control_limit[i, 0]
                umax[i] = self.system.control_limit[i, 1]
        else:
            umin = -umin*np.inf
            umax = umax*np.inf
        
        xmin = self.xmin.ravel()
        xmax = self.xmax.ravel()
        lineq = np.hstack([xmin, np.kron(np.ones(self.horizon - 1), umin)])
        print("lineq: ", lineq.shape)
        uineq = np.hstack([xmax, np.kron(np.ones(self.horizon - 1), umax)])

        self.states = self.target_states
        self.min_cost = np.inf
        for iter in range(self.maxIter):
            Ad = []
            Bd = []
            for i in range(0, self.horizon - 2):
                Ai = system.compute_df_dx(self.states[i, :], self.inputs[i, :])
                Bi = system.compute_df_du(self.states[i, :], self.inputs[i, :])
                Ad.append(Ai)
                Bd.append(Bi)
            A_1_block = []
            A_2_block = []
            for it in Ad:
                A_1_block.append(-it)
                A_2_block.append(1 + it)
            # print(A_1_block)  
            unit_offset_A = np.zeros(((self.horizon - 2)*self.n_states, self.n_states))
            print("unit_offset_A: ", unit_offset_A.shape)
            A_1 = sparse.csr_matrix(block_diag(*A_1_block))
            print("A_1: ", A_1.shape)
            A_2 = sparse.csr_matrix(block_diag(*A_2_block))
            print("A_2: ", A_2.shape)
            A_3 = sparse.eye((self.horizon - 2)*self.n_states)
            print("A_3: ", A_3.shape)
            Ax = sparse.hstack([A_1, unit_offset_A, unit_offset_A]) + sparse.hstack([unit_offset_A, A_2, unit_offset_A]) + sparse.hstack([unit_offset_A, unit_offset_A, A_3])
            print(Ax.shape)
            
            # off_set = np.zeros(((self.horizon - 1)*self.n_states, self.n_states))
            # Ax = sparse.hstack([Ax, off_set])
            # Ax_offset = np.hstack([off_set, -np.eye(self.n_states*(self.horizon - 1))])
            # Ax = Ax_offset + Ax
            # Bd = np.array([
            #     [0, 0],
            #     [0, 0],
            #     [self.dt, 0],
            #     [0, self.dt]
            # ])
            # Bu = sparse.kron(sparse.eye(self.horizon - 1), sparse.csr_matrix(Bd))
            unit_offset_B = np.zeros(((self.horizon - 2)*self.n_states, self.m_inputs))
            B_1_block = []
            for it in Bd:
                B_1_block.append(-it)
            B_1 = sparse.csr_matrix(block_diag(*B_1_block))
            B_2 = sparse.csr_matrix(block_diag(*Bd))
            Bu = sparse.hstack([B_1, unit_offset_B]) + sparse.hstack([unit_offset_B, B_2])
            print(Bu.shape)
            # Bu = sparse.csr_matrix(block_diag(*Bd))
            Aeq = sparse.hstack([Ax, Bu])
            
            init_x = np.zeros((self.n_states, Aeq.shape[1]))
            init_u = np.zeros((self.m_inputs, Aeq.shape[1]))
            for i in range(self.n_states):
                init_x[i, i] = 1
            for i in range(self.m_inputs):
                init_u[i, self.horizon*self.n_states + i] = 1
            Aeq = sparse.vstack([init_x, init_u, Aeq])
            print(Aeq)
            print("Aeq.shape")
            print(Aeq.shape)
            Aineq = sparse.eye(self.horizon*self.n_states + (self.horizon - 1)*self.m_inputs)
            print("Aineq")
            print(Aineq.shape)
            A = sparse.vstack([Aeq, Aineq]).tocsc()
            # leq = np.zeros((self.horizon - 1)*self.n_states)
            x0 = self.target_states[0, :]
            u0 = self.inputs[0, :]
            leq = np.hstack([x0, u0, np.zeros((self.horizon - 2)*self.n_states)])
            print("leq: ", leq.shape)
            # leq = []
            # leq.extend(self.target_states[0, :])
            # for i in range(0, self.horizon - 1):
            #     d = -self.states[i + 1, :] + np.dot(Ad[i], self.states[i, :]) + np.dot(Bd[i], self.inputs[i, :])
            #     leq.extend(d)
            ueq = leq
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
            self.states = self.sim(self.target_states[0, :], self.inputs)
            cost = self.cost()
            print("cost: ", cost)
            if abs(1 - cost/self.min_cost) < self.eps: # or cost > self.min_cost:
                break
            else:
                self.min_cost = cost



eps = 1e-3
num_state = 4
num_input = 2
horizon = 3

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
system.Q_f = system.Q*horizon/100#*50
system.set_control_limit(np.array([[-1, 1], [-0.2, 0.2]]))
system.control_limited = False
init_inputs = np.zeros((ntimesteps - 1, num_input))


# for i in range(40, ntimesteps):
#     if ref_vel[i - 1] > v_max:
#         a = 0
#     ref_vel[i] = ref_vel[i - 1] + a*dt

for i in range(1, ntimesteps):
    target_states[i, 0] = target_states[i-1, 0] + np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
    target_states[i, 1] = target_states[i-1, 1] + np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
    target_states[i, 2] = ref_vel[i]
    target_states[i, 3] = target_states[i-1, 3] + curv*dt
    noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 0.5)
    noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 0.5)
    noisy_targets[i, 2] = ref_vel[i]
    noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.1)

# corner inputs case

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

for i in range(1, ntimesteps):
    init_inputs[i - 1, 0] = (noisy_targets[i, 2] - noisy_targets[i - 1, 2])/dt
    init_inputs[i - 1, 1] = (noisy_targets[i, 3] - noisy_targets[i - 1, 3])/dt

# State constraints
for i in range(ntimesteps):
    xmax[i, 0] = target_states[i, 0] + 0.5
    xmax[i, 1] = target_states[i, 1] + 0.5
    xmax[i, 2] = target_states[i, 2] + 0.1
    xmax[i, 3] = target_states[i, 3] + 0.1
    xmin[i, 0] = target_states[i, 0] - 0.5
    xmin[i, 1] = target_states[i, 1] - 0.5
    xmin[i, 2] = target_states[i, 2] - 0.1
    xmin[i, 3] = target_states[i, 3] - 0.1


start = time.time()
mpc_optimizer= iterative_MPC_optimizer(system, noisy_targets, dt)
# mpc_optimizer.set_bounds(xmax, xmin)
mpc_optimizer.inputs = init_inputs
mpc_optimizer()
# print(states[:, 0])
end = time.time()
print("Computation time: ", end - start)
mpc_optimizer.plot()





