import numpy as np
from numpy.linalg import inv
import osqp
import scipy.sparse as sparse
import random
import timeit
from systems import *
from constraints import CircleConstraintForCar

from matplotlib import pyplot as plt
"iterative LQR with Quadratic cost"


class cilqr:
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
        self.initial_state = target_states[:, 0]
        self.horizon = self.target_states.shape[1]
        self.dt = dt
        self.converge = False
        self.system = sys
        self.n_states = sys.state_size
        self.m_inputs = sys.control_size
        self.Q = sys.Q
        self.R = sys.R
        self.Qf = sys.Q_f
        self.maxIter = 5
        self.constraints = []
        self.Q_UX = np.zeros((self.system.control_size, self.system.state_size, self.horizon))
        self.Q_UU = np.zeros((self.system.control_size, self.system.control_size, self.horizon))
        self.Q_U = np.zeros((self.system.control_size, self.horizon))
        self.LM_parameter = 0.0
        self.constraints = []
        self.reg_factor = 0.001
        self.reg_factor_u = 0.001
        self.active_set_tol = 0.01
        self.states = np.zeros(
            (self.n_states, self.horizon))
        self.inputs = np.zeros(
            (self.m_inputs, self.horizon - 1))

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def cost(self):
        states_diff = self.states - self.target_states
        cost = 0.0
        for i in range(self.horizon - 1):
            state = np.reshape(states_diff[:, i], (-1, 1))
            control = np.reshape(self.inputs[:, i], (-1, 1))
            cost += np.dot(np.dot(state.T, self.Q), state) + \
                np.dot(np.dot(control.T, self.R), control)
        state = np.reshape(states_diff[:, -1], (-1, 1))
        cost += np.dot(np.dot(state.T, self.Qf), state)
        return cost

    def forward_pass(self):
        x = np.copy(self.initial_state)
        feasible = False
        trust_region_scale = 1
        while not feasible:
            feasible = True
            # current_J = 0
            x_new_trajectories = np.zeros((self.system.state_size, self.horizon))
            u_new_trajectories = np.zeros((self.system.control_size, self.horizon - 1))
            # x_new_trajectories = self.states
            # u_new_trajectories = self.inputs
            x = np.copy(self.initial_state)
            x_new_trajectories[:, 0] = np.copy(self.initial_state)
            for i in range(self.horizon - 1):
                delta_x = x - self.states[:, i]
                # delta_x = self.states[:, i + 1] - self.states[:, i]
                x_new_trajectories[:, i] = np.copy(x)
                Q_ux = self.Q_UX[:, :, i]
                Q_u = self.Q_U[:, i]
                P = sparse.csr_matrix(self.Q_UU[:, : , i])
                q = (Q_ux.dot(delta_x) + Q_u)
                '''lb = -self.system.control_bound - self.u_trajectories[:, i]
                ub = self.system.control_bound - self.u_trajectories[:, i]
                lb *= trust_region_scale
                ub *= trust_region_scale'''

                #constraint_A = sparse.csr_matrix(np.identity(self.system.control_size))

                #initialize contraint matrix and bound
                constraint_A = np.zeros((self.system.control_size + len(self.constraints), self.system.control_size))
                lb = np.zeros(self.system.control_size + len(self.constraints))
                ub = np.zeros(self.system.control_size + len(self.constraints))

                #control limit contraint
                constraint_A[0:self.system.control_size, 0:self.system.control_size] = np.identity(self.system.control_size)
                # lb[0:self.system.control_size] = -self.system.control_bound - self.inputs[:, i]
                # ub[0:self.system.control_size] = self.system.control_bound - self.inputs[:, i]
                # lb *= trust_region_scale
                # ub *= trust_region_scale

                # #formulate linearized state constraints
                # f_u = self.system.compute_df_du(x, self.inputs[:, i])
                # # print("df_du", df_du)
                # f_x = self.system.compute_df_dx(x, self.inputs[:, i])
                # # f_x, f_u = self.system.transition_J(x, self.u_trajectories[:, i])				
                # constraint_index = self.system.control_size
                # for constraint in self.constraints:
                #     if i <= self.horizon - 2:#current action might cause state constraint violation
                #         x_temp = self.system.model_f(x, self.inputs[:, i])
                #         D = constraint.evaluate_constraint(x_temp)
                #         #print("constraint eval", D, i, x)
                #         C = constraint.evaluate_constraint_J(x_temp)
                #         #print(C.shape, f_u.shape)
                #         C = C.dot(f_u)
                #         constraint_A[constraint_index, :] = np.copy(C)
                #         lb[constraint_index] = -np.inf #no lower bound
                #         ub[constraint_index] = -D
                #     constraint_index += 1

                constraint_A = sparse.csr_matrix(constraint_A)
                prob = osqp.OSQP()
                prob.setup(P, q, constraint_A, lb, ub, alpha=1.5, verbose=False)
                res = prob.solve()
                if res.info.status != 'solved':
                    feasible = False
                    #print("infeasible, reduce trust region")
                    trust_region_scale *= 0.9
                    break
                delta_u = res.x[0:self.system.control_size]
                u = delta_u + self.inputs[:, i]
                u_new_trajectories[:, i] = np.copy(u)
                # current_J += self.system.calculate_cost(x, u)
                x = self.system.model_f(x, u)
                x_new_trajectories[:, i + 1] = self.system.model_f(x_new_trajectories[:, i], u)
            # x_new_trajectories[:, self.horizon - 1] = np.copy(x)
            # current_J = self.cost()
            if feasible == True:
                # self.x_trajectories = np.copy(x_new_trajectories)
                self.states = np.copy(x_new_trajectories)
                self.inputs = np.copy(u_new_trajectories)
                # self.u_trajectories = np.copy(u_new_trajectories)
                print("total cost", self.cost())
                #self.system.draw_trajectories(self.x_trajectories)
                #self.system.draw_u_trajectories(self.u_trajectories)

    def backward_pass(self):
        self.k = np.zeros((self.horizon - 1, self.m_inputs, 1))
        self.K = np.zeros((self.horizon - 1, self.m_inputs, self.n_states))
        Vx = 2.0 *\
            np.dot(
                self.Qf, self.states[:, -1] - self.target_states[:, -1])
        Vxx = 2.0*self.Qf
        dl_dxdx = 2.0*self.Q
        dl_dudu = 2.0*self.R
        dl_dudx = np.zeros((self.m_inputs, self.n_states))
        for i in range(self.horizon - 2, -1, -1):
            u = self.inputs[:, i]
            x = self.states[:, i]
            df_du = self.system.compute_df_du(x, u)
            df_dx = self.system.compute_df_dx(x, u)
            dl_dx = 2.0*np.dot(self.Q, x - self.target_states[:, i])
            dl_du = 2.0*np.dot(self.R, u)
            Qx = dl_dx + np.dot(df_dx.T, Vx)
            Qu = dl_du + np.dot(df_du.T, Vx)
            Vxx_augmented = Vxx + self.LM_parameter * np.eye(self.n_states)
            Qxx = dl_dxdx + np.dot(np.dot(df_dx.T, Vxx_augmented), df_dx)
            Quu = dl_dudu + np.dot(np.dot(df_du.T, Vxx_augmented), df_du)
            Qux = dl_dudx + np.dot(np.dot(df_du.T, Vxx_augmented), df_dx)
            # Quu_inv = np.linalg.inv(Quu)
            # k = -np.dot(Quu_inv, Qu)
            # K = -np.dot(Quu_inv, Qux)
            # C = np.empty((self.system.control_size + len(self.constraints), self.system.control_size))
            # D = np.empty((self.system.control_size + len(self.constraints), self.system.state_size))
            # index = 0
            # constraint_index = np.zeros((2 * self.system.control_size + len(self.constraints) * self.system.state_size, self.horizon))
            # for j in range(self.system.control_size):
            #     if u[j] >= self.system.control_bound[j] - self.active_set_tol:
            #         e = np.zeros(self.system.control_size)
            #         e[j] = 1
            #         C[index, :] = e
            #         D[index, :] = np.zeros(self.system.state_size)
            #         index += 1
            #         constraint_index[j, i] = 1
            #     elif u[j] <= -self.system.control_bound[j] + self.active_set_tol:
            #         e = np.zeros(self.system.control_size)
            #         e[j] = -1
            #         C[index, :] = e
            #         D[index, :] = np.zeros(self.system.state_size)
            #         index += 1
            #         constraint_index[j + self.system.control_size, i] = 1
            # if i <= self.horizon - 2: #state constraint can be violated
            #     for j in range(len(self.constraints)):
            #         D_constraint = self.constraints[j].evaluate_constraint(self.states[:, i+1])
            #         #print("constraint", D_constraint, i)
            #         if abs(D_constraint) <= self.active_set_tol:
            #             C_constraint = self.constraints[j].evaluate_constraint_J(self.states[:, i+1])
            #             C[index, :] = C_constraint.dot(df_du)
            #             #print(C_constraint.dot(f_u))
            #             D[index, :] = -C_constraint.dot(df_dx)
            #             index = index + 1
            #             constraint_index[2 * self.system.control_size + j, i] = 1

            # if index == 0: #no constraint active
            #     K = -inv(Quu).dot(Qux)
            #     k = -inv(Quu).dot(Qu)
            # else:
            #     C = C[0:index, :]
            #     D = D[0:index, :]
            #     lambda_temp = C.dot(inv(Quu)).dot(C.T)
            #     lambda_temp = -inv(lambda_temp).dot(C).dot(inv(Quu)).dot(Qu)

            #     #remove active constraint with lambda < 0
            #     index = 0
            #     delete_index = []
            #     #control constraint
            #     for j in range(self.system.control_size):
            #         if constraint_index[j, i] == 1:
            #             if lambda_temp[index] < 0:
            #                 constraint_index[j, i] = 0
            #                 C[index, :] = np.zeros(self.system.control_size)
            #                 delete_index.append(index)
            #             index = index + 1
            #         elif constraint_index[j + self.system.control_size, i] == 1:
            #             if lambda_temp[index] < 0:
            #                 constraint_index[j + self.system.control_size, i] = 0
            #                 C[index, :] = np.zeros(self.system.control_size)
            #                 delete_index.append(index)
            #             index = index + 1
            #     #state constrait
            #     for j in range(len(self.constraints)):
            #         if constraint_index[j + 2 * self.system.control_size, i] == 1:
            #             if lambda_temp[index] < 0:
            #                 constraint_index[j + 2 * self.system.control_size, i] = 0
            #                 C[index, :] = np.zeros(self.system.control_size)
            #                 delete_index.append(index)
            #             index += 1

            #     if len(delete_index) < C.shape[0]:
            #         C = np.delete(C, delete_index, axis=0)
            #         D = np.delete(D, delete_index, axis=0)
            #         C_star = inv(C.dot(inv(Quu)).dot(C.T)).dot(C).dot(inv(Quu))
            #         H_star = inv(Quu).dot(np.identity(self.system.control_size) - C.T.dot(C_star))
            #         k = -H_star.dot(Qu)
            #         K = -H_star.dot(Qux) + C_star.T.dot(D)
            #     else:
            #         K = -inv(Quu).dot(Qux)
            #         k = -inv(Quu).dot(Qu)
            
            
            K = -inv(Quu).dot(Qux)
            k = -inv(Quu).dot(Qu)

            self.k[i, :, :] = np.reshape(k, (-1, 1), order='F')
            self.K[i, :, :] = K
            Vx = Qx + np.dot(K.T, Qu)
            Vxx = Qxx + np.dot(K.T, Qux)

            self.Q_UX[:, :, i] = Qux
            self.Q_UU[:, :, i] = Quu
            self.Q_U[:, i] = Qu

    def __call__(self):
        for iter in range(self.maxIter):
            if (self.converge):
                break
            self.backward_pass()
            self.forward_pass()
        return self.states


if __name__ == '__main__':
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
        np.diag([5000.0, 5000.0, 1.0, 1.0]), np.diag([1.0, 1.0]))
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
        noisy_targets[0, i] = target_states[0, i] + random.uniform(0, 5)
        noisy_targets[1, i] = target_states[1, i] + random.uniform(0, 5)
        noisy_targets[2, i] = target_states[2, i]
        noisy_targets[3, i] = target_states[3, i] + random.uniform(0, 1)
    
    for i in range(1, ntimesteps):
        init_inputs[0, i - 1] = (noisy_targets[2, i] - noisy_targets[2, i - 1])/dt
        init_inputs[1, i - 1] = (noisy_targets[3, i] - noisy_targets[3, i - 1])/dt
    
    myiLQR = cilqr(car_system, noisy_targets, dt)
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
    plt.title('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(noisy_targets[0, :],
             noisy_targets[1, :], '--r', label='Target', linewidth=2)
    plt.plot(myiLQR.states[0, :], myiLQR.states[1, :],
             '-+b', label='iLQR', linewidth=1.0)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
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
    plt.show()