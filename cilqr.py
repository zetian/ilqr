import numpy as np
import osqp

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
        self.LM_parameter = 0.0
        self.constraints = []
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
        prev_states = np.copy(self.states)
        prev_inputs = np.copy(self.inputs)
        prev_cost = self.cost()
        alpha = 1.0
        while (True):
            for i in range(0, self.horizon - 1):
                self.inputs[:, i] = self.inputs[:, i] + alpha*np.reshape(self.k[i, :, :], (-1,)) + np.reshape(
                    np.dot(self.K[i, :, :], np.reshape(self.states[:, i] - prev_states[:, i], (-1, 1))), (-1,))
                if self.system.control_limited:
                    for j in range(self.m_inputs):
                        self.inputs[j, i] = min(max(
                            self.inputs[j, i], self.system.control_limit[j, 0]), self.system.control_limit[j, 1])
                self.states[:, i + 1] = self.system.model_f(
                    self.states[:, i], self.inputs[:, i])
            cost = self.cost()
            if cost < prev_cost:
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
            Quu_inv = np.linalg.inv(Quu)
            # k = -np.dot(Quu_inv, Qu)
            # K = -np.dot(Quu_inv, Qux)
            C = np.empty((self.system.control_size + len(self.constraints), self.system.control_size))
            D = np.empty((self.system.control_size + len(self.constraints), self.system.state_size))
            index = 0
            constraint_index = np.zeros((2 * self.system.control_size + len(self.constraints) * self.system.state_size, self.horizon))
            for j in range(self.system.control_size):
                if u[j] >= self.system.control_bound[j] - self.active_set_tol:
                    e = np.zeros(self.system.control_size)
                    e[j] = 1
                    C[index, :] = e
                    D[index, :] = np.zeros(self.system.state_size)
                    index += 1
                    constraint_index[j, i] = 1
                elif u[j] <= -self.system.control_bound[j] + self.active_set_tol:
                    e = np.zeros(self.system.control_size)
                    e[j] = -1
                    C[index, :] = e
                    D[index, :] = np.zeros(self.system.state_size)
                    index += 1
                    constraint_index[j + self.system.control_size, i] = 1
            if i <= self.horizon - 2: #state constraint can be violated
                for j in range(len(self.constraints)):
                    D_constraint = self.constraints[j].evaluate_constraint(self.x_trajectories[:, i+1])
                    #print("constraint", D_constraint, i)
                    if abs(D_constraint) <= self.active_set_tol:
                        C_constraint = self.constraints[j].evaluate_constraint_J(self.x_trajectories[:, i+1])
                        C[index, :] = C_constraint.dot(f_u)
                        #print(C_constraint.dot(f_u))
                        D[index, :] = -C_constraint.dot(f_x)
                        index = index + 1
                        constraint_index[2 * self.system.control_size + j, i] = 1

            if index == 0: #no constraint active
            K = -inv(Q_uu).dot(Q_ux)
            k = -inv(Q_uu).dot(Q_u)
            else:
            C = C[0:index, :]
            D = D[0:index, :]
            lambda_temp = C.dot(inv(Q_uu)).dot(C.T)
            lambda_temp = -inv(lambda_temp).dot(C).dot(inv(Q_uu)).dot(Q_u)

            #remove active constraint with lambda < 0
            index = 0
            delete_index = []
            #control constraint
            for j in range(self.system.control_size):
                if constraint_index[j, i] == 1:
                    if lambda_temp[index] < 0:
                        constraint_index[j, i] = 0
                        C[index, :] = np.zeros(self.system.control_size)
                        delete_index.append(index)
                    index = index + 1
                elif constraint_index[j + self.system.control_size, i] == 1:
                    if lambda_temp[index] < 0:
                        constraint_index[j + self.system.control_size, i] = 0
                        C[index, :] = np.zeros(self.system.control_size)
                        delete_index.append(index)
                    index = index + 1
            #state constrait
            for j in range(len(self.constraints)):
                if constraint_index[j + 2 * self.system.control_size, i] == 1:
                    if lambda_temp[index] < 0:
                        constraint_index[j + 2 * self.system.control_size, i] = 0
                        C[index, :] = np.zeros(self.system.control_size)
                        delete_index.append(index)
                    index += 1

            if len(delete_index) < C.shape[0]:
                C = np.delete(C, delete_index, axis=0)
                D = np.delete(D, delete_index, axis=0)
                C_star = inv(C.dot(inv(Q_uu)).dot(C.T)).dot(C).dot(inv(Q_uu))
                H_star = inv(Q_uu).dot(np.identity(self.system.control_size) - C.T.dot(C_star))
                k = -H_star.dot(Q_u)
                K = -H_star.dot(Q_ux) + C_star.T.dot(D)
            else:
                K = -inv(Q_uu).dot(Q_ux)
                k = -inv(Q_uu).dot(Q_u)


            self.k[i, :, :] = np.reshape(k, (-1, 1), order='F')
            self.K[i, :, :] = K
            Vx = Qx + np.dot(K.T, Qu)
            Vxx = Qxx + np.dot(K.T, Qux)

    def __call__(self):
        for iter in range(self.maxIter):
            if (self.converge):
                break
            self.backward_pass()
            self.forward_pass()
        return self.states
