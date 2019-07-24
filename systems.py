import numpy as np


class System:
    def __init__(self, state_size, control_size):
        self.control_limited = False
        self.state_size = state_size
        self.control_size = control_size

    def set_dt(self, dt):
        self.dt = dt

    def set_cost(self, Q, R):
        self.Q = Q
        self.R = R
        self.Q_f = Q*10

    def set_final_cost(self, Q_f):
        self.Q_f = Q_f

    def set_control_limit(self, limit):
        self.control_limited = True
        self.control_limit = limit


class Car(System):
    def __init__(self):
        super().__init__(4, 2)
        self.dt = 0.2
        self.control_bound = np.array([0.5, 5])

    def model_f(self, x, u):
        theta = x[3]
        v = x[2]
        acc = u[0]
        theta_rate = u[1]
        x_next = np.reshape(x, (-1, 1), order='F') + np.array(
            [[v*np.cos(theta)], [v*np.sin(theta)], [acc], [theta_rate]])*self.dt
        return np.reshape(x_next, (-1,))

    def compute_df_dx(self, x, u):
        theta = x[3]
        v = x[2]
        df_dx = np.array([[1.0, 0.0, np.cos(theta)*self.dt, -np.sin(theta)*v*self.dt],
                          [0.0, 1.0, np.sin(theta)*self.dt, np.cos(theta)*v*self.dt],
                          [0.0, 0.0,  1.0, 0.0],
                          [0.0, 0.0,  0.0, 1.0]
                          ])
        return df_dx

    def compute_df_du(self, x, u):
        df_du = np.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])*self.dt
        return df_du

class CarAcceleration(System):
    def __init__(self):
        super().__init__(5, 2)
        self.dt = 0.2

    def model_f(self, x, u):
        theta = x[4]
        v = x[2]
        acc = x[3]
        jerk = u[0]
        theta_rate = u[1]
        x_next = np.reshape(x, (-1, 1), order='F') + np.array(
            [[v*np.cos(theta)], [v*np.sin(theta)], [acc], [jerk], [theta_rate]])*self.dt
        return np.reshape(x_next, (-1,))

    def compute_df_dx(self, x, u):
        theta = x[4]
        v = x[2]
        df_dx = np.array([[1.0, 0.0, np.cos(theta)*self.dt, 0.0, -np.sin(theta)*v*self.dt],
                          [0.0, 1.0, np.sin(theta)*self.dt, 0.0, np.cos(theta)*v*self.dt],
                          [0.0, 0.0,  1.0, self.dt, 0.0],
                          [0.0, 0.0,  0.0, 1.0, 0.0],
                          [0.0, 0.0,  0.0, 0.0, 1.0]])
        return df_dx

    def compute_df_du(self, x, u):
        df_du = np.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])*self.dt
        return df_du    

class DubinsCar(System):
    def __init__(self):
        super().__init__(3, 2)
        self.dt = 0.2

    def model_f(self, x, u):
        theta = x[2]
        v = u[0]
        curvature = u[1]
        x_next = np.reshape(x, (-1, 1), order='F') + np.array(
            [[v*np.cos(theta)], [v*np.sin(theta)], [v*curvature]])*self.dt
        return np.reshape(x_next, (-1,))

    def compute_df_dx(self, x, u):
        theta = x[2]
        v = u[0]
        df_dx = np.array([[1.0, 0.0,  -np.sin(theta)*v*self.dt],
                          [0.0, 1.0,   np.cos(theta)*v*self.dt],
                          [0.0, 0.0,  1.0]
                          ])
        return df_dx

    def compute_df_du(self, x, u):
        theta = x[2]
        v = u[0]
        curvature = u[1]
        df_du = np.array([[np.cos(theta), 0.0],
                          [np.sin(theta), 0.0],
                          [curvature, v]])*self.dt
        return df_du
