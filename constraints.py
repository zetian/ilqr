import numpy as np
from scipy.linalg import block_diag

class Constraint:
    def __init__(self, state_size, constraint_size):
        self.state_size = state_size
        self.constraint_size = constraint_size
        self.horizon = 0

class BubbleConstraint(Constraint):
    def __init__(self, horizon):
        super().__init__(4, 2)
        self.horizon = horizon
    
    def setup(self, centers, radius, vel_bounds):
        self.centers = centers
        self.radius = radius
        self.vel_bounds = vel_bounds

    def get_h(self, x, center, r):
        h = (x[0] - center[0])**2 + (x[1] - center[1])**2 - r**2
        return h
    
    def dh_dx(self, x, center):
        dhdx = np.array([[2*(x[0] - center[0]), 2*(x[1] - center[1]), 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        return dhdx

    def get_uieq(self, state, center, r):
        dhdx = self.dh_dx(state, center)
        return dhdx[0, 0]*state[0] + dhdx[0, 1]*state[1] - self.get_h(state, center, r)

    def get_linear_constraint(self, states):
        Cd = []
        for i in range(self.horizon):
            Ci = self.dh_dx(states[i, :], self.centers[i, :])
            Cd.append(Ci)
        Cu = block_diag(*Cd)
        return Cu

    def get_bounds(self, states):
        xmax = np.array([])
        xmin = np.array([])
        for i in range(0, self.horizon):
            lower = np.array([-np.inf, self.vel_bounds[0]])
            upper = np.array([self.get_uieq(states[i, :], self.centers[i, :], self.radius[i]), self.vel_bounds[1]])
            xmax = np.append(xmax, upper)
            xmin = np.append(xmin, lower)
        return xmin, xmax
