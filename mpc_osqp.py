import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.linalg import block_diag

# Discrete time model of a quadcopter
# Ad = sparse.csc_matrix([
#   [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
#   [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
#   [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
#   [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
#   [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
#   [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
#   [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
#   [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
#   [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
#   [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
#   [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
#   [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
# ])

Ad = sparse.csc_matrix([
    [0, 0, 0.3, 0.7],
    [0, 0, 0.7, -0.3],
    [0, 0, 0, 0],
    [0, 0, 0.5, 0]
])

# Bd = sparse.csc_matrix([
#   [0.,      -0.0726,  0.,     0.0726],
#   [-0.0726,  0.,      0.0726, 0.    ],
#   [-0.0152,  0.0152, -0.0152, 0.0152],
#   [-0.,     -0.0006, -0.,     0.0006],
#   [0.0006,   0.,     -0.0006, 0.0000],
#   [0.0106,   0.0106,  0.0106, 0.0106],
#   [0,       -1.4512,  0.,     1.4512],
#   [-1.4512,  0.,      1.4512, 0.    ],
#   [-0.3049,  0.3049, -0.3049, 0.3049],
#   [-0.,     -0.0236,  0.,     0.0236],
#   [0.0236,   0.,     -0.0236, 0.    ],
#   [0.2107,   0.2107,  0.2107, 0.2107]])

Bd = sparse.csc_matrix([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
])


[nx, nu] = Bd.shape

# Constraints
u0 = 10.5916
umin = np.array([-1.5, -0.3])
umax = np.array([1.5, 0.3])
# xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
#                  -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
xmin = np.array([-np.inf, -np.inf, 0, -np.pi/2])

# xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
#                   np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
xmax = np.array([np.inf, np.inf, 10, np.pi/2])
# Objective function
# Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
Q = sparse.diags([10, 10, 10, 100])
QN = Q*10
R = 10*sparse.eye(2)

# Initial and reference states
x0 = np.zeros(4)
xr = np.array([10, 10, 0, 0])

# Prediction horizon
N = 2

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
# P_test = block_diag([np.kron(np.eye(N), Q), QN,
#                        np.kron(np.eye(N), R)])
# print(P_test)
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)]).tocsc()
print(P)
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq]).tocsc()
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
# prob = osqp.OSQP()

# Setup workspace
# prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 15
# for i in range(nsim):
#     # Solve
#     res = prob.solve()

#     # Check solver status
#     if res.info.status != 'solved':
#         raise ValueError('OSQP did not solve the problem!')

#     # Apply first control input to the plant
#     ctrl = res.x[-N*nu:-(N-1)*nu]
#     x0 = Ad.dot(x0) + Bd.dot(ctrl)

#     # Update initial state
#     l[:nx] = -x0
#     u[:nx] = -x0
#     prob.update(l=l, u=u)
