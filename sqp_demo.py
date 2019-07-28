from systems import *
from sqp import *
from constraints import *
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


def random_example():
    horizon = 80
    target_states = np.zeros((horizon, 4))
    noisy_targets = np.zeros((horizon, 4))
    noisy_targets[0, 2] = 1
    target_states[0, 2] = 1
    centers = np.zeros((horizon, 2))
    ref_vel = np.ones(horizon)
    noisy = 0.5
    dt = 0.2
    curv = 0.1
    system = Car()
    system.set_dt(dt)
    system.set_cost(np.diag([50.0, 50.0, 10.0, 1.0]), np.diag([300.0, 1000.0]))
    system.Q_f = system.Q*horizon/100  # *50
    system.set_control_limit([-1, -0.3], [1, 0.3])
    init_inputs = np.zeros((horizon - 1, system.control_size))
    for i in range(1, horizon):
        target_states[i, 0] = target_states[i-1, 0] + \
            np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
        target_states[i, 1] = target_states[i-1, 1] + \
            np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
        target_states[i, 2] = ref_vel[i]
        target_states[i, 3] = target_states[i-1, 3] + curv*dt
        noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, noisy)
        noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, noisy)
        noisy_targets[i, 2] = ref_vel[i]
        noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.1)
        centers[i, 0] = noisy_targets[i, 0]
        centers[i, 1] = noisy_targets[i, 1]
    for i in range(1, horizon):
        init_inputs[i - 1, 0] = (noisy_targets[i, 2] -
                                 noisy_targets[i - 1, 2])/dt
        init_inputs[i - 1, 1] = (noisy_targets[i, 3] -
                                 noisy_targets[i - 1, 3])/dt
    constraint = BubbleConstraint(horizon)
    radius = []
    vel_bounds = [0, 2]
    r = 0.6
    for i in range(horizon):
        radius.append(r)
    constraint.setup(centers, radius, vel_bounds)
    start = time.time()
    mpc_optimizer = sequential_QP_optimizer(
        system, constraint, noisy_targets, dt)
    mpc_optimizer.set_init_inputs(init_inputs)
    mpc_optimizer()
    end = time.time()
    print("Computation time: ", end - start)

    plt.figure(figsize=(8*1.1, 6*1.1))
    currentAxis = plt.gca()
    plt.title('SQP: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(mpc_optimizer.target_states[:, 0], mpc_optimizer.target_states[:, 1],
             '--r', label='Target', linewidth=2)
    plt.plot(mpc_optimizer.states[:, 0], mpc_optimizer.states[:, 1],
             '-+k', label='SQP', linewidth=1.0)
    plt.legend(loc='upper left')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    for i in range(mpc_optimizer.horizon):
        x = mpc_optimizer.target_states[i, 0]
        y = mpc_optimizer.target_states[i, 1]
        r = radius[i]
        currentAxis.add_patch(Circle((x, y), radius=r, alpha=1))
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('SQP: state vs. time.  ')
    plt.plot(mpc_optimizer.states[:, 2], '-b', linewidth=1.0, label='speed')
    plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
    plt.ylabel('speed')
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('SQP: input vs. time.  ')
    plt.plot(mpc_optimizer.inputs[:, 0], '-b',
             linewidth=1.0, label='turning rate')
    plt.ylabel('inputs')
    plt.show()


def corner_example():
    horizon = 80
    target_states = np.zeros((horizon, 4))
    noisy_targets = np.zeros((horizon, 4))
    noisy_targets[0, 2] = 1
    target_states[0, 2] = 1
    centers = np.zeros((horizon, 2))
    ref_vel = np.ones(horizon)
    dt = 0.2
    system = Car()
    system.set_dt(dt)
    system.set_cost(np.diag([50.0, 50.0, 10.0, 1.0]), np.diag([300.0, 1000.0]))
    system.Q_f = system.Q*horizon/100  # *50
    system.set_control_limit([-1, -0.3], [1, 0.3])
    init_inputs = np.zeros((horizon - 1, system.control_size))
    for i in range(1, 40):
        target_states[i, 0] = target_states[i-1, 0] + dt*ref_vel[i - 1]
        target_states[i, 1] = target_states[i-1, 1]
        target_states[i, 2] = ref_vel[i]
        target_states[i, 3] = 0
        noisy_targets[i, 0] = target_states[i, 0]
        noisy_targets[i, 1] = target_states[i, 1]
        noisy_targets[i, 2] = target_states[i, 3]
        noisy_targets[i, 3] = target_states[i, 3]
        centers[i, 0] = noisy_targets[i, 0]
        centers[i, 1] = noisy_targets[i, 1]

    for i in range(40, 80):
        target_states[i, 0] = target_states[i-1, 0]
        target_states[i, 1] = target_states[i-1, 1] + dt*ref_vel[i - 1]
        target_states[i, 2] = ref_vel[i]
        target_states[i, 3] = np.pi/2
        noisy_targets[i, 0] = target_states[i, 0]
        noisy_targets[i, 1] = target_states[i, 1]
        noisy_targets[i, 2] = target_states[i, 3]
        noisy_targets[i, 3] = target_states[i, 3]
        centers[i, 0] = noisy_targets[i, 0]
        centers[i, 1] = noisy_targets[i, 1]
    for i in range(1, horizon):
        init_inputs[i - 1, 0] = (noisy_targets[i, 2] -
                                 noisy_targets[i - 1, 2])/dt
        init_inputs[i - 1, 1] = (noisy_targets[i, 3] -
                                 noisy_targets[i - 1, 3])/dt
    constraint = BubbleConstraint(horizon)
    radius = []
    vel_bounds = [0, 2]
    r = 0.8
    for i in range(horizon):
        radius.append(r)
    constraint.setup(centers, radius, vel_bounds)
    start = time.time()
    mpc_optimizer = sequential_QP_optimizer(
        system, constraint, noisy_targets, dt)
    mpc_optimizer.set_init_inputs(init_inputs)
    mpc_optimizer()
    end = time.time()
    print("Computation time: ", end - start)

    plt.figure(figsize=(8*1.1, 6*1.1))
    currentAxis = plt.gca()
    plt.title('SQP: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(mpc_optimizer.target_states[:, 0], mpc_optimizer.target_states[:, 1],
             '--r', label='Target', linewidth=2)
    plt.plot(mpc_optimizer.states[:, 0], mpc_optimizer.states[:, 1],
             '-+k', label='SQP', linewidth=1.0)
    plt.legend(loc='upper left')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    for i in range(mpc_optimizer.horizon):
        x = mpc_optimizer.target_states[i, 0]
        y = mpc_optimizer.target_states[i, 1]
        r = radius[i]
        currentAxis.add_patch(Circle((x, y), radius=r, alpha=1))
    # plt.figure(figsize=(8*1.1, 6*1.1))
    # plt.title('SQP: state vs. time.  ')
    # plt.plot(mpc_optimizer.states[:, 2], '-b', linewidth=1.0, label='speed')
    # plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
    # plt.ylabel('speed')
    # plt.figure(figsize=(8*1.1, 6*1.1))
    # plt.title('SQP: input vs. time.  ')
    # plt.plot(mpc_optimizer.inputs[:, 0], '-b', linewidth=1.0, label='turning rate')
    # plt.ylabel('inputs')
    plt.show()


def min_dist(line_x, line_y, x, y):
    min_dist = np.inf
    for i in range(line_x.shape[0]):
        dist = ((x - line_x[i])**2 + (y - line_y[i])**2)**0.5
        if dist < min_dist:
            min_dist = dist
    return min_dist


def random_example_2():

    x1 = -0.5
    y1 = 0.2
    x2 = 5
    y2 = 5.6
    x3 = 0
    y3 = 10

    line_x = np.concatenate(
        (np.linspace(x1, x2, num=100), np.linspace(x2, x3, num=100)), axis=0)
    line_y = np.concatenate(
        (np.linspace(y1, y2, num=100), np.linspace(y2, y3, num=100)), axis=0)

    horizon = 80
    target_states = np.zeros((horizon, 4))
    noisy_targets = np.zeros((horizon, 4))
    noisy_targets[0, 2] = 1
    target_states[0, 2] = 1
    centers = np.zeros((horizon, 2))
    ref_vel = np.ones(horizon)
    noisy = 0.4
    dt = 0.2
    curv = 0.2
    system = Car()
    system.set_dt(dt)
    system.set_cost(np.diag([50.0, 50.0, 10.0, 1.0]), np.diag([300.0, 1000.0]))
    system.Q_f = system.Q*horizon/2
    system.set_control_limit([-1, -0.3], [1, 0.3])
    init_inputs = np.zeros((horizon - 1, system.control_size))
    for i in range(1, horizon):
        target_states[i, 0] = target_states[i-1, 0] + \
            np.cos(target_states[i-1, 3])*dt*ref_vel[i - 1]
        target_states[i, 1] = target_states[i-1, 1] + \
            np.sin(target_states[i-1, 3])*dt*ref_vel[i - 1]
        target_states[i, 2] = ref_vel[i]
        target_states[i, 3] = target_states[i-1, 3] + curv*dt
        noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, noisy)
        noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, noisy)
        noisy_targets[i, 2] = ref_vel[i]
        noisy_targets[i, 3] = target_states[i, 3] + random.uniform(0, 0.1)
        centers[i, 0] = noisy_targets[i, 0]
        centers[i, 1] = noisy_targets[i, 1]
    for i in range(1, horizon):
        init_inputs[i - 1, 0] = (noisy_targets[i, 2] -
                                 noisy_targets[i - 1, 2])/dt
        init_inputs[i - 1, 1] = (noisy_targets[i, 3] -
                                 noisy_targets[i - 1, 3])/dt
    constraint = BubbleConstraint(horizon)
    radius = []
    vel_bounds = [0, 2]
    for i in range(horizon):
        r = min_dist(line_x, line_y, noisy_targets[i, 0], noisy_targets[i, 1])
        radius.append(r)
    constraint.setup(centers, radius, vel_bounds)
    start = time.time()
    mpc_optimizer = sequential_QP_optimizer(
        system, constraint, noisy_targets, dt)
    mpc_optimizer.set_init_inputs(init_inputs)
    mpc_optimizer()
    end = time.time()
    print("Computation time: ", end - start)

    plt.figure(figsize=(8*1.1, 6*1.1))
    currentAxis = plt.gca()
    plt.title('SQP: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot([x1, x2], [y1, y2], 'k', linewidth=3)
    plt.plot([x2, x3], [y2, y3], 'k', linewidth=3)
    plt.plot(mpc_optimizer.target_states[:, 0], mpc_optimizer.target_states[:, 1],
             '--r', label='Target', linewidth=2)
    plt.plot(mpc_optimizer.states[:, 0], mpc_optimizer.states[:, 1],
             '-+k', label='SQP', linewidth=1.0)
    plt.legend(loc='upper left')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    for i in range(mpc_optimizer.horizon):
        x = mpc_optimizer.target_states[i, 0]
        y = mpc_optimizer.target_states[i, 1]
        r = radius[i]
        currentAxis.add_patch(Circle((x, y), radius=r, alpha=1))
    plt.show()


if __name__ == "__main__":
    # random_example()
    # corner_example()
    random_example_2()

# plot(mpc_optimizer)
