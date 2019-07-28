import numpy as np
import random
from matplotlib import pyplot as plt
import timeit
from systems import *
from ilqr import iterative_LQR


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


def example_jerk():
    horizon = 100
    target_states = np.zeros((horizon, 5))
    noisy_targets = np.zeros((horizon, 5))
    ref_vel = np.zeros(horizon)
    ref_acc = np.zeros(horizon)
    dt = 0.2
    curv = 0.1
    a = 1.5
    v_max = 11

    car_system = CarAcceleration()
    car_system.set_dt(dt)
    car_system.set_cost(
        np.diag([50.0, 50.0, 1000.0, 1000, 0.0]), np.diag([3000.0, 1000.0]))
    car_system.set_control_limit([-6, -0.2], [6, 0.2])
    init_inputs = np.zeros((horizon - 1, car_system.control_size))

    for i in range(40, horizon):
        if ref_vel[i - 1] > v_max:
            a = 0
        ref_acc[i] = a
        ref_vel[i] = ref_vel[i - 1] + a*dt

    for i in range(1, horizon):
        target_states[i, 0] = target_states[i - 1, 0] + \
            np.cos(target_states[i - 1, 4])*dt*ref_vel[i - 1]
        target_states[i, 1] = target_states[i - 1, 1] + \
            np.sin(target_states[i - 1, 4])*dt*ref_vel[i - 1]
        target_states[i, 2] = ref_vel[i]
        target_states[i, 3] = ref_acc[i]
        target_states[i, 4] = target_states[i - 1, 4] + curv*dt
        noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 1)
        noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 1)
        noisy_targets[i, 2] = target_states[i, 2]
        noisy_targets[i, 3] = target_states[i, 3]
        noisy_targets[i, 4] = target_states[i, 4] + random.uniform(0, 0.1)

    for i in range(1, horizon):
        init_inputs[i - 1, 0] = (noisy_targets[i, 3] -
                                 noisy_targets[i - 1, 3])/dt
        init_inputs[i - 1, 1] = (noisy_targets[i, 4] -
                                 noisy_targets[i - 1, 4])/dt

    optimizer = iterative_LQR(
        car_system, noisy_targets, dt)
    optimizer.inputs = init_inputs

    start_time = timeit.default_timer()
    optimizer()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)
    plt.figure
    plt.title('Acceleration')
    plt.plot(optimizer.states[:, 3], '--r', label='Acceleration', linewidth=2)

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(noisy_targets[:, 0],
             noisy_targets[:, 1], '--r', label='Target', linewidth=2)
    plt.plot(optimizer.states[:, 0], optimizer.states[:, 1],
             '-+k', label='iLQR', linewidth=1.0)
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
             linewidth=1.0, label='Jerk')
    plt.plot(optimizer.inputs[:, 1], '-b',
             linewidth=1.0, label='Turning rate')
    plt.ylabel('Jerk and turning rate input')
    plt.legend
    plt.show()


def example_dubins():
    horizon = 200
    target_states = np.zeros((horizon, 3))
    noisy_targets = np.zeros((horizon, 3))
    dt = 0.2
    v = 1.0
    curv = 0.1
    for i in range(1, horizon):
        target_states[i, 0] = target_states[i - 1, 0] + \
            np.cos(target_states[i - 1, 2])*v*dt
        target_states[i, 1] = target_states[i - 1, 1] + \
            np.sin(target_states[i - 1, 2])*v*dt
        target_states[i, 2] = target_states[i - 1, 2] + v*curv*dt
        noisy_targets[i, 0] = target_states[i, 0] + random.uniform(0, 1)
        noisy_targets[i, 1] = target_states[i, 1] + random.uniform(0, 1)
        noisy_targets[i, 2] = target_states[i, 2] + random.uniform(0, 1)
    dubins_car_system = DubinsCar()
    dubins_car_system.set_dt(dt)
    dubins_car_system.set_cost(
        100*np.diag([1.0, 1.0, 1.0]), np.diag([10.0, 100.0]))
    dubins_car_system.set_control_limit([0, -0.2], [2, 0.2])
    optimizer = iterative_LQR(
        dubins_car_system, noisy_targets, dt)

    start_time = timeit.default_timer()
    optimizer()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.suptitle('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(optimizer.target_states[:, 0],
             optimizer.target_states[:, 1], '--r', label='Target', linewidth=2)
    plt.plot(optimizer.states[:, 0], optimizer.states[:, 1],
             '-+k', label='iLQR', linewidth=1.0)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.show()


if __name__ == '__main__':
    # example_dubins()
    # example_jerk()
    example_acc()
