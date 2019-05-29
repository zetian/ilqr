import numpy as np
import random
from matplotlib import pyplot as plt
import timeit
from systems import Car, DubinsCar
from ilqr import iterative_LQR_quadratic_cost
# from constraints import CircleConstraintForCar

def example_1():
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
        np.diag([50.0, 50.0, 1000.0, 0.0]), np.diag([1000.0, 1000.0]))
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
    myiLQR.inputs = init_inputs

    start_time = timeit.default_timer()
    myiLQR()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)
    plt.figure
    plt.title('theta')
    plt.plot(myiLQR.states[3, :], '--r', label='theta', linewidth=2)

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


def example_2():
    ntimesteps = 200
    target_states = np.zeros((3, ntimesteps))
    noisy_targets = np.zeros((3, ntimesteps))
    dt = 0.2
    v = 1.0
    curv = 0.1
    for i in range(1, ntimesteps):
        target_states[0, i] = target_states[0, i - 1] + \
            np.cos(target_states[2, i - 1])*v*dt
        target_states[1, i] = target_states[1, i - 1] + \
            np.sin(target_states[2, i - 1])*v*dt
        target_states[2, i] = target_states[2, i - 1] + v*curv*dt
        noisy_targets[0, i] = target_states[0, i] + random.uniform(0, 1)
        noisy_targets[1, i] = target_states[1, i] + random.uniform(0, 1)
        noisy_targets[2, i] = target_states[2, i] + random.uniform(0, 1)

    dubins_car_system = DubinsCar()
    dubins_car_system.set_dt(dt)
    dubins_car_system.set_cost(
        100*np.diag([1.0, 1.0, 1.0]), np.diag([10.0, 100.0]))
    dubins_car_system.set_control_limit(np.array([[0, 2.0], [-0.2, 0.2]]))
    myiLQR = iterative_LQR_quadratic_cost(
        dubins_car_system, noisy_targets, dt)

    start_time = timeit.default_timer()
    myiLQR()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.suptitle('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(myiLQR.target_states[0, :], myiLQR.target_states[1,
                                                              :], '--r', label='Target', linewidth=2)
    plt.plot(myiLQR.states[0, :], myiLQR.states[1, :],
             '-+b', label='iLQR', linewidth=1.0)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.show()

def example_3():
    ntimesteps = 100
    target_states = np.zeros((4, ntimesteps))
    noisy_targets = np.zeros((4, ntimesteps))
    ref_vel = np.zeros(ntimesteps)
    dt = 0.2
    curv = 0.1
    a = 1.5
    v_max = 11
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
    car_system = Car()
    car_system.set_dt(dt)
    car_system.set_cost(
        np.diag([50.0, 50.0, 1000.0, 0.0]), np.diag([1000.0, 1000.0]))

    constraint = CircleConstraintForCar(np.array([0, 40]), 10.0, car_system)

if __name__ == '__main__':
    example_1()
    # example_2()
