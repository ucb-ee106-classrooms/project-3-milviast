import matplotlib.pyplot as plt
import numpy as np
import time
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][1] is the thrust of the quadrotor
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is translational position in x (m),
            x[i][1] is translational position in z (m),
            x[i][2] is the bearing (rad) of the quadrotor
            x[i][3] is translational velocity in x (m/s),
            x[i][4] is translational velocity in z (m/s),
            x[i][5] is angular velocity (rad/s),
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][1] is distance to the landmark (m)
            y[i][2] is relative bearing (rad) w.r.t. the landmark
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.

    Notes
    ----------
        The landmark is positioned at (0, 5, 5).
    """
    # noinspection PyTypeChecker
    def __init__(self, is_noisy=False):
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.t = []
        self.fig, self.axd = plt.subplot_mosaic(
            [['xz', 'phi'],
             ['xz', 'x'],
             ['xz', 'z']], figsize=(20.0, 10.0))
        self.ln_xz, = self.axd['xz'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xz_hat, = self.axd['xz'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_z, = self.axd['z'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_z_hat, = self.axd['z'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'

        # Defined in dynamics.py for the dynamics model
        # m is the mass and J is the moment of inertia of the quadrotor 
        self.gr = 9.81 
        self.m = 0.92
        self.J = 0.0023
        # These are the X, Y, Z coordinates of the landmark
        self.landmark = (0, 5, 5)

        # This is a (N,12) where it's time, x, u, then y_obs 
        if is_noisy:
            with open('noisy_data.npy', 'rb') as f:
                self.data = np.load(f)
        else:
            with open('data.npy', 'rb') as f:
                self.data = np.load(f)

        self.dt = self.data[-1][0]/self.data.shape[0]


    def run(self):
        for i, data in enumerate(self.data):
            self.t.append(np.array(data[0]))
            self.x.append(np.array(data[1:7]))
            self.u.append(np.array(data[7:9]))
            self.y.append(np.array(data[9:12]))
            if i == 0:
                self.x_hat.append(self.x[-1])
            else:
                self.update(i)
        average_runtime = np.mean(self.update_runtimes)
        print(f"Average update runtime: {average_runtime:.6f} seconds")
        print('Mean Squared Error: ', np.mean(np.square(np.array(self.x) - np.array(self.x_hat))))
        return self.x_hat

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xz'].set_title(self.canvas_title)
        self.axd['xz'].set_xlabel('x (m)')
        self.axd['xz'].set_ylabel('z (m)')
        self.axd['xz'].set_aspect('equal', adjustable='box')
        self.axd['xz'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].set_xlabel('t (s)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].set_xlabel('t (s)')
        self.axd['x'].legend()
        self.axd['z'].set_ylabel('z (m)')
        self.axd['z'].set_xlabel('t (s)')
        self.axd['z'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xzline(self.ln_xz, self.x)
        self.plot_xzline(self.ln_xz_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_zline(self.ln_z, self.x)
        self.plot_zline(self.ln_z_hat, self.x_hat)

    def plot_xzline(self, ln, data):
        if len(data):
            x = [d[0] for d in data]
            z = [d[1] for d in data]
            ln.set_data(x, z)
            self.resize_lim(self.axd['xz'], x, z)

    def plot_philine(self, ln, data):
        if len(data):
            t = self.t
            phi = [d[2] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = self.t
            x = [d[0] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_zline(self, ln, data):
        if len(data):
            t = self.t
            z = [d[1] for d in data]
            ln.set_data(t, z)
            self.resize_lim(self.axd['z'], t, z)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])

class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ python drone_estimator_node.py --estimator oracle_observer
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ python drone_estimator_node.py --estimator dead_reckoning
    """
    def __init__(self, is_noisy=False):
        super().__init__(False)
        self.index = 0
        self.previousState = 0
        self.canvas_title = 'Dead Reckoning'
        self.update_runtimes = []

    def update(self, _):
        start_time = time.time()  # Start timing

        if len(self.x_hat) > 0 and len(self.u) > self.index:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation
            if self.index == 0:
                self.previousState = self.x[0]
            lastPhi = self.previousState[2]
            model = np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [-(np.sin(lastPhi) / self.m), 0],
                        [(np.cos(lastPhi) / self.m), 0],
                        [0, (1 / self.J)]])
            inputs = np.array([self.u[self.index][0], self.u[self.index][1]])

            # print("Model: ", model)
            # print("Model Shape: ", model.shape)
            # print("Inputs: ", inputs)
            # print("Inputs Shape: ", inputs.shape)

            nextState = (model @ inputs) + np.array([self.previousState[3], self.previousState[4], self.previousState[5], 0, -self.gr, 0])

            # print("nextState: ", nextState)
            # print("nextState Shape: ", nextState.shape)
            # print("nextState type: ", type(nextState))

            # print("Matrix Multiplication: ", model @ inputs)

            stateEstimate = self.previousState + nextState * self.dt

            # print("State Estimate: ", stateEstimate)
            # print("SE Shape: ", stateEstimate.shape)

            self.previousState = stateEstimate            
            self.x_hat.append(stateEstimate)
            
            self.index += 1

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        self.update_runtimes.append(elapsed_time)  # Store runtime

# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.
            landmark[2] is the z coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ python drone_estimator_node.py --estimator extended_kalman_filter
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Extended Kalman Filter'
        self.A = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.B = None
        self.C = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        self.Q = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.R = np.array([[1, 0],
                           [0, 1]])
        self.P = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.previous_state = None
        self.index = 0
        self.update_runtimes = []

    # noinspection DuplicatedCode
    def update(self, i):
        start_time = time.time()

        if len(self.x_hat) > 0: # and self.x_hat[-1][0] < self.x[-1][0]:
            # You may use self.u, self.y, and self.x[0] for estimation
            if self.index== 0:
                self.previous_state = self.x[0]
            
            # State extrapolation
            next_x = self.g(self.previous_state, self.u[-1])
            # print("next_x: ", next_x)

            # Dynamics linearization
            At = self.approx_A(self.previous_state, self.u[-1])
            # print("At: ", At)

            # Covariance extrapolation
            Pt1 = At @ self.P @ self.A.T + self.Q
            # print("Pt1: ", Pt1)

            # Measurement linearization
            Ct = self.approx_C(next_x)
            # print("Ct: ", Ct)

            # Kalman gain
            Kt1 = Pt1 @ Ct.T @ np.linalg.inv(Ct @ Pt1 @ Ct.T + self.R)
            # print("Kt1: ", Kt1)
            # print("Kt1 Shape: ", Kt1.shape)

            # print("self.y[-1]: ", np.array(self.y[-1]))
            # print("self.y[-1] shape: ", np.array(self.y[-1]).shape)
            # print("self.h(next_x, None): ", self.h(next_x, None))
            # print("h shape: ", self.h(next_x, None).shape)
            # print("Kt1 @ (self.y[-1] - self.h(next_x, None): ", Kt1 @ (self.y[-1] - self.h(next_x, None)))
            # print("Mult shape: ", (Kt1 @ (self.y[-1] - self.h(next_x, None))).shape)

            # State update
            next_state = next_x + Kt1 @ (self.y[-1] - self.h(next_x, None)) # double check y in self.h

            # Covariance update
            self.P = (np.eye(6) - (Kt1 @ self.C)) @ Pt1

            # State estimate update
            state_estimate = np.zeros(6)
            # print("State Estimate: ", state_estimate)
            # print("Next State: ", next_state)
            state_estimate = next_state

            self.previous_state = state_estimate
            self.x_hat.append(state_estimate)
            self.index += 1
        
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        self.update_runtimes.append(elapsed_time)  # Store runtime

    def g(self, x, u):
        # Dynamics model
        f = np.hstack((x[3:], [0, -self.gr, 0])) + np.array(([0, 0],
                                                             [0, 0],
                                                             [0, 0],
                                                             [-np.sin(x[2])/self.m, 0],
                                                             [np.cos(x[2])/self.m, 0],
                                                             [0, 1/self.J])) @ u 
        
        return x + f * self.dt

    def h(self, x, y_obs):
        # Measurement model
        h = np.array(([np.sqrt((self.landmark[0] - x[0])**2 + self.landmark[1]**2 + (self.landmark[2] - x[1])**2), x[2]]))
        return h

    def approx_A(self, x, u):
        # Linear approx of g w.r.t. x
        A = np.eye(6)
        A[0, 3] = self.dt
        A[1, 4] = self.dt
        A[2, 5] = self.dt
        A[3, 2] = -np.cos(x[2]) * u[0] * self.dt / self.m
        A[4, 2] = -np.sin(x[2]) * u[0] * self.dt / self.m
        return A
    
    def approx_C(self, x):
        # Linear approx of h w.r.t. x
        distance = np.sqrt((self.landmark[0] - x[0])**2 + self.landmark[1]**2 + (self.landmark[2] - x[1])**2)
        C = np.zeros((2, 6))
        C[0, 0] = (x[0] - self.landmark[0]) / distance
        C[0, 1] = (x[1] - self.landmark[1]) / distance
        return C
