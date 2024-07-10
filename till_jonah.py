import numpy as np


class Random:
    def __init__(self, shape):
        self.shape = shape
        self.x = np.zeros(self.shape)
        self.P = np.eye(self.shape)
        self.F = np.eye(self.shape)
        self.H = np.eye(self.shape)
        self.Q = np.zeros((self.shape, self.shape))

    def reset(self, measurement):
        self.x = np.zeros(self.shape)
        self.P = np.eye(self.shape)
        return self.x

    def update(self, dt, measurement):
        z = np.array(measurement[:2])
        R_t = np.array(measurement[2:]).reshape(2, 2)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R_t)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.shape) - K @ self.H) @ self.P

        return self.x


class Velocity2:
    def __init__(self, shape):
        self.shape = shape
        self.internal_shape = 4
        self.measurement_shape = 10
        self.x = np.zeros(self.internal_shape)
        self.P = np.eye(self.internal_shape)
        self.F = np.eye(self.internal_shape)
        self.H = np.zeros((self.measurement_shape, self.internal_shape))
        for i in range(5):
            self.H[2 * i, 0] = 1
            self.H[2 * i + 1, 1] = 1
        self.Q = np.zeros((self.internal_shape, self.internal_shape))

    def reset(self, measurement):
        self.x = np.zeros(self.internal_shape)
        self.P = np.eye(self.internal_shape)
        return self.x[: self.shape]

    def update(self, dt, measurement):
        z = np.array(measurement[: self.measurement_shape])
        R = np.diag(measurement[self.measurement_shape :])

        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        q = 0.1
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        self.Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ]
        )

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.internal_shape) - K @ self.H) @ self.P

        return self.x[: self.shape]


class ConstantTurn:
    def __init__(self, shape):
        self.shape = shape
        self.internal_shape = 4
        self.measurement_shape = 10
        self.x = np.zeros(self.internal_shape)
        self.P = np.eye(self.internal_shape)
        self.a = 0.001
        self.H = np.zeros((self.measurement_shape, self.internal_shape))
        for i in range(5):
            self.H[2 * i, 0] = 1
            self.H[2 * i + 1, 1] = 1
        self.Q = np.zeros((self.internal_shape, self.internal_shape))

    def reset(self, measurement):
        self.x = np.zeros(self.internal_shape)
        self.P = np.eye(self.internal_shape)
        return self.x[: self.shape]

    def update(self, dt, measurement):
        z = np.array(measurement[: self.measurement_shape])
        R = np.diag(measurement[self.measurement_shape :])

        cos_a_dt = np.cos(self.a * dt)
        sin_a_dt = np.sin(self.a * dt)
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, cos_a_dt, -sin_a_dt],
                [0, 0, sin_a_dt, cos_a_dt],
            ]
        )

        q = 0.1
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        self.Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ]
        )

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.internal_shape) - K @ self.H) @ self.P

        return self.x[: self.shape]


class Constant:
    def __init__(self, shape):
        self.shape = shape
        self.x = np.zeros(self.shape)
        self.P = np.eye(self.shape)
        self.F = np.eye(self.shape)
        self.H = np.eye(self.shape)
        self.Q = np.zeros((self.shape, self.shape))
        self.R = np.eye(self.shape) * 0.04

    def reset(self, measurement):
        self.x = np.zeros(self.shape)
        self.P = np.eye(self.shape)
        return self.x

    def update(self, dt, measurement):
        z = np.array(measurement)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.shape) - K @ self.H) @ self.P

        return self.x


class Angular:
    def __init__(self, shape):
        self.shape = shape
        self.measurement_shape = 2
        self.x = np.zeros(self.shape)
        self.P = np.eye(self.shape)
        self.R = np.array([[0.0100, 0.0000], [0.0000, 0.0025]])

    def reset(self, measurement):
        r = measurement[0]
        phi = measurement[1]
        self.x = np.array([r * np.cos(phi), r * np.sin(phi)])
        self.P = np.eye(self.shape)
        return self.x

    def h(self, x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        phi = np.arctan2(x[1], x[0])
        return np.array([r, phi])

    def H_jacobian(self, x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        if r == 0:
            r = 1e-6

        H = np.array([[x[0] / r, x[1] / r], [-x[1] / (r**2), x[0] / (r**2)]])
        return H

    def update(self, dt, measurement):
        z = np.array(measurement[: self.measurement_shape])

        x_pred = self.x
        P_pred = self.P

        H = self.H_jacobian(x_pred)
        z_pred = self.h(x_pred)

        S = H @ P_pred @ H.T + self.R

        K = P_pred @ H.T @ np.linalg.inv(S)

        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        self.x = x_pred + K @ y

        self.P = (np.eye(self.shape) - K @ H) @ P_pred

        return self.x


class Velocity:
    def __init__(self, shape):
        self.shape = shape
        self.state_shape = 4
        self.x = np.zeros(self.state_shape)
        self.P = np.eye(self.state_shape)
        self.F = np.eye(self.state_shape)
        self.H = np.zeros((self.shape, self.state_shape))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.zeros((self.state_shape, self.state_shape))
        self.R = np.eye(self.shape) * 0.04

    def reset(self, measurement):
        self.x = np.zeros(self.state_shape)
        self.P = np.eye(self.state_shape)
        return self.x[: self.shape]

    def update(self, dt, measurement):
        z = np.array(measurement)

        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        q = 0.1
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        self.Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ]
        )

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.state_shape) - K @ self.H) @ self.P

        return self.x[: self.shape]
