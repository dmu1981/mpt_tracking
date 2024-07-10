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
