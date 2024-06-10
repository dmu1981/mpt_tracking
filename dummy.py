import numpy as np

class DummyFilter:
    def __init__(self, shape):
        self.shape = shape

    def reset(self, measurement):
        return np.zeros(self.shape)

    def update(self, dt, measurement):
        return np.zeros(self.shape)

class ConstantPositionFilter:
    def __init__(self):
        # Initial state (location and velocity)
        self.x = np.zeros(2)  # Initial position (x, y)

        # State covariance matrix
        self.P = np.eye(2)

        # State transition matrix
        self.F = np.eye(2)

        # Measurement matrix
        self.H = np.eye(2)

        # Measurement covariance matrix (R)
        self.R = np.eye(2) * 0.04

        # Process covariance matrix (Q)
        self.Q = np.eye(2) * 0.01

    def reset(self, measurement):
        self.x = np.array(measurement[:2])
        self.P = np.eye(2)
        return self.x

    def update(self, dt, measurement):
        z = np.array(measurement[:2])

        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update step
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.x
