import numpy as np


class KalmanFilter:
    def __init__(self, shape):
        # We need 4 variables, position x, y and velocity x, y: [x, y, vx, vy]
        # Measurement only has the coordinates [x, y]
        self.dim_x = 4
        self.dim_z = shape

        # Initial state vector [x, y, vx, vy]
        self.x = np.zeros((self.dim_x, 1))

        # Covariance matrix (with large initial uncertainty)
        self.P = np.eye(self.dim_x) * 100

        # Measurement noise covariance matrix
        self.R = np.eye(self.dim_z) * 0.04

        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y

    def reset(self, measurement):
        self.x[:2] = np.array(measurement).reshape((2, 1))  # Set back to start positon
        self.x[2:] = 0  # Set back to unknown velocity
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # To predicted state
        # Process model matrix
        F = np.eye(self.dim_x)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt

        # Process noise covariance matrix, scaled by dt
        Q = np.eye(self.dim_x) * 0.01 * dt

        # Prediction (state & covariance)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Measurement
        # Innovation
        z = np.array(measurement).reshape((2, 1))
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # To posterior state
        # Update prediction
        self.x = self.x + K @ y

        # Update covariance matrix
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2].flatten()
