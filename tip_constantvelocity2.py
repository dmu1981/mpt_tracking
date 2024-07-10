import numpy as np


class constantvelocity2_EKF:
    def __init__(self, shape):
        self.dim_z = shape
        self.dim_x = 4
        self.num_coordinates = 10

        Matrix_I = np.eye(shape)
        Matrix_I = np.vstack((Matrix_I, Matrix_I, Matrix_I, Matrix_I, Matrix_I))
        self.H = np.hstack((Matrix_I, np.zeros((self.num_coordinates, self.dim_z))))

        self.Q = np.zeros((self.dim_x, self.dim_x))

    def reset(self, measurement):
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        z = np.array(measurement[: self.num_coordinates]).reshape(self.num_coordinates)
        R = np.diag(measurement[self.num_coordinates :])
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Process noise covariance matrix, scaled by dt
        Q = np.eye(self.dim_x) * 0.01 * dt

        # Prediction (state & covariance)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # To posterior state
        # Update prediction
        self.x = self.x + K @ y

        # Update covariance matrix
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2]
