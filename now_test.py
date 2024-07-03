import numpy as np

class ConstantVelocityKalmanFilter:
    def __init__(self):
        self.state_dim = 4
        self.measurement_dim = 2
        self.x = np.zeros((self.state_dim, 1))  # Initial state vector [x, y, vx, vy]
        self.P = np.eye(self.state_dim)  # Initial covariance matrix with some uncertainty
        self.F = np.eye(self.state_dim)  # State transition matrix (to be updated with dt)
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Observation matrix
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.eye(self.state_dim) * 1e-6  # Small process noise to avoid overconfidence
        self.R = np.eye(self.measurement_dim) * 0.04  # Measurement noise covariance matrix

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0  # Initial velocities set to zero
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # Update the state transition matrix with the new dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement update step
        z = measurement[:2].reshape(2, 1)
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x[:2].flatten()
