import numpy as np

state_dim = 2
measurement_dim = 2

class RandomNoise:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros((state_dim, 1))  # Initial state vector
        self.P = np.eye(state_dim) * 1.0  # Initial covariance matrix with some uncertainty
        self.F = np.eye(state_dim)  # State transition matrix
        self.H = np.eye(measurement_dim, state_dim)  # Observation matrix
        self.Q = np.eye(state_dim) * 1e-5  # Small process noise to avoid overconfidence

    def reset(self, measurement):
        self.x = measurement[:self.state_dim].reshape(self.state_dim, 1)
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x.flatten()

    def update(self, dt, measurement):
        # Extract the measured positions and the measurement noise covariance matrix
        z = measurement[:self.measurement_dim].reshape(self.measurement_dim, 1)
        Rt = measurement[2:].reshape(2, 2)
        
        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update step
        S = self.H @ P_pred @ self.H.T + Rt  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)  # Kalman gain with numerical stability
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x.flatten()


