import numpy as np

"""
The Kalman filter is a recursive algorithm that estimates the state of a dynamic system over time.
It combines measurements (which are often noisy and inaccurate) with a model of the system to provide a more accurate estimate of the state.
"""

class KalmanFilter:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state_estimate = np.zeros(state_dim)  # initial state (location and velocity)
        self.uncertainty_covariance = np.eye(state_dim[0])  # uncertainty covariance
        self.state_transition = np.eye(state_dim[0])  # state transition matrix
        self.measurement_matrix = np.eye(state_dim[0])  # measurement matrix
        self.measurement_uncertainty = np.eye(state_dim[0]) * 0.04  # measurement uncertainty (noise) matrix
        self.identity_matrix = np.eye(state_dim[0])  # identity matrix
        
    def reset(self, measurement):
        self.state_estimate = measurement[:2]  # initialize state with first measurement
        self.uncertainty_covariance = np.eye(2)  # reset uncertainty covariance
        return self.state_estimate
    
    def update(self, dt, measurement):
        measured_value = measurement[:2]  # measurement vector / measured value
        measurement_residual = measured_value - np.dot(self.measurement_matrix, self.state_estimate)  # measurement residual: difference between measured and estimated values
        residual_covariance = np.dot(self.measurement_matrix, np.dot(self.uncertainty_covariance, self.measurement_matrix.T)) + self.measurement_uncertainty  # residual covariance: uncertainty of measurement residual
        kalman_gain = np.dot(np.dot(self.uncertainty_covariance, self.measurement_matrix.T), np.linalg.inv(residual_covariance))  # Kalman gain: weighting of measurement residual to update state
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, measurement_residual)  # update of state estimate: addition of weighted measurement residual
        self.uncertainty_covariance = np.dot((self.identity_matrix - np.dot(kalman_gain, self.measurement_matrix)), self.uncertainty_covariance)  # updated estimate covariance: reduction of uncertainty
        return self.state_estimate

class KalmanFilterRandomNoise:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state_estimate = np.zeros(state_dim)
        self.uncertainty_covariance = np.eye(state_dim)
        self.state_transition = np.eye(state_dim)
        self.measurement_matrix = np.eye(state_dim)
        self.identity_matrix = np.eye(state_dim)

    def reset(self, measurement):
        self.state_estimate = measurement[:self.state_dim]
        self.uncertainty_covariance = np.eye(self.state_dim)
        return self.state_estimate

    def update(self, dt, measurement):
        measured_value = measurement[:self.state_dim]
        measurement_noise_covariance = measurement[self.state_dim:].reshape(self.state_dim, self.state_dim)
        measurement_residual = measured_value - np.dot(self.measurement_matrix, self.state_estimate)
        residual_covariance = np.dot(self.measurement_matrix, np.dot(self.uncertainty_covariance, self.measurement_matrix.T)) + measurement_noise_covariance
        kalman_gain = np.dot(np.dot(self.uncertainty_covariance, self.measurement_matrix.T), np.linalg.inv(residual_covariance))
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, measurement_residual)
        self.uncertainty_covariance = np.dot((self.identity_matrix - np.dot(kalman_gain, self.measurement_matrix)), self.uncertainty_covariance)
        return self.state_estimate