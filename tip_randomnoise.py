import numpy as np


# static object with random noise -> position is x(t) = x(t-1)
# noise z(t) = x(t) + et
# covariance matrix of measurements -> Rt = measurement[2:].reshape(2,2)
class KalmanFilter:
    def __init__(self, shape):
        self.shape = shape
        self.estimate = np.zeros(shape)
        self.cov_matrix = np.eye(shape)
        self.transition = np.eye(shape)
        self.measurements = np.eye(shape)
        self.identity_matrix = np.eye(shape)

    def reset(self, measurement):
        self.estimate = measurement[: self.shape]
        self.cov_matrix = np.eye(self.shape)
        return self.estimate

    def update(self, dt, measurement):
        measured_value = measurement[: self.shape]
        measurement_noise_covariance = measurement[self.shape :].reshape(
            self.shape, self.shape
        )
        measurement_residual = measured_value - np.dot(self.measurements, self.estimate)
        residual_covariance = (
            np.dot(
                self.measurements,
                np.dot(self.cov_matrix, self.measurements.T),
            )
            + measurement_noise_covariance
        )
        kalman_gain = np.dot(
            np.dot(self.cov_matrix, self.measurements.T),
            np.linalg.inv(residual_covariance),
        )
        self.estimate = self.estimate + np.dot(kalman_gain, measurement_residual)
        self.cov_matrix = np.dot(
            (self.identity_matrix - np.dot(kalman_gain, self.measurements)),
            self.cov_matrix,
        )
        return self.estimate
