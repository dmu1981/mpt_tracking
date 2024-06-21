import numpy as np

class Minimal_Variance_Fusion():
    def __init__(self, shape):
        self.shape = shape
        self.measurements = []
        self.cov_matrix = np.eye(shape) * 0.04  # initial cov-matrix as identity matrix
        self.measurement_noise_matrix = np.eye(shape) * 0.04  # doesn't change (for updating cov-matrix later)

    def reset(self, measurement):
        self.measurements = [np.array(measurement)]
        self.cov_matrix = np.eye(self.shape) * 0.04
        return self.measurements[0]

    def update(self, dt, measurement):
        z = np.array(measurement)
        self.measurements.append(z)
        
        # New estimate x as the arithmetic mean of all measurements
        self.x = np.mean(self.measurements, axis=0)

        # Update covariance matrix
        self.cov_matrix = self.measurement_noise_matrix / len(self.measurements)

        return self.x