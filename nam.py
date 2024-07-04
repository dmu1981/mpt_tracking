import numpy as np


class NamFilter:
    def __init__(self):
        self.state = np.zeros(2)
        self.uncertainty = np.eye(2) * 500
        self.measurement_noise = 0.2
        self.process_noise = 1e-5

    def reset(self, measurement):
        self.state = np.array(measurement[:2])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, measurement):
        measurement = np.array(measurement[:2])
        measurement_uncertainty = np.eye(2) * self.measurement_noise**2
        
        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)
        self.state = self.state + kalman_gain @ (measurement - self.state)
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

        return self.state