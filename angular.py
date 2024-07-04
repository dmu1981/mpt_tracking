# angular.py
import numpy as np

class AdngularFilter:
    def __init__(self):
        self.state = np.zeros(2)
        self.uncertainty = np.eye(2) * 500
        self.measurement_noise = np.array([[0.01, 0], [0, 0.0025]])
        self.process_noise = 1e-5

    def reset(self, measurement):
        r, phi = measurement
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        self.state = np.array([x, y])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, measurement):
        print(f"Update called with measurement: {measurement}")  # Logging
        r, phi = measurement
        x_meas = r * np.cos(phi)
        y_meas = r * np.sin(phi)
        measurement_cartesian = np.array([x_meas, y_meas])

        measurement_uncertainty = self.measurement_noise

        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)
        self.state = self.state + kalman_gain @ (measurement_cartesian - self.state)
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

        return self.state

