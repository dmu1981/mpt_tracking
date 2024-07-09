import numpy as np

class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(2)
        self.uncertainty = np.eye(2)
        self.measurement_noise = 0.2
        self.process_noise = 1e-8

    def reset(self, measurement):
        self.state = np.array(measurement[:2])
        self.uncertainty = np.eye(2)
        return self.state

    def predict(self):
        F = np.eye(2)  # State transition model (identity for static object)
        Q = np.eye(2) * self.process_noise  # Process noise covariance
        self.state = F @ self.state  # State prediction
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def calculate_kalman_gain(self, measurement_uncertainty):
        return self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)

    def update_state(self, measurement, kalman_gain):
        self.state = self.state + kalman_gain @ (measurement - self.state)

    def update_uncertainty(self, kalman_gain):
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

    def update(self, dt, measurement):
        self.predict()

        # Measurement update step
        measurement = np.array(measurement[:2])
        measurement_uncertainty = np.eye(2) * self.measurement_noise**2

        # Calculate Kalman gain
        kalman_gain = self.calculate_kalman_gain(measurement_uncertainty)

        # Update state
        self.update_state(measurement, kalman_gain)

        # Update uncertainty
        self.update_uncertainty(kalman_gain)

        return self.state

