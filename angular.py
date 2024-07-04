import numpy as np

class AngularFilter:
    def __init__(self):
        self.state = np.zeros(2)  # Initialer Zustand (x, y)
        self.uncertainty = np.eye(2) * 500  # Initiale Unsicherheit
        self.measurement_noise = np.array([[0.01, 0], [0, 0.0025]])  # Messrauschen
        self.process_noise = 1e-5  # Prozessrauschen

    def reset(self, measurement):
        print(f"Resetting with measurement: {measurement} (type: {type(measurement)})")
        if isinstance(measurement, (np.ndarray, list, tuple)):
            measurement = tuple(measurement)
        if not isinstance(measurement, tuple) or len(measurement) != 2:
            raise ValueError(f"Measurement must be a tuple, list, or numpy array with two elements (r, phi), but got {measurement}.")

        r, phi = measurement
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        self.state = np.array([x, y])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, measurement, *args, **kwargs):
        print(f"Received measurement: {measurement} (type: {type(measurement)})")
        if isinstance(measurement, (np.ndarray, list, tuple)):
            measurement = tuple(measurement)
        if not isinstance(measurement, tuple) or len(measurement) != 2:
            raise ValueError(f"Measurement must be a tuple, list, or numpy array with two elements (r, phi), but got {measurement}.")

        # Umwandlung der Messung von Polarkoordinaten in kartesische Koordinaten
        r, phi = measurement
        x_meas = r * np.cos(phi)
        y_meas = r * np.sin(phi)
        measurement_cartesian = np.array([x_meas, y_meas])

        # Kalman-Filter Berechnungen
        measurement_uncertainty = self.measurement_noise
        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)
        self.state = self.state + kalman_gain @ (measurement_cartesian - self.state)
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

        return self.state

    def get_state(self):
        return self.state

    def get_uncertainty(self):
        return self.uncertainty
