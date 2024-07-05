import numpy as np

#Problem 1
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

    def update(self, dt, measurement):
        measurement = np.array(measurement[:2])
        measurement_uncertainty = np.eye(2) * self.measurement_noise**2
        
        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)
        self.state = self.state + kalman_gain @ (measurement - self.state)
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

        return self.state
    
    #Problem 2
class RandomNoiseFilter:
    def __init__(self):
        self.state = np.zeros(2)  # Initial state (x, y)
        self.uncertainty = np.eye(2) * 500  # Initial uncertainty
        self.process_noise = 1e-5  # Process noise

    def reset(self, measurement):
        # Extract initial state from the measurement
        self.state = np.array(measurement[:2])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, measurement):
        # Extract the state measurement and the measurement noise covariance
        measurement = np.array(measurement[:2])
        Rt = np.array(measurement[2:]).reshape(2, 2)
        
        # Compute the Kalman gain
        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + Rt)
        
        # Update the state estimate
        self.state = self.state + kalman_gain @ (measurement - self.state)
        
        # Update the uncertainty
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

        return self.state   
    
    #Problem3
    
import numpy as np

class AngularFilter:
    def __init__(self):
        self.state = np.zeros(2)  # Initial state (x, y)
        self.uncertainty = np.eye(2) * 500  # Initial uncertainty
        self.measurement_noise = np.array([[0.01, 0], [0, 0.0025]])  # Measurement noise
        self.process_noise = 1e-5  # Process noise

    def reset(self, measurement):
        if isinstance(measurement, (np.ndarray, list, tuple)):
            measurement = tuple(measurement)
        else:
            raise ValueError(f"Measurement must be a tuple, list, or numpy array with two elements (r, phi), but got {measurement} of type {type(measurement)}")
        if len(measurement) != 2:
            raise ValueError(f"Measurement must have exactly two elements (r, phi), but got {len(measurement)} elements.")
        
        r, phi = measurement
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        self.state = np.array([x, y])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, measurement, *args, **kwargs):
        # Debug statement to print the received measurement
        print(f"Received measurement: {measurement} (type: {type(measurement)})")
        
        # Handle unexpected single float input gracefully
        if isinstance(measurement, float):
            raise ValueError(f"Expected a tuple, list, or numpy array with two elements (r, phi), but got a single float: {measurement}")

        if isinstance(measurement, (np.ndarray, list, tuple)):
            measurement = tuple(measurement)
        else:
            raise ValueError(f"Measurement must be a tuple, list, or numpy array with two elements (r, phi), but got {measurement} of type {type(measurement)}")

        if len(measurement) != 2:
            raise ValueError(f"Measurement must have exactly two elements (r, phi), but got {len(measurement)} elements.")

        r, phi = measurement
        measured_x = r * np.cos(phi)
        measured_y = r * np.sin(phi)
        measurement_cartesian = np.array([measured_x, measured_y])

        # Convert measurement noise from polar to Cartesian coordinates
        H = np.array([
            [np.cos(phi), -r * np.sin(phi)],
            [np.sin(phi),  r * np.cos(phi)]
        ])
        R_cartesian = H @ self.measurement_noise @ H.T

        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + R_cartesian)
        self.state = self.state + kalman_gain @ (measurement_cartesian - self.state)
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

        return self.state