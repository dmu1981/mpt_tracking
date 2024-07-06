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
    def __init__(self, process_noise_scale=1e-5, initial_uncertainty=600):
        self.state = np.zeros(2)  # Initial state (x, y)
        self.uncertainty = np.eye(2) * initial_uncertainty  # Initial uncertainty
        self.process_noise_scale = process_noise_scale
        self.process_noise = np.eye(2) * self.process_noise_scale  # Process noise matrix

    def reset(self, measurement):
        # Extract initial state from the measurement
        self.state = np.array(measurement[:2])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, dt, measurement):
        # Predict step: Increase uncertainty due to process noise
        self.uncertainty += self.process_noise
        
        # Extract the state measurement and the measurement noise covariance
        z = np.array(measurement[:2])
        Rt = np.array(measurement[2:]).reshape(2, 2)
        
        # Compute the Kalman gain
        S = self.uncertainty + Rt
        K = self.uncertainty @ np.linalg.inv(S)
        
        # Update the state estimate
        y = z - self.state  # Measurement residual
        self.state = self.state + K @ y
        
        # Update the uncertainty
        I = np.eye(2)
        self.uncertainty = (I - K) @ self.uncertainty

        return self.state
    
    #Problem3
class AngularFilter:
    def __init__(self):
        self.state = np.zeros(2)  # Initial state (x, y)
        self.uncertainty = np.eye(2) * 500  # Initial uncertainty
        self.measurement_noise = np.array([[0.01, 0], [0, 0.0025]])  # Measurement noise
        self.process_noise = np.eye(2) * 1e-5  # Process noise

    def reset(self, measurement):

        r, phi = measurement
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        self.state = np.array([x, y])
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, dt, measurement, *args, **kwargs):

        
        r, phi = measurement
        
        # Predict step: Increase uncertainty due to process noise
        self.uncertainty += self.process_noise
        
        # Convert polar coordinates to Cartesian coordinates
        measured_x = r * np.cos(phi)
        measured_y = r * np.sin(phi)
        measurement_cartesian = np.array([measured_x, measured_y])
        
        # Convert measurement noise from polar to Cartesian coordinates
        H = np.array([
            [np.cos(phi), -r * np.sin(phi)],
            [np.sin(phi),  r * np.cos(phi)]
        ])
        R_cartesian = H @ self.measurement_noise @ H.T

        # Compute the Kalman gain
        S = self.uncertainty + R_cartesian
        K = self.uncertainty @ np.linalg.inv(S)
        
        # Update the state estimate
        y = measurement_cartesian - self.state  # Measurement residual
        self.state = self.state + K @ y
        
        # Update the uncertainty
        I = np.eye(2)
        self.uncertainty = (I - K) @ self.uncertainty

        return self.state
    
    
#problem 6 
class ConstantTurnFilter:
    def __init__(self, turn_rate, initial_uncertainty=500, process_noise_scale=1e-5):
        self.turn_rate = turn_rate
        self.process_noise_scale = process_noise_scale
        self.initial_uncertainty = initial_uncertainty
        self.state = np.zeros(4)  # Initial state (x, y, vx, vy)
        self.uncertainty = np.eye(4) * initial_uncertainty

    def reset(self, measurement):
        self.state[:2] = np.mean(measurement[:10].reshape(5, 2), axis=0)
        self.state[2:] = 0  # Initial velocities are set to 0
        self.uncertainty = np.eye(4) * self.initial_uncertainty
        return self.state[:2]  # Return only the position (x, y)

    def update(self, dt, measurement):
        # Extract measurements and their uncertainties
        measurements = measurement[:10].reshape(5, 2)
        uncertainties = measurement[10:].reshape(5, 2)

        # Average measurement and uncertainty
        avg_measurement = np.mean(measurements, axis=0)
        avg_uncertainty = np.mean(uncertainties, axis=0)
        
        # State transition matrix
        a_cos = np.cos(self.turn_rate * dt)
        a_sin = np.sin(self.turn_rate * dt)
        #Die Matrix berechnet den neuen Zustand basierend auf dem alten Zustand und der Drehung.
        F = np.array([
            [1, 0, dt * a_cos, -dt * a_sin],
            [0, 1, dt * a_sin,  dt * a_cos],
            [0, 0,        a_cos,        -a_sin],
            [0, 0,        a_sin,         a_cos]
        ])

        # Process noise
        Q = np.eye(4) * self.process_noise_scale

        # Predict step
        #Der Zustand und die Unsicherheit werden vorhergesagt.
        self.state = F @ self.state
        self.uncertainty = F @ self.uncertainty @ F.T + Q

        # Measurement noise
        #Das Messrauschen wird berechnet.
        R = np.diag(np.hstack((avg_uncertainty, avg_uncertainty))) ** 2

        # Kalman gain
        H = np.eye(4)
        S = H @ self.uncertainty @ H.T + R
        K = self.uncertainty @ H.T @ np.linalg.inv(S)

        # Update step
        z = np.hstack((avg_measurement, [0, 0]))  # Measurement vector
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.uncertainty = (np.eye(4) - K @ H) @ self.uncertainty

        return self.state[:2]  # Return only the position (x, y)