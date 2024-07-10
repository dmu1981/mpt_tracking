import numpy as np

class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(2)
        self.uncertainty = np.eye(4)
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

class FilterRandomNoise():
    def __init__(self, process_noise=1e-6):
        self.state = np.zeros(2)  # Initial state
        self.uncertainty = np.eye(2)  # Initial uncertainty
        self.process_noise = process_noise  # Process noise covariance

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

    def update(self,dt, measurement):
        self.predict()

        # Measurement update step
        measurement_position = np.array(measurement[:2])
        measurement_covariance = np.array(measurement[2:]).reshape(2, 2)

        # Calculate Kalman gain
        kalman_gain = self.calculate_kalman_gain(measurement_covariance)

        # Update state
        self.update_state(measurement_position, kalman_gain)

        # Update uncertainty
        self.update_uncertainty(kalman_gain)

        return self.state
    
    
class AngularKalmanFilter:
    def __init__(self):
        self.state = np.zeros(2)  # Initial state (x, y)
        self.uncertainty = np.eye(2)  # Initial uncertainty
        self.process_noise = 1e-8  # Very small process noise since the object is static
        self.measurement_noise_r = 0.1  # Noise standard deviation for r
        self.measurement_noise_phi = 0.05  # Noise standard deviation for phi

    def reset(self, measurement):
        r, phi = measurement
        self.state = self.polar_to_cartesian(r, phi)
        self.uncertainty = np.eye(2)
        return self.state

    def predict(self):
        F = np.eye(2)  # State transition model (identity for static object)
        Q = np.eye(2) * self.process_noise  # Process noise covariance
        self.state = F @ self.state  # State prediction
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def update(self, dt, measurement):
        self.predict()

        # Convert polar measurement to Cartesian coordinates
        r, phi = measurement
        measurement_cartesian = self.polar_to_cartesian(r, phi)

        # Measurement noise covariance in Cartesian coordinates
        R = np.array([[0.0100, 0.0000], [0.0000, 0.0025]])

        # Calculate Kalman gain
        kalman_gain = self.calculate_kalman_gain(R)

        # Update state
        self.update_state(measurement_cartesian, kalman_gain)

        # Update uncertainty
        self.update_uncertainty(kalman_gain)

        return self.state

    def calculate_kalman_gain(self, measurement_uncertainty):
        return self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)

    def update_state(self, measurement, kalman_gain):
        self.state = self.state + kalman_gain @ (measurement - self.state)

    def update_uncertainty(self, kalman_gain):
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

    @staticmethod
    def polar_to_cartesian(r, phi):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.array([x, y])    


class ConstantVelocityKalmanFilter:
    def __init__(self, process_noise=1e-6, measurement_noise=0.69):
        self.dt = 1  # Default time step, will be updated dynamically
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.uncertainty = np.eye(4)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def reset(self, measurement):
        self.state[:2] = np.array(measurement[:2])
        self.state[2:] = 0  # Initial velocity is unknown, set to 0
        self.uncertainty = np.eye(4) / 2
        return self.state[:2]  # Return only the position part

    def predict(self, dt):
        self.dt = dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        Q = np.eye(4) * self.process_noise
        
        self.state = F @ self.state
        self.uncertainty = F @ self.uncertainty @ F.T + Q

    def calculate_kalman_gain(self, measurement_uncertainty):
        H = np.eye(2, 4)  # Measurement matrix
        return self.uncertainty @ H.T @ np.linalg.inv(H @ self.uncertainty @ H.T + measurement_uncertainty)

    def update_state(self, measurement, kalman_gain):
        H = np.eye(2, 4)  # Measurement matrix
        self.state = self.state + kalman_gain @ (measurement - H @ self.state)

    def update_uncertainty(self, kalman_gain):
        H = np.eye(2, 4)  # Measurement matrix
        self.uncertainty = (np.eye(4) - kalman_gain @ H) @ self.uncertainty

    def update(self, dt, measurement):
        self.predict(dt)
        
        measurement = np.array(measurement[:2])
        measurement_uncertainty = np.eye(2) * self.measurement_noise**2

        kalman_gain = self.calculate_kalman_gain(measurement_uncertainty)

        self.update_state(measurement, kalman_gain)

        self.update_uncertainty(kalman_gain)

        return self.state[:2]
    
class ConstantVelocityMultiMeasurementKalmanFilter:
    def __init__(self, process_noise=2e-4, measurement_noise=0.00001):
         # Default time step, will be updated dynamically
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.uncertainty = np.eye(2)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def reset(self, measurement):
        self.state[:2] = np.mean(measurement[:10].reshape(-1, 2), axis=0)
        self.state[2:] = 0  # Initial velocity is unknown, set to 0
        self.uncertainty = np.eye(4)
        return self.state[:2]  # Return only the position part

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        Q = np.eye(4) * self.process_noise 
        
        self.state = F @ self.state
        self.uncertainty = F @ self.uncertainty @ F.T + Q

    def calculate_kalman_gain(self, measurement_uncertainty):
        H = np.eye(2, 4)  # Measurement matrix for position measurements
        S = H @ self.uncertainty @ H.T + measurement_uncertainty
        K = self.uncertainty @ H.T @ np.linalg.inv(S)
        return K

    def update_state(self, measurement, kalman_gain):
        H = np.eye(2, 4)  # Measurement matrix for position measurements
        innovation = measurement - H @ self.state
        self.state = self.state + kalman_gain @ innovation

    def update_uncertainty(self, kalman_gain):
        H = np.eye(2, 4)  # Measurement matrix for position measurements
        self.uncertainty = (np.eye(4) - kalman_gain @ H) @ self.uncertainty

    def update(self, dt, measurement):
        self.predict(dt)
        
        measurements = np.array(measurement[:10]).reshape(-1, 2)
        measurement_uncertainties = np.array(measurement[10:]).reshape(-1, 2)

        # Compute average measurement and combined measurement uncertainty
        avg_measurement = np.mean(measurements, axis=0)
        combined_uncertainty = np.zeros((2, 2))
        
        for i in range(len(measurements)):
            combined_uncertainty += np.diag(measurement_uncertainties[i]) ** 2

        combined_uncertainty = combined_uncertainty / len(measurements) + np.eye(2) * self.measurement_noise 

        kalman_gain = self.calculate_kalman_gain(combined_uncertainty)

        self.update_state(avg_measurement, kalman_gain)
        self.update_uncertainty(kalman_gain)

        return self.state[:2]
    
class ConstantTurnRateKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.01, alpha=0.69):
        self.dt = 1  # Default time step, will be updated dynamically
        self.state = np.zeros(7)  # [x, y, vx, vy, ax, ay, w]
        self.uncertainty = np.eye(7)
        self.process_noise = np.diag([1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])  # Process noise tuning
        self.measurement_noise = measurement_noise
        self.alpha = alpha  # Smoothing factor for EWMA
        self.smooth_position = np.zeros(2)

    def reset(self, measurement):
        x_init = np.mean(measurement[:10:2])
        y_init = np.mean(measurement[1:10:2])
        self.state = np.array([x_init, y_init, 0, 0, 0, 0, 0])  # Initial position, zero velocity and turn rate
        self.uncertainty = np.eye(7)  # High initial uncertainty
        self.smooth_position = np.array([x_init, y_init])  # Initialize smoothed position
        return self.state[:2]  # Return only the position

    def predict(self, dt):
        self.dt = dt
        theta = self.state[6] * dt  # Dynamic turn rate * dt
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # State transition matrix incorporating dynamic turn rate
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 0, 1, 0, dt, 0, 0],
            [0, 0, 0, 1, 0, dt, 0],
            [0, 0, 0, 0, cos_theta, -sin_theta, 0],
            [0, 0, 0, 0, sin_theta, cos_theta, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.state = F @ self.state
        self.uncertainty = F @ self.uncertainty @ F.T + self.process_noise
        return self.state

    def calculate_kalman_gain(self, measurement_uncertainty):
        H = np.zeros((10, 7))
        for i in range(5):
            H[2*i, 0] = 1
            H[2*i+1, 1] = 1

        S = H @ self.uncertainty @ H.T + measurement_uncertainty
        K = self.uncertainty @ H.T @ np.linalg.inv(S)
        return K

    def update_state(self, measurement, kalman_gain):
        H = np.zeros((10, 7))
        for i in range(5):
            H[2*i, 0] = 1
            H[2*i+1, 1] = 1
        innovation = measurement[:10] - H @ self.state
        self.state = self.state + kalman_gain @ innovation

    def update_uncertainty(self, kalman_gain):
        H = np.zeros((10, 7))
        for i in range(5):
            H[2*i, 0] = 1
            H[2*i+1, 1] = 1
        self.uncertainty = (np.eye(7) - kalman_gain @ H) @ self.uncertainty

    def update(self, dt, measurement):
        self.predict(dt)
        
        measurement_uncertainties = measurement[10:].reshape(-1, 2)

        # Compute combined measurement uncertainty
        combined_uncertainty = np.zeros((10, 10))
        for i in range(len(measurement_uncertainties)):
            combined_uncertainty[2*i:2*i+2, 2*i:2*i+2] = np.diag(measurement_uncertainties[i]) ** 2

        kalman_gain = self.calculate_kalman_gain(combined_uncertainty)

        self.update_state(measurement, kalman_gain)
        self.update_uncertainty(kalman_gain)

        # Exponentially weighted moving average (EWMA) for smoothing
        self.smooth_position = self.alpha * self.state[:2] + (1 - self.alpha) * self.smooth_position

        return self.smooth_position