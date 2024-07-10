import numpy as np

class KalmanFilter:
    def __init__(self):
        """Initialize Kalman Filter with default values."""
        self.state = np.zeros(2)  # Initial state vector [position]
        self.uncertainty = np.eye(4)  # Initial uncertainty covariance matrix
        self.measurement_noise = 0.2  # Standard deviation of measurement noise
        self.process_noise = 1e-8  # Process noise

    def reset(self, measurement):
        """Reset the filter with an initial measurement."""
        self.state = np.array(measurement[:2])  # Initialize state with measurement
        self.uncertainty = np.eye(2)  # Reset uncertainty
        return self.state

    def predict(self):
        """Predict the next state and uncertainty."""
        F = np.eye(2)  # State transition model (identity matrix for static object)
        Q = np.eye(2) * self.process_noise  # Process noise covariance matrix
        self.state = F @ self.state  # State prediction (no change for static model)
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def calculate_kalman_gain(self, measurement_uncertainty):
        """Calculate the Kalman Gain."""
        return self.uncertainty @ np.linalg.inv(
            self.uncertainty + measurement_uncertainty
        )

    def update_state(self, measurement, kalman_gain):
        """Update the state estimate."""
        self.state = self.state + kalman_gain @ (measurement - self.state)

    def update_uncertainty(self, kalman_gain):
        """Update the uncertainty covariance."""
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

    def update(self, dt, measurement):
        """Perform a full update cycle: predict and update."""
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


class FilterRandomNoise:
    def __init__(self, process_noise=1e-6):
        """Initialize Filter for random noise with default values."""
        self.state = np.zeros(2)  # Initial state vector [position]
        self.uncertainty = np.eye(2)  # Initial uncertainty covariance matrix
        self.process_noise = process_noise  # Process noise

    def reset(self, measurement):
        """Reset the filter with an initial measurement."""
        self.state = np.array(measurement[:2])  # Initialize state with measurement
        self.uncertainty = np.eye(2)  # Reset uncertainty
        return self.state

    def predict(self):
        """Predict the next state and uncertainty."""
        F = np.eye(2)  # State transition model (identity matrix for static object)
        Q = np.eye(2) * self.process_noise  # Process noise covariance matrix
        self.state = F @ self.state  # State prediction (no change for static model)
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def calculate_kalman_gain(self, measurement_uncertainty):
        """Calculate the Kalman Gain."""
        return self.uncertainty @ np.linalg.inv(
            self.uncertainty + measurement_uncertainty
        )

    def update_state(self, measurement, kalman_gain):
        """Update the state estimate."""
        self.state = self.state + kalman_gain @ (measurement - self.state)

    def update_uncertainty(self, kalman_gain):
        """Update the uncertainty covariance."""
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

    def update(self, dt, measurement):
        """Perform a full update cycle: predict and update."""
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
        """Initialize Angular Kalman Filter with default values."""
        self.state = np.zeros(2)  # Initial state vector [x, y]
        self.uncertainty = np.eye(2)  # Initial uncertainty covariance matrix
        self.process_noise = 1e-8  # Process noise
        self.measurement_noise_r = 0.1  # Measurement noise standard deviation for radius (r)
        self.measurement_noise_phi = 0.05  # Measurement noise standard deviation for angle (phi)

    def reset(self, measurement):
        """Reset the filter with an initial measurement in polar coordinates."""
        r, phi = measurement
        self.state = self.polar_to_cartesian(r, phi)  # Convert polar to Cartesian coordinates
        self.uncertainty = np.eye(2)  # Reset uncertainty
        return self.state

    def predict(self):
        """Predict the next state and uncertainty."""
        F = np.eye(2)  # State transition model (identity matrix for static object)
        Q = np.eye(2) * self.process_noise  # Process noise covariance matrix
        self.state = F @ self.state  # State prediction (no change for static model)
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def update(self, dt, measurement):
        """Perform a full update cycle: predict and update."""
        self.predict()

        # Convert polar measurement to Cartesian coordinates
        r, phi = measurement
        measurement_cartesian = self.polar_to_cartesian(r, phi)

        # Measurement noise covariance matrix in Cartesian coordinates
        R = np.array([[0.0100, 0.0000], [0.0000, 0.0025]])

        # Calculate Kalman gain
        kalman_gain = self.calculate_kalman_gain(R)

        # Update state
        self.update_state(measurement_cartesian, kalman_gain)

        # Update uncertainty
        self.update_uncertainty(kalman_gain)

        return self.state

    def calculate_kalman_gain(self, measurement_uncertainty):
        """Calculate the Kalman Gain."""
        return self.uncertainty @ np.linalg.inv(
            self.uncertainty + measurement_uncertainty
        )

    def update_state(self, measurement, kalman_gain):
        """Update the state estimate."""
        self.state = self.state + kalman_gain @ (measurement - self.state)

    def update_uncertainty(self, kalman_gain):
        """Update the uncertainty covariance."""
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty

    @staticmethod
    def polar_to_cartesian(r, phi):
        """Convert polar coordinates to Cartesian coordinates."""
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.array([x, y])


class ConstantVelocityKalmanFilter:
    def __init__(self, process_noise=1e-6, measurement_noise=0.69):
        """Initialize Constant Velocity Kalman Filter with default values."""
        self.dt = 1  # Default time step
        self.state = np.zeros(4)  # Initial state vector [x, y, vx, vy]
        self.uncertainty = np.eye(4)  # Initial uncertainty covariance matrix
        self.process_noise = process_noise  # Process noise
        self.measurement_noise = measurement_noise  # Measurement noise

    def reset(self, measurement):
        """Reset the filter with an initial measurement."""
        self.state[:2] = np.array(measurement[:2])  # Initialize position part of the state
        self.state[2:] = 0  # Initial velocity is unknown, set to zero
        self.uncertainty = np.eye(4) / 2  # Reset uncertainty
        return self.state[:2]  # Return only the position part

    def predict(self, dt):
        """Predict the next state and uncertainty."""
        self.dt = dt
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.eye(4) * self.process_noise  # Process noise covariance matrix

        self.state = F @ self.state  # State prediction
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def calculate_kalman_gain(self, measurement_uncertainty):
        """Calculate the Kalman Gain."""
        H = np.eye(2, 4)  # Measurement matrix (mapping state to measurement space)
        return (
            self.uncertainty
            @ H.T
            @ np.linalg.inv(H @ self.uncertainty @ H.T + measurement_uncertainty)
        )

    def update_state(self, measurement, kalman_gain):
        """Update the state estimate."""
        H = np.eye(2, 4)  # Measurement matrix
        self.state = self.state + kalman_gain @ (measurement - H @ self.state)

    def update_uncertainty(self, kalman_gain):
        """Update the uncertainty covariance."""
        H = np.eye(2, 4)  # Measurement matrix
        self.uncertainty = (np.eye(4) - kalman_gain @ H) @ self.uncertainty

    def update(self, dt, measurement):
        """Perform a full update cycle: predict and update."""
        self.predict(dt)

        measurement = np.array(measurement[:2])
        measurement_uncertainty = np.eye(2) * self.measurement_noise**2

        kalman_gain = self.calculate_kalman_gain(measurement_uncertainty)

        self.update_state(measurement, kalman_gain)

        self.update_uncertainty(kalman_gain)

        return self.state[:2]


class ConstantVelocityMultiMeasurementKalmanFilter:
    def __init__(self, process_noise=2e-4, measurement_noise=0.00001):
        """Initialize Constant Velocity Multi-Measurement Kalman Filter."""
        self.state = np.zeros(4)  # Initial state vector [x, y, vx, vy]
        self.uncertainty = np.eye(2)  # Initial uncertainty covariance matrix
        self.process_noise = process_noise  # Process noise
        self.measurement_noise = measurement_noise  # Measurement noise

    def reset(self, measurement):
        """Reset the filter with initial measurements."""
        self.state[:2] = np.mean(measurement[:10].reshape(-1, 2), axis=0)
        self.state[2:] = 0  # Initial velocity is unknown, set to zero
        self.uncertainty = np.eye(4)  # Reset uncertainty
        return self.state[:2]  # Return only the position part

    def predict(self, dt):
        """Predict the next state and uncertainty."""
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.eye(4) * self.process_noise  # Process noise covariance matrix

        self.state = F @ self.state  # State prediction
        self.uncertainty = F @ self.uncertainty @ F.T + Q  # Uncertainty prediction

    def calculate_kalman_gain(self, measurement_uncertainty):
        """Calculate the Kalman Gain."""
        H = np.eye(2, 4)  # Measurement matrix (mapping state to measurement space)
        S = H @ self.uncertainty @ H.T + measurement_uncertainty
        K = self.uncertainty @ H.T @ np.linalg.inv(S)
        return K

    def update_state(self, measurement, kalman_gain):
        """Update the state estimate."""
        H = np.eye(2, 4)  # Measurement matrix
        innovation = measurement - H @ self.state
        self.state = self.state + kalman_gain @ innovation

    def update_uncertainty(self, kalman_gain):
        """Update the uncertainty covariance."""
        H = np.eye(2, 4)  # Measurement matrix
        self.uncertainty = (np.eye(4) - kalman_gain @ H) @ self.uncertainty

    def update(self, dt, measurement):
        """Perform a full update cycle: predict and update."""
        self.predict(dt)

        measurements = np.array(measurement[:10]).reshape(-1, 2)
        measurement_uncertainties = np.array(measurement[10:]).reshape(-1, 2)

        # Compute average measurement and combined measurement uncertainty
        avg_measurement = np.mean(measurements, axis=0)
        combined_uncertainty = np.zeros((2, 2))

        for i in range(len(measurements)):
            combined_uncertainty += np.diag(measurement_uncertainties[i]) ** 2

        combined_uncertainty = (
            combined_uncertainty / len(measurements)
            + np.eye(2) * self.measurement_noise
        )

        kalman_gain = self.calculate_kalman_gain(combined_uncertainty)

        self.update_state(avg_measurement, kalman_gain)
        self.update_uncertainty(kalman_gain)

        return self.state[:2]


class ConstantTurnRateKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.01, alpha=0.69):
        """Initialize Constant Turn Rate Kalman Filter with default values."""
        self.dt = 1  # Default time step
        self.state = np.zeros(7)  # Initial state vector [x, y, vx, vy, ax, ay, w]
        self.uncertainty = np.eye(7)  # Initial uncertainty covariance matrix
        self.process_noise = np.diag(
            [1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
        )  # Process noise tuning
        self.measurement_noise = measurement_noise  # Measurement noise
        self.alpha = alpha  # Smoothing factor for EWMA (Exponential Weighted Moving Average)
        self.smooth_position = np.zeros(2)  # Smoothed position

    def reset(self, measurement):
        """Reset the filter with initial measurements."""
        x_init = np.mean(measurement[:10:2])
        y_init = np.mean(measurement[1:10:2])
        self.state = np.array(
            [x_init, y_init, 0, 0, 0, 0, 0]
        )  # Initial position, zero velocity and turn rate
        self.uncertainty = np.eye(7)  # High initial uncertainty
        self.smooth_position = np.array(
            [x_init, y_init]
        )  # Initialize smoothed position
        return self.state[:2]  # Return only the position

    def predict(self, dt):
        """Predict the next state and uncertainty."""
        self.dt = dt
        theta = self.state[6] * dt  # Dynamic turn rate * dt
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # State transition matrix incorporating dynamic turn rate
        F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 0, 1, 0, dt, 0, 0],
                [0, 0, 0, 1, 0, dt, 0],
                [0, 0, 0, 0, cos_theta, -sin_theta, 0],
                [0, 0, 0, 0, sin_theta, cos_theta, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.state = F @ self.state  # State prediction
        self.uncertainty = F @ self.uncertainty @ F.T + self.process_noise  # Uncertainty prediction
        return self.state

    def calculate_kalman_gain(self, measurement_uncertainty):
        """Calculate the Kalman Gain."""
        H = np.zeros((10, 7))
        for i in range(5):
            H[2 * i, 0] = 1
            H[2 * i + 1, 1] = 1

        S = H @ self.uncertainty @ H.T + measurement_uncertainty
        K = self.uncertainty @ H.T @ np.linalg.inv(S)
        return K

    def update_state(self, measurement, kalman_gain):
        """Update the state estimate."""
        H = np.zeros((10, 7))
        for i in range(5):
            H[2 * i, 0] = 1
            H[2 * i + 1, 1] = 1
        innovation = measurement[:10] - H @ self.state
        self.state = self.state + kalman_gain @ innovation

    def update_uncertainty(self, kalman_gain):
        """Update the uncertainty covariance."""
        H = np.zeros((10, 7))
        for i in range(5):
            H[2 * i, 0] = 1
            H[2 * i + 1, 1] = 1
        self.uncertainty = (np.eye(7) - kalman_gain @ H) @ self.uncertainty

    def update(self, dt, measurement):
        """Perform a full update cycle: predict and update."""
        self.predict(dt)

        measurement_uncertainties = measurement[10:].reshape(-1, 2)

        # Compute combined measurement uncertainty
        combined_uncertainty = np.zeros((10, 10))
        for i in range(len(measurement_uncertainties)):
            combined_uncertainty[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = (
                np.diag(measurement_uncertainties[i]) ** 2
            )

        kalman_gain = self.calculate_kalman_gain(combined_uncertainty)

        self.update_state(measurement, kalman_gain)
        self.update_uncertainty(kalman_gain)

        # Exponentially weighted moving average (EWMA) for smoothing
        self.smooth_position = (
            self.alpha * self.state[:2] + (1 - self.alpha) * self.smooth_position
        )

        return self.smooth_position



