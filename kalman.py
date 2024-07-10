import numpy as np

"""
The Kalman filter is a recursive algorithm that estimates the state of a dynamic system over time.
It combines measurements (which are often noisy and inaccurate) with a model of the system to provide a more accurate estimate of the state.
"""

class KalmanFilter:
    """
    This class adds a Kalman filter, recursively processing data 
    to estimate the state of a linear dynamic system from a series of noisy measurements.
    Assuming a 2-dimensional state vector of position and velocity, it processes the 2D position measurements.
    """
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state_estimate = np.zeros(state_dim)  # initial state (location and velocity)
        self.uncertainty_covariance = np.eye(state_dim[0])  # uncertainty covariance
        self.state_transition = np.eye(state_dim[0])  # state transition matrix
        self.measurement_matrix = np.eye(state_dim[0])  # measurement matrix
        self.measurement_uncertainty = np.eye(state_dim[0]) * 0.04  # measurement uncertainty (noise) matrix
        self.identity_matrix = np.eye(state_dim[0])  # identity matrix
        
    def reset(self, measurement):
        self.state_estimate = measurement[:2]  # initialize state with first measurement
        self.uncertainty_covariance = np.eye(2)  # reset uncertainty covariance
        return self.state_estimate
    
    def update(self, dt, measurement):
        measured_value = measurement[:2]  # measurement vector / measured value
        measurement_residual = measured_value - np.dot(self.measurement_matrix, self.state_estimate)  # measurement residual: difference between measured and estimated values
        residual_covariance = np.dot(self.measurement_matrix, np.dot(self.uncertainty_covariance, self.measurement_matrix.T)) + self.measurement_uncertainty  # residual covariance: uncertainty of measurement residual
        kalman_gain = np.dot(np.dot(self.uncertainty_covariance, self.measurement_matrix.T), np.linalg.inv(residual_covariance))  # Kalman gain: weighting of measurement residual to update state
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, measurement_residual)  # update of state estimate: addition of weighted measurement residual
        self.uncertainty_covariance = np.dot((self.identity_matrix - np.dot(kalman_gain, self.measurement_matrix)), self.uncertainty_covariance)  # updated estimate covariance: reduction of uncertainty
        return self.state_estimate

class KalmanFilterRandomNoise:
    """
    This class works more or less the same as the normal Kalman filter.
    Only change here was the addition of the measurement noise covariance,
    which quantifies the level of confidence in the measurements
    and therefore is needed to update the state estimate and its uncertainty/noise.
    """
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state_estimate = np.zeros(state_dim)
        self.uncertainty_covariance = np.eye(state_dim)
        self.state_transition = np.eye(state_dim)
        self.measurement_matrix = np.eye(state_dim)
        self.identity_matrix = np.eye(state_dim)

    def reset(self, measurement):
        self.state_estimate = measurement[:self.state_dim]
        self.uncertainty_covariance = np.eye(self.state_dim)
        return self.state_estimate

    def update(self, dt, measurement):
        measured_value = measurement[:self.state_dim]
        measurement_noise_covariance = measurement[self.state_dim:].reshape(self.state_dim, self.state_dim)
        measurement_residual = measured_value - np.dot(self.measurement_matrix, self.state_estimate)
        residual_covariance = np.dot(self.measurement_matrix, np.dot(self.uncertainty_covariance, self.measurement_matrix.T)) + measurement_noise_covariance
        kalman_gain = np.dot(np.dot(self.uncertainty_covariance, self.measurement_matrix.T), np.linalg.inv(residual_covariance))
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, measurement_residual)
        self.uncertainty_covariance = np.dot((self.identity_matrix - np.dot(kalman_gain, self.measurement_matrix)), self.uncertainty_covariance)
        return self.state_estimate
    
class KalmanFilterAngular:
    """
    Kalman filter adjusted for polar coordinates.
    For tracking a static object with direct measurements in polar coordinates
    """
    def __init__(self):
        self.state = None
        self.covariance = None
        self.R = np.array([[0.0100, 0.0000], [0.0000, 0.0025]]) # measurement noise covariance matrix

    def reset(self, measurement):
        r, phi = measurement[0], measurement[1] # polar coordinates
        x = r * np.cos(phi) # convert into cartesian
        y = r * np.sin(phi) # convert into cartesian
        self.state = np.array([x, y])
        self.covariance = np.eye(2) * 0.001  # initial covariance, can be fine-tuned
        return self.state

    def update(self, dt, measurement):
        
        r, phi = measurement[0], measurement[1] # measurement update (correction)
        cartesian = np.array([r * np.cos(phi), r * np.sin(phi)]) # measurement in cartesian coordinates
        polar = np.array([np.sqrt(self.state[0]**2 + self.state[1]**2), np.arctan2(self.state[1], self.state[0])]) # convert state to polar coordinates
        jacobian = self._calculate_jacobian(self.state) # jacobian of the measurement function
        
        measurement_residual = cartesian - polar # innovation/measurement residual
        measurement_residual[1] = self._normalize_angle(measurement_residual[1]) # normalize the angle residual
        innovation_covariance = jacobian @ self.covariance @ jacobian.T + self.R # quantify consistency between prediction and actual measurement
        
        kalman_gain = self.covariance @ jacobian.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ measurement_residual # update state estimate
        
        # Update covariance estimate
        covariance_matrix = np.eye(self.covariance.shape[0])
        self.covariance = (covariance_matrix - kalman_gain @ jacobian) @ self.covariance
        
        return self.state

    def _calculate_jacobian(self, state):
        px, py = state[0], state[1] # state variables
        range = np.sqrt(px**2 + py**2) # distance between origin and point
        
        if range == 0:
            return np.zeros((2, 2))
        
        # partial derivates of range
        drange_dpx = px / range
        drange_dpy = py / range
        
        # partial derivates of phi
        dphi_dpx = -py / (range**2)
        dphi_dpy = px / (range**2)
        
        jacobian = np.array([
            [drange_dpx, drange_dpy],
            [dphi_dpx, dphi_dpy]
        ])
        return jacobian
    
    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
import numpy as np

class KalmanFilterConstantTurn:
    def __init__(self):
        self.dt = 0.1  # time step
        self.state = None
        self.covariance = None
        self.R = np.eye(2) * 0.04  # measurement noise covariance
        self.Q = np.eye(4) * 0.01  # process noise covariance

    def reset(self, measurement):
        # state vector [x, y, vx, vy] from first measurement
        self.state = np.array([measurement[0], measurement[1], 0, 0])
        self.covariance = np.eye(4) * 0.1  # initial state covariance
        return self.state[:2]

    def update(self, dt, measurement):
        self.dt = dt
        self._predict()
        self._update(measurement)
        return self.state[:2]

    def _predict(self):
        x, y, vx, vy = self.state
        a = 0.1  # assumed turn rate

        # state transition matrix for model
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, np.cos(a * self.dt), -np.sin(a * self.dt)],
            [0, 0, np.sin(a * self.dt), np.cos(a * self.dt)]
        ])

        self.state = F @ self.state # predicted state estimate

        self.covariance = F @ self.covariance @ F.T + self.Q # predicted covariance estimate

    def _update(self, measurement):
        z = measurement[:10].reshape(5, 2)
        sigma = measurement[10:].reshape(5, 2)
        
        # measurement matrix
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        for i in range(5):
            innovation = z[i] - H @ self.state

            R = np.diag(sigma[i]**2) # measurement noise covariance for this measurement
            S = H @ self.covariance @ H.T + R # innovation covariance
            K = self.covariance @ H.T @ np.linalg.inv(S) # kalman gain

            self.state = self.state + K @ innovation

            # updated covariance estimate
            I = np.eye(self.covariance.shape[0])
            self.covariance = (I - K @ H) @ self.covariance