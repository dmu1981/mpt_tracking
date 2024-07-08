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
        
#Der Kalman-Gewinn wird berechnet, um zu bestimmen, wie stark die Vorhersage anhand der neuen Messung angepasst werden sollte. Mathematisch: K = p*(p+r)^-1 
 #P ist die Unsicherheitsmatrix und R die Messunsicherheit
        kalman_gain = self.uncertainty @ np.linalg.inv(self.uncertainty + measurement_uncertainty)
#Zustand wird aktualisiert 
#Der neue Zustand wird basierend auf dem Kalman-Gewinn und der Differenz zwischen der Messung und dem vorhergesagten Zustand aktualisiert. Mathematisch: x = x + K * (z - x)
# x = vorhergesagter Zustand K = Kalman Gewinn und z = Messung
        self.state = self.state + kalman_gain @ (measurement - self.state)
#unsicherheit wird aktualisiert 
#Die Unsicherheitsmatrix wird angepasst um die verbesserte Schätzung wiederzuspiegeln: P=(I - K * H) * P 
# I = Einheitsmatrix
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty
        return self.state
    
    #Problem 2
class RandomNoiseFilter:
    def __init__(self, process_noise_scale=0.04, initial_uncertainty=600):
        self.state = np.zeros((2, 1))  # Initial state vector (x, y)
        self.uncertainty = np.eye(2) * initial_uncertainty  #Initial uncertainty Covar Matrix P.We can print P after every update to see 
        #if the diagonal high numbers (uncertainty) goes down
        self.process_noise_scale = process_noise_scale
        self.process_noise = np.eye(2) * self.process_noise_scale  # Process noise matrix Q, here stays const, unlike meas noise matrix R

    def reset(self, measurement):
        # Extract initial state from the measurement
        measurement = np.array(measurement[:2]).reshape((2, 1))
        self.state = measurement 
        self.uncertainty = np.eye(2) * 500
        return self.state.flatten()

    def update(self, dt, measurement):  
# Extract the state measurements xy from z and reshape it to the shape of self.state, save the Covar meas Matrix R as a 2x2 matrix
# for the further S calculation. 
        z = np.array(measurement[:2]).reshape((2, 1))
        R = np.array(measurement[2:6]).reshape((2, 2))
        
# Compute the Kalman gain. K optimizes the distribution of weights for our prediction and the new measurement  
# These weights determine the relative influence of the new measurement and the prediction on the updated state estimate
# A higher Kalman gain means more trust is placed in the new measurement, whereas a lower Kalman gain means more trust for our prediction
        S = self.uncertainty + R
        K = self.uncertainty @ np.linalg.inv(S)
        
#Innovation y shows us the difference between the actual measurement z from our state estimate (basically out prediction errror, also called measurement residual)
        y = z - self.state  # Innovation y
        self.state = self.state + K @ y #Update of our state estimate
        I = np.eye(self.uncertainty.shape[0]) #Identity Matrix I
        # Update the uncertainty Covar P
        self.uncertainty = (I - K) @ self.uncertainty
        return self.state.flatten() # returns the current state estimate as a 1d array bs otherwise main.py will throw an error
     
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
    
    
#Problem 6 
class ConstantTurnFilter:
    def __init__(self, turn_rate=0, initial_uncertainty=500, process_noise_std=0.01):
        self.turn_rate = turn_rate # Estimated turn rate
        self.process_noise_std = process_noise_std
        self.initial_uncertainty = initial_uncertainty
        self.Q = np.diag([process_noise_std**2, process_noise_std**2, (process_noise_std*10)**2, (process_noise_std*10)**2])  # scaled Process noise Covar Matrix Q
        self.state = np.zeros(4)  # Initial state (x, y, vx, vy)
        self.uncertainty = np.eye(4) * initial_uncertainty  # Initial Covar Matrix P, that we later gonna update with our prediction error 

    def reset(self, measurement):
        self.state[:2] = measurement[:2]
        self.state[2:] = 0  # Initial velocities are set to 0
        self.uncertainty = np.diag([10, 10, 1, 1])  # Adjusted initial covariance
        return self.state[:2]  # Return only the position (x, y)

    def update(self, dt, measurements, turn_rate=None):
# Extract measurements and their uncertainties
        if turn_rate is not None:
            self.turn_rate = turn_rate
        else:
# Estimate turn rate from current state
            v = np.linalg.norm(self.state[2:])
            if v > 0.1:
                self.turn_rate = (self.state[2] * -self.state[3] + self.state[3] * self.state[2]) / (v ** 2)
        
# Calculating the Jocobi state transition matrix F. It is calculated the new state based on the old state  and the turn rate.
        a_cos = np.cos(self.turn_rate * dt)
        a_sin = np.sin(self.turn_rate * dt)
        
        F = np.array([
            [1, 0, dt * a_cos, -dt * a_sin],
            [0, 1, dt * a_sin,  dt * a_cos],
            [0, 0,        a_cos,        -a_sin],
            [0, 0,        a_sin,         a_cos]
        ])

# Predict step for the estimated state and the uncertainty Covar Matrix P
        self.state = F @ self.state
        self.uncertainty = F @ self.uncertainty @ F.T + self.Q * dt # we scale the process noise Matrix Q by dt

# Extract measurements and their uncertainties 
        H = np.zeros((10, 4))
        H[0::2, 0] = 1  # x measurements
        H[1::2, 1] = 1  # y measurements

        z = measurements[:10]  # First 10 elements are measurements
        R = np.diag(measurements[10:]**2)  # Last 10 elements are std deviations
        
        # Kalman gain
        innovation = z - H @ self.state
        S = H @ self.uncertainty @ H.T + R
        K = self.uncertainty @ H.T @ np.linalg.inv(S)

        # Update step
        self.state = self.state + K @ innovation
        self.uncertainty = (np.eye(4) - K @ H) @ self.uncertainty
        return self.state[:2]  # Return only the position (x, y)
