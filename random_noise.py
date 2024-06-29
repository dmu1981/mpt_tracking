import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import logging

class RandomNoiseFilter:
    def __init__(self, process_noise_var):
        # State vector xy. Will be updated with predictions
        self.x = np.zeros((2, 1))
        
        # State Covar matrix P is initially an identity matrix with high values on diagonal
        self.P = np.eye(2) * 100  # 100 is random big number to reflect high initial uncertainty
        #its a good indicator if the diagonal high numbers go down. So we should print P after every update step
        
        # State transition matrix F. In problem 2 always stays the same because the state is constant
        self.F = np.eye(2)

        # Measurement matrix H
        self.H = np.eye(2)
        
        # Process noise Covar matrix Q. Here constant, unlike measurement noise Covar matrix R
        self.Q = np.eye(2) * process_noise_var 

    def predict(self):
        # Predict the next state estimate of vector xy
        self.x = self.F @ self.x
        
        # Update the state Covar P. Here plus Q because we are adding process noise
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement, Rt):
        # Measurement vector z. Here we only take xy coordinates from the measurement
        measurement = np.array(measurement)

        z = np.array(measurement[:2]).reshape((2, 1))
        
        # Innovation y shows the difference between the actual measurement z from our state estimate (basically out prediction errror)
        # recalculated after every update step
        y = z - self.H @ self.x
        
        # Innovation Covar S
        S = self.H @ self.P @ self.H.T + Rt
        
        # Kalman gain implements a minimal Var fusion by distributing the weights for our prediction and the new measurement.
        # The highter the weight, the more trust we put in either new measurement or our new prediction
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # New step for the state estimate of vector xy with added Kalman gain
        self.x = self.x + K @ y
        
        # New state Covar matrix P
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return y 
    
    def get_state_estimate(self):
        # returns the current state estimate as a column vector. Lets flatten it to print as a row vector as shown in Dummy()
        return self.x.flatten()
    
    def get_state_covariance(self):
        # Return the current state Covar matrix
        return self.P

rn_filter = RandomNoiseFilter(process_noise_var=0.04)

# Read mesurements from the randomnoise.pk file
# Here chellenging since measurements are stored in a list
# with new values appended to the old ones 
with open('randomnoise.pk', 'rb') as file:
    pk_data = pickle.load(file)


for entry in pk_data:
    if 'measurements' in entry:
        measurements = entry['measurements']
        for measurement in measurements:
            if len(measurement) < 6:
                print(f"Measurement does not have enough elements, skipping: {measurement}")
                continue
    z = measurement[:2] # xy coordinates of the measurement z
# Rt is a measurement noise Covar matrix at time t. Rt weithts up the trustworthiness of the measurements in relation to the current state estimate.
# Since our measurement vector z has form (x, y, Rt), we extract Rt from the measurements_array and reshape it to a 2x2 matrix. 
# We pass it to the update method and print it after every update step to see if the noise in the measurements is correlated
    Rt = np.array(measurement[2:6]).reshape((2, 2)) 
    print("Measurement:", z, "Noise CovVar Rt:", Rt)
    rn_filter.predict()
    y = rn_filter.update(z, Rt)  #its our prediction error y
    current_state_estimate = rn_filter.get_state_estimate() #fusion of the new predicted state and the new measurement, incl. our prediction error y
    current_state_covariance = rn_filter.get_state_covariance()
    print("Current state estimate:", current_state_estimate)
    print("Current state covariance matrix P:\n", current_state_covariance)
    print("Prediction error:", y.flatten())
    
    def reset(self):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 100 
        self.H = np.eye(2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    rn_filter = RandomNoiseFilter(process_noise_var=0.04) 
    rn_filter.reset()