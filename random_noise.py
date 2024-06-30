import numpy as np
import pandas as pd
import pickle

process_noise_var = 0.04

class RandomNoiseFilter:
    def __init__(self, process_noise_var=process_noise_var):
        self.x = np.zeros((2, 1)) # state vector xy
        self.P = np.eye(2) * 100  #we can print P after every update step to see 
        #if the diagonal high numbers (uncertainty) goes down
        self.Q = np.eye(2) * process_noise_var  # Process noise stays constant, unlike measurement noise Covar matrix R

    def reset(self, measurement):
        measurement = np.array(measurement[:2]).reshape((2, 1))
        self.x = measurement
        self.P = np.eye(2) * 100
        return self.x.flatten()

    def update(self, dt, measurement):
# extract xy from z and reshape it to the shape of self.x, save Rt as a 2x2 matrix
# for the further S calculation
        z = np.array(measurement[:2]).reshape((2, 1))
        Rt = np.array(measurement[2:6]).reshape((2, 2))
        S = self.P + Rt  # Innovation Covar S

# K optimizes the distribution of weights for our prediction and the new measurement  
# These weights determine the relative influence of the new measurement and the prediction on the updated state estimate
# A higher Kalman gain means more trust is placed in the new measurement, whereas a lower Kalman gain means more trust for our prediction
        K = self.P @ np.linalg.inv(S)

#Innovation shows us the difference between the actual measurement z from our state estimate (basically out prediction errror)
        y = z - self.x

# New state estimate
        self.x = self.x + K @ y

# New the state Covar P
        self.P = (np.eye(self.P.shape[0]) - K) @ self.P

        return self.x.flatten() # returns the current state estimate as a 1d array
#bs otherwise main.py will throw an error

    def get_state_covariance(self):
        return self.P

# Initiate the filter
rn_filter = RandomNoiseFilter(process_noise_var)

# Read the data form the pk file
with open('randomnoise.pk', 'rb') as file:
    pk_data = pickle.load(file)

dt=5 #lets assume the time step is 5 sec, it doesnt change the outcome of this filter 
#but lets keep it for the further problems

# iterate through the pk measurements and update the filter
for entry in pk_data:
    if 'measurements' in entry:
        measurements = entry['measurements']
        for measurement in measurements: 
            current_state_estimate = rn_filter.update(dt, measurement)
            current_state_covariance = rn_filter.get_state_covariance()
            #print(f"Current State Estimate: {current_state_estimate}")
            #print(f"Current State Covariance: {current_state_covariance}")
         
