import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

class ConstantPositionFilter:

        def __init__(self, process_noise_var=.04):
                self.process_noise_var = process_noise_var
                self.x = np.zeros(2)  # initial state of the vector xy
                self.P = np.eye(2)  # initial Covar matrix P of the state vector xy
                self.H = np.eye(2)    # measurement matrix H
                self.R = np.eye(2) * self.process_noise_var   # measurement noise Covar matrix R
                self.Q = np.eye(2) # process noise Covar matrix Q

        def reset(self, measurement):
                self.x = np.array(measurement[:2])
                self.P = np.eye(2)
                return self.x

        def update(self, dt, measurement):
                z = np.array(measurement[:2]) # measurement vector z

        # Prediction step
                self.x = self.x # no change in state vector xy
                self.P = self.P + self.Q # updated Covar matrix P 

        # Measurement step
                y = z - self.x # innovation y
                S = self.P + self.R # innovation Covar S
                K = self.P @ np.linalg.inv(S) # Kalman gain K

                self.x = self.x + K @ y # new state estimate
                self.P = (np.eye(2) - K @ self.H) @ self.P # new Covar matrix P

                return self.x

if __name__ == "__main__":
    filter = ConstantTurnFilter()
    filter.run()

