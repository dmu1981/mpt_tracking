import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

class ConstantPositionFilter:

        def __init__(self, process_noise_var=.04):
                self.process_noise_var = process_noise_var
                # Initial state (location and velocity)
                self.x = np.zeros(2)  # Initial position (x, y)

                 # State covariance matrix
                self.P = np.eye(2)

        # State transition matrix (shows how the state or position changes from one time step to the next,
        # here constant
                self.F = np.eye(2)
                print(f"State transition matrix F:\n{self.F}")

        # Measurement matrix
                self.H = np.eye(2)

        # Measurement covariance matrix (R)
                self.R = np.eye(2) * self.process_noise_var

        # Process covariance matrix (Q)
                self.Q = np.eye(2) * 0.01

        def reset(self, measurement):
                logging.debug(f"Resetting filter with measurement: {measurement}")
                self.x = np.array(measurement[:2])
                self.P = np.eye(2)
                logging.debug(f"Filter state after reset: x={self.x}, P={self.P}")
                return self.x

        def update(self, dt, measurement):
                z = np.array(measurement[:2])

        # Prediction step
                self.x = self.F @ self.x
                self.P = self.F @ self.P @ self.F.T + self.Q

        # Update step
                y = z - self.H @ self.x
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)

                self.x = self.x + K @ y
                self.P = (np.eye(2) - K @ self.H) @ self.P

                return self.x

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    filter = ConstantPositionFilter(process_noise_var=0.04) 
    measurement = [0, 0] 
    filter.reset(measurement)

