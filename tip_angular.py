import numpy as np

class angular_EKF():
    def __init__(self, shape):
        """
        Extended Kalman Filter (EKF) for Problem angular.
        """
        self.shape = shape
        # R is given in task description:
        self.Messunsicherheit_R = np.array([[0.01, 0.0],[0.0, 0.0025]])


    def reset(self, measurement):
        """
        Reset the filter with an initial measurement.

        measurement contains the distance from the origin (distanz_pred) and the angle to the x-axis (phi_pred).
        self.x = initial guess for vector x = coordinates
        """
        self.x = np.array([0.19, -0.16])
        self.Unsicherheit_P = np.array([[0.099, 0.00016], [0.00016, 0.0021]]) 
        return self.x


    def update(self, dt, measurement):
        """
        Perform a prediction-update cycle of the Extended Kalman Filter.

        dt (float): Time step (not needed here).
        measurement (np.array): Current measurement as distance and angle.
        """
        # Since the coordinates are static there is no "Prozessmodel" to update x and P
        x_pred = self.x
        P_pred = self.Unsicherheit_P

        # Prediction step
        distanz_pred = np.sqrt(x_pred[0]**2 + x_pred[1]**2) # Pythagoras
        phi_pred = np.arctan2(x_pred[1], x_pred[0]) # Angle in radians
        z_pred = np.array([distanz_pred, phi_pred]) # Prediction k1_k0

        # Jacobian matrix with partial derivatives
        Jacobi_H = np.array([
            [x_pred[0] / distanz_pred, x_pred[1] / distanz_pred],
            [-x_pred[1] / (distanz_pred**2), x_pred[0] / (distanz_pred**2)]
        ])

        # Innovation covariance matrix
        Innovation_S = Jacobi_H @ P_pred @ Jacobi_H.T + self.Messunsicherheit_R 

        # Kalman-Gain
        Gain_K = P_pred @ Jacobi_H.T @ np.linalg.inv(Innovation_S)

        # Update state vector and uncertanty covariance matrix
        z = np.array(measurement)
        self.x = x_pred + Gain_K @ (z - z_pred)
        self.Unsicherheit_P = (np.eye(len(self.x)) - Gain_K @ Jacobi_H) @ P_pred

        return self.x

