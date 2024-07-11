import numpy as np


class constantturn_EKF:
    def __init__(self, shape):
        """
        Extended Kalman Filter (EKF) for constant turn rate model.

        Attributes:
            vector_shape (int): Dimension of state vector [x, y, vx, vy], with v = velocity (turning rate)
            num_coordinates (int): Number of 2D position measurements.
            velocities (np.array): Initial velocities.
            Jacobi_H (np.array): Jacobian matrix H.
            Prozessrauschen_Q (np.array): Process noise covariance matrix Q.
        """
        self.vector_shape = 4
        self.num_coordinates = 10
        self.velocity = 0.1
        self.velocities = np.ones((2, 1)) * self.velocity  # initial guess

        # Jacobi matrix (H) for multiplying only with the coordinates in meassurement
        Matrix_I = np.eye(shape)
        Matrix_I = np.vstack((Matrix_I, Matrix_I, Matrix_I, Matrix_I, Matrix_I))
        self.Jacobi_H = np.hstack((Matrix_I, np.zeros((self.num_coordinates, shape))))

        # noise for uncertancy matrix (P) update
        self.Prozessrauschen_Q = np.eye(self.vector_shape) * 1e-3

    def reset(self, measurement):
        """
        Reset the filter with initial guesses.
        """
        self.guess_x = np.vstack(
            (np.array([-0.6, 0.35]).reshape(2, 1), self.velocities)
        )
        self.x = self.guess_x[:2].flatten()
        self.Unsicherheit_P = np.eye(self.vector_shape) * 1e-2
        return self.x

    def update(self, dt, measurement):
        """
        Perform a prediction-update cycle of the Extended Kalman Filter.
        """
        # Update the measurement noise covariance matrix
        self.Messunsicherheit_R = np.diag(measurement[10:] ** 2)

        # Update matrix F "Prozessmodel"
        self.Prozessmodel_F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, np.cos(self.velocity * dt), -np.sin(self.velocity * dt)],
                [0, 0, np.sin(self.velocity * dt), np.cos(self.velocity * dt)],
            ]
        )

        # Prediction step
        x_pred = self.Prozessmodel_F @ self.guess_x
        P_pred = (
            self.Prozessmodel_F @ self.Unsicherheit_P @ self.Prozessmodel_F.T
            + self.Prozessrauschen_Q
        )

        # measured positions (z)
        measurement_z = measurement[:10].reshape(self.num_coordinates, 1)

        # Innovation covariance matrix
        Innovation_S = (
            self.Jacobi_H @ P_pred @ self.Jacobi_H.T + self.Messunsicherheit_R
        )

        # Kalman-Gain
        Gain_K = (
            P_pred @ self.Jacobi_H.T @ np.linalg.inv(Innovation_S)
        )  # Kalman gain with numerical stability

        # Update state vector and covariance matrix with measurement
        self.guess_x = x_pred + Gain_K @ (measurement_z - self.Jacobi_H @ x_pred)
        self.Unsicherheit_P = (
            np.eye(self.vector_shape) - Gain_K @ self.Jacobi_H
        ) @ P_pred

        # Extract the predicted coordinates
        self.x = self.guess_x[:2].flatten()
        return self.x
