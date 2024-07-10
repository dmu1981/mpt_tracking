import numpy as np


# Problem 1
class ConstantpositionFilter:
    def __init__(self):
        # Der Anfangszustand wird auf einen Nullvektor gesetzt
        self.state = np.zeros(2)
        # Anfangsunsicherheit wird auf eine Matrix mit 500 auf der Diagonale gesetzt
        self.uncertainty = np.eye(2) * 500
        # Messrauschen wird auf 0.2 gesetzt
        self.measurement_noise = 0.2
        # Prozessrauschen wird auf einen sehr kleinen Wert gesetzt
        self.process_noise = 1e-5

    def reset(self, measurement):
        # Zustand wird auf die ersten zwei Messwerte gesetzt
        self.state = np.array(measurement[:2])
        # Unsicherheit wird zurückgesetzt
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, dt, measurement):
        measurement = np.array(measurement[:2])
        measurement_uncertainty = np.eye(2) * self.measurement_noise**2

        # Der Kalman-Gewinn wird berechnet, um zu bestimmen, wie stark die Vorhersage anhand der neuen Messung angepasst werden sollte.
        # Mathematisch: K = p*(p+r)^-1
        # P ist die Unsicherheitsmatrix und R die Messunsicherheit
        kalman_gain = self.uncertainty @ np.linalg.inv(
            self.uncertainty + measurement_uncertainty
        )
        # Zustand wird aktualisiert
        # Der neue Zustand wird basierend auf dem Kalman-Gewinn und der Differenz zwischen der Messung 
        # und dem vorhergesagten Zustand aktualisiert. Mathematisch: x = x + K * (z - x)
        # x = vorhergesagter Zustand K = Kalman Gewinn und z = Messung
        self.state = self.state + kalman_gain @ (measurement - self.state)
        # unsicherheit wird aktualisiert
        # Die Unsicherheitsmatrix wird angepasst um die verbesserte Schätzung wiederzuspiegeln: P=(I - K * H) * P
        # I = Einheitsmatrix
        self.uncertainty = (np.eye(2) - kalman_gain) @ self.uncertainty
        return self.state

    # Problem 2


class RandomNoiseFilter:
    def __init__(self, process_noise_scale=0.04, initial_uncertainty=600):
        self.state = np.zeros((2, 1))  # Initial state vector (x, y)
        self.uncertainty = (
            np.eye(2) * initial_uncertainty
        )  # Initial uncertainty Covar Matrix P.We can print P after every update to see
        # if the diagonal high numbers (uncertainty) goes down
        self.process_noise_scale = process_noise_scale
        self.process_noise = (
            np.eye(2) * self.process_noise_scale
        )  # Process noise matrix Q, here stays const, unlike meas noise matrix R

    def reset(self, measurement):
        # Extract initial state from the measurement
        measurement = np.array(measurement[:2]).reshape((2, 1))
        self.state = measurement
        self.uncertainty = np.eye(2) * 500
        return self.state.flatten()

    def update(self, dt, measurement):
        # Extract the state measurements xy from z and reshape it to the shape of self.state, 
        # save the Covar meas Matrix R as a 2x2 matrix
        # for the further S calculation.
        z = np.array(measurement[:2]).reshape((2, 1))
        R = np.array(measurement[2:6]).reshape((2, 2))

        # Compute the Kalman gain. K optimizes the distribution of weights for our prediction and the new measurement
        # These weights determine the relative influence of the new measurement and the prediction on the updated state estimate
        # A higher Kalman gain means more trust is placed in the new measurement, 
        # whereas a lower Kalman gain means more trust for our prediction
        S = self.uncertainty + R
        K = self.uncertainty @ np.linalg.inv(S)

        # Innovation y shows us the difference between the actual measurement z from our state estimate 
        # (basically out prediction errror, also called measurement residual)
        y = z - self.state  # Innovation y
        self.state = self.state + K @ y  # Update of our state estimate
        I = np.eye(self.uncertainty.shape[0])  # Identity Matrix I
        # Update the uncertainty Covar P
        self.uncertainty = (I - K) @ self.uncertainty
        return (
            self.state.flatten()
        )  # returns the current state estimate as a 1d array bs otherwise main.py will throw an error

    # Problem3


class AngularFilter:
    def __init__(self):
        # Der Anfangszustand wird auf einen Nullvektor gesetzt
        self.state = np.zeros(2)  # Initial state (x, y)
        # Anfangsunsicherheit wird auf eine Matrix mit 500 auf der Diagonale gesetzt
        self.uncertainty = np.eye(2) * 500  # Initial uncertainty
        # Messrauschen wird als Matrix mit unterschiedlichen Werten für x und y gesetzt
        self.measurement_noise = np.array([[0.01, 0], [0, 0.0025]])  # Measurement noise
        # Prozessrauschen wird auf eine sehr kleine Matrix gesetzt
        self.process_noise = np.eye(2) * 1e-5  # Process noise

    def reset(self, measurement):

        # Messung in Polarkoordinaten
        r, phi = measurement
        # Umrechnung der Polarkoordinaten in kartesische Koordinaten
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        # Anfangszustand wird auf die umgerechneten kartesischen Koordinaten gesetzt
        self.state = np.array([x, y])
        # Unsicherheit wird zurückgesetzt
        self.uncertainty = np.eye(2) * 500
        return self.state

    def update(self, dt, measurement, *args, **kwargs):
        # Messung in Polarkoordinaten
        r, phi = measurement

        # Vorhersageschritt: Unsicherheit aufgrund von Prozessrauschen erhöhe
        self.uncertainty += self.process_noise

        # Umrechnung der Polarkoordinaten in kartesische Koordinaten
        measured_x = r * np.cos(phi)
        measured_y = r * np.sin(phi)
        measurement_cartesian = np.array([measured_x, measured_y])

        # Umrechnung des Messrauschens von Polarkoordinaten in kartesische Koordinaten
        H = np.array([[np.cos(phi), -r * np.sin(phi)], [np.sin(phi), r * np.cos(phi)]])
        R_cartesian = H @ self.measurement_noise @ H.T

        # Berechnung des Kalman-Gewinns
        S = self.uncertainty + R_cartesian
        K = self.uncertainty @ np.linalg.inv(S)

        # Aktualisierung der Zustandsschätzung
        y = measurement_cartesian - self.state  # Measurement residual
        self.state = self.state + K @ y

        # Aktualisierung der Unsicherheit
        I = np.eye(2)
        self.uncertainty = (I - K) @ self.uncertainty

        return self.state


# Problem 4
class KalmanFilterConstantVelocity:
    def __init__(self):
        # Der Anfangszustand wird auf einen Nullvektor gesetzt (Position und Geschwindigkeit)
        # Zustandsvektor [x,y,vx,vy]
        self.x = np.zeros(4)
        # Anfangsunsicherheit wird auf die Einheitsmatrix gesetzt
        self.P = np.eye(4)
        # Messrauschen wird als Matrix mit 0.04 auf der Diagonale gesetzt
        self.R = np.eye(2) * 0.04
        # Einheitsmatrix
        self.I = np.eye(4)

    def reset(self, measurement):
        # Der Anfangszustand wird auf einen Nullvektor gesetzt
        self.x = np.zeros(4)
        # Die ersten zwei Werte des Zustands werden auf die Messwerte gesetzt
        self.x[:2] = measurement[:2]  # Set position to the measurement
        # Unsicherheit wird zurückgesetzt
        self.P = np.eye(4)  # Reset uncertainty covariance
        return self.x[:2]

    def update(self, dt, measurement):
        # Zustandsübergangsmatrix für konstante Geschwindigkeit
        self.F = np.eye(4)  # State transition matrix
        self.F[0, 2] = dt  # Position x ändert sich mit Geschwindigkeit vx
        self.F[1, 3] = dt  # Position y ändert sich mit Geschwindigkeit vy

        # Vorhersage: Aktualisiere den Zustand und die Unsicherheit basierend auf dem Modell
        self.x = self.F @ self.x  # Predicted state estimate
        self.P = self.F @ self.P @ self.F.T  # Predicted uncertainty covariance

        # Messmatrix, die den Zustand auf die Messung abbildet
        H = np.eye(2, 4)

        # Korrekturschritt: Berechne die Differenz zwischen Messung und Vorhersage
        # Tatsächliche Messung
        z = measurement[:2]
        # Messabweichung (residual)
        y = z - H @ self.x

        # Berechne den Kalman-Gewinn
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        # Kalman Gewinn
        K = self.P @ H.T @ np.linalg.inv(S)

        # Aktualisiere den Zustand basierend auf der Messabweichung und dem Kalman-Gewinn
        self.x = self.x + K @ y

        # Aktualisiere die Unsicherheit basierend auf dem Kalman-Gewinn
        self.P = (self.I - K @ H) @ self.P
        # Rückgabe der geschätzten Position (x, y)
        return self.x[:2]


# Problem 5
class KalmanFilterConstantVelocityMultiple:
    def __init__(self):
        # Der Anfangszustand wird auf einen Nullvektor gesetzt (Position und Geschwindigkeit)
        # Zustandsvektor [x, y, vx, vy]
        self.x = np.zeros(4)
        # Anfangsunsicherheit wird auf die Einheitsmatrix gesetzt
        # Kovarianzmatrix
        self.P = np.eye(4)
        # Einheitsmatrix
        self.I = np.eye(4)

    def reset(self, measurement):
        # Der Anfangszustand wird auf einen Nullvektor gesetzt
        self.x = np.zeros(4)
        # Die ersten zwei Werte des Zustands werden auf die durchschnittliche Position der ersten 10 Messwerte gesetzt
        self.x[:2] = np.mean(
            measurement[:10].reshape(5, 2), axis=0
        )  # Initialize with the average position
        # Unsicherheit Matrix wird zurückgesetzt
        self.P = np.eye(4)
        return self.x[:2]

    def update(self, dt, measurement):
        # Extrahiere die fünf Positionsmessungen und die Standardabweichungen
        z = measurement[:10].reshape(5, 2)  # Extract the five position measurements
        R_values = measurement[10:]  # Extract the standard deviations
        # Erstelle die Diagonalmatrix der Messrauschkovarianzen
        R = np.diag(
            R_values
        )  # Create the diagonal matrix of measurement noise covariances

        # Zustandsübergangsmatrix für konstante Geschwindigkeit
        self.F = np.eye(4)  # State transition matrix
        self.F[0, 2] = dt  # Position x ändert sich mit Geschwindigkeit vx
        self.F[1, 3] = dt  # Position y ändert sich mit Geschwindigkeit vy

        # Vorhersage: Aktualisiere den Zustand und die Unsicherheit basierend auf dem Modell
        self.x = self.F @ self.x  # Predicted state estimate
        self.P = self.F @ self.P @ self.F.T  # Predicted covariance matrix

        # Berechnung der Residuen und der Jacobian-Matrix
        H = np.zeros((10, 4))
        for i in range(5):
            H[2 * i : 2 * i + 2, :2] = np.eye(2)  # Measurement matrix

        # Korrektur: Berechne die durchschnittliche Position und die Residuen
        z_mean = np.mean(z, axis=0)  # Average position
        y = (
            z_mean - self.x[:2]
        )  # Residuum basierend auf der durchschnittlichen Position

        # Skalieren der Residuen entsprechend der Standardabweichungen
        y = np.concatenate([y for _ in range(5)])

        # Berechne den Kalman-Gewinn
        S = H @ self.P @ H.T + R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Aktualisiere den Zustand basierend auf den Residuen und dem Kalman-Gewinn
        self.x = self.x + K @ y  # Updated state estimate

        # Aktualisiere die Unsicherheit basierend auf dem Kalman-Gewinn
        self.P = (self.I - K @ H) @ self.P  # Updated covariance matrix

        return self.x[:2]  # Rückgabe der geschätzten Position (x, y)


# Problem 6
class ConstantTurnFilter:
    def __init__(self, turn_rate=0, initial_uncertainty=500, process_noise_std=0.01):
        self.turn_rate = turn_rate  # Estimated turn rate
        self.process_noise_std = process_noise_std
        self.initial_uncertainty = initial_uncertainty
        self.Q = np.diag(
            [
                process_noise_std**2,
                process_noise_std**2,
                (process_noise_std * 10) ** 2,
                (process_noise_std * 10) ** 2,
            ]
        )  # scaled Process noise Covar Matrix Q
        self.state = np.zeros(4)  # Initial state (x, y, vx, vy)
        self.uncertainty = (
            np.eye(4) * initial_uncertainty
        )  # Initial Covar Matrix P, that we later gonna update with our prediction error

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
                self.turn_rate = (
                    self.state[2] * -self.state[3] + self.state[3] * self.state[2]
                ) / (v**2)

        # Calculating the Jocobi state transition matrix F. It calculates the new state based on the old state and turn rate.
        a_cos = np.cos(self.turn_rate * dt)
        a_sin = np.sin(self.turn_rate * dt)

        F = np.array(
            [
                [1, 0, dt * a_cos, -dt * a_sin],
                [0, 1, dt * a_sin, dt * a_cos],
                [0, 0, a_cos, -a_sin],
                [0, 0, a_sin, a_cos],
            ]
        )

        # Predict step for the estimated state and the uncertainty Covar Matrix P
        self.state = F @ self.state
        self.uncertainty = (
            F @ self.uncertainty @ F.T + self.Q * dt
        )  # we scale the process noise Matrix Q by dt

        # Extract measurements and their uncertainties
        H = np.zeros((10, 4))
        H[0::2, 0] = 1  # x measurements
        H[1::2, 1] = 1  # y measurements

        z = measurements[:10]  # First 10 elements are measurements
        R = np.diag(measurements[10:] ** 2)  # Last 10 elements are std deviations

        # Kalman gain
        innovation = z - H @ self.state
        S = H @ self.uncertainty @ H.T + R
        K = self.uncertainty @ H.T @ np.linalg.inv(S)

        # Update step
        self.state = self.state + K @ innovation
        self.uncertainty = (np.eye(4) - K @ H) @ self.uncertainty
        return self.state[:2]  # Return only the position (x, y)
