import constanposition
import constantvelocity
import randomnoise
import constantturn
import angular_kalman
import constantvelocity2

# TODO: Add your filters here

filters = {
    "Glotzkowski": {
        "color": [0.5, 0.3, 0.9],
        "constantposition": constanposition.KalmanFilter(2),
        "constantvelocity": constantvelocity.KalmanFilter(2),
        "constantvelocity2": constantvelocity2.ConstantVelocityKalmanFilter2(4, 2),
        "constantturn": constantturn.Constantturn_KalmanFilter(4, 20),        
        "randomnoise": randomnoise.Randomnoise_KalmanFilter(2),
        "angular": angular_kalman.AngularKalmanFilter(2)
    }
}

# Aufruf mit python main.py --mode=constantposition --index=5
# Visualisierung python main.py --mode=constantposition --debug