import dummy
import constanposition
import constantvelocity
import randomnoise
import constantturn
import angular_kalman
import constantvelocity2

# TODO: Add your filters here

filters = {
    "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": constantvelocity.KalmanFilter(2),
        "constantvelocity2": constantvelocity.KalmanFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    },
    "Glotzkowski": {
        "color": [0.5, 0.3, 0.9],
        "constantposition": constanposition.KalmanFilter(2),
        "randomnoise": randomnoise.KalmanFilter(2),
        "constantturn": constantturn.ExtendedKalmanFilter(4, 20)
    }

}

# Aufruf mit python main.py --mode=constantposition --index=5
# Visualisierung python main.py --mode=constantposition --debug