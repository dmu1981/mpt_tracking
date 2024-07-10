import dummy
import constanposition
import constantvelocity
import randomnoise

# TODO: Add your filters here

filters = {
    "GLoTzKoWsKi":{
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": constantvelocity.KalmanFilter(2),
        "constantvelocity2": constantvelocity.KalmanFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    }
}
# Aufruf mit python main.py --mode=constantposition --index=5
# Visualisierung python main.py --mode=constantposition --debug