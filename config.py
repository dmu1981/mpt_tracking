import dummy
import constanposition
import randomnoise

# TODO: Add your filters here
filters = {
    "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": constanposition.KalmanFilter(2), #KalmanFilter
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": randomnoise.RandomNoiseFilter(2),
        "angular": dummy.DummyFilter(2),
    }
}
# Aufruf mit python main.py --mode=constantposition --index=5
# Visualisierung python main.py --mode=constantposition --debug