import glotzkowski

filters = {
    "Glotzkowski": {
        "color": [0.5, 0.3, 0.9],
        "constantposition": glotzkowski.Constantposition_KalmanFilter(2),
        "constantvelocity": glotzkowski.Constantvelocity_KalmanFilter(2),
        "constantvelocity2": glotzkowski.ConstantVelocityKalmanFilter2(4, 2),
        "constantturn": glotzkowski.Constantturn_KalmanFilter(4, 20),        
        "randomnoise": glotzkowski.Randomnoise_KalmanFilter(2),
        "angular": glotzkowski.AngularKalmanFilter(2)
    }
}

# Aufruf mit python main.py --mode=constantposition --index=5
# Visualisierung python main.py --mode=constantposition --debug