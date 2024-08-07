import dummy
import NWFE
import numpy as np
import nikolucas_filters
import till_jonah
import mueller
import nam
from filters import (
    KalmanFilter,
    KalmanFilterRandomNoise,
    KalmanFilterAngular,
    KalmanFilterConstantTurn,
    ConstantVelocity2,
    ConstantVelocity,
)
import glotzkowski
import filter_nick_redion
import nofilter
import tip_constantposition
import tip_randomnoise
import tip_angular
import tip_constantvelocity
import tip_constantvelocity2
import tip_constantturn

# TODO: Add your filters here
filters = {
    "Mueller": {
      "color": [0.4, 0.7, 0.9],
      "constantposition": mueller.ConstantPositionFilter(),
      "constantvelocity": mueller.ConstantVelocityFilter(),
      "constantvelocity2": mueller.ConstantVelocityFilter2(),
      "constantturn": mueller.ConstantTurnRate(),
      "randomnoise": mueller.RandomNoiseFilter(),
      "angular": mueller.AngularFilter(),
    }, 
    "Glotzkowski": {
        "color": [0.5, 0.3, 0.9],
        "constantposition": glotzkowski.Constantposition_KalmanFilter(2),
        "constantvelocity": glotzkowski.Constantvelocity_KalmanFilter(2),
        "constantvelocity2": glotzkowski.ConstantVelocityKalmanFilter2(4, 2),
        "constantturn": glotzkowski.Constantturn_KalmanFilter(4, 20),        
        "randomnoise": glotzkowski.Randomnoise_KalmanFilter(2),
        "angular": glotzkowski.AngularKalmanFilter(2)
    },
    "Redion_Nick": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": filter_nick_redion.KalmanFilter(2),
        "constantvelocity": dummy.DummyFilter(2), # Provided filter crashes #filter_nick_redion.KalmanFilterConstantVelocity(2),
        "constantvelocity2": filter_nick_redion.ConstantVelocity(2),
        "constantturn": filter_nick_redion.ConstantTurn(2),
        "randomnoise": filter_nick_redion.RandomNoise(2),
        "angular": filter_nick_redion.ExtendedKalmanFilter(2),
    },
    "MeMaMa": {
        "color": [0.2, 0.2, 0.6],
        "constantposition": KalmanFilter((2,)),
        "randomnoise": KalmanFilterRandomNoise(2),
        "angular": KalmanFilterAngular(),
        "constantvelocity2": ConstantVelocity2(),
        "constantvelocity": ConstantVelocity(),
        "constantturn": KalmanFilterConstantTurn(),
    },
    "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    },
    "NAMTeam": {
        "color": [0.5, 0.1, 0.9],
        "constantposition": nam.ConstantpositionFilter(),
        "randomnoise": nam.RandomNoiseFilter(),
        "angular": nam.AngularFilter(),
        "constantvelocity": nam.KalmanFilterConstantVelocity(),
        "constantvelocity2": nam.KalmanFilterConstantVelocityMultiple(),
        "constantturn": nam.ConstantTurnFilter(),
    },
    "Kalman (Fabrice, Niels)": {
        "color": [1.0, 0.0, 0.0],
        "constantposition": NWFE.KalmanFilter(),
        "randomnoise": NWFE.FilterRandomNoise(),
        "angular": NWFE.AngularKalmanFilter(),
        "constantvelocity": NWFE.ConstantVelocityKalmanFilter(),
        "constantvelocity2": NWFE.ConstantVelocityMultiMeasurementKalmanFilter(),
        "constantturn": NWFE.ConstantTurnRateKalmanFilter(),
    },
    "Nicolucas": {
        "color": [0.6, 0.6, 0.2],
        "constantposition": nikolucas_filters.KalmanFilter(),
        "constantvelocity": nikolucas_filters.ConstantVelocityKalmanFilter(),
        "constantvelocity2": nikolucas_filters.ConstantVelocity2(),
        "constantturn": nikolucas_filters.ConstantTurnRateFilter(4),
        "randomnoise": nikolucas_filters.RandomNoise(2, 2),
        "angular": nikolucas_filters.ExtendedKalmanFilter(),
    },
    "die süßen Mäuse": {
        "color": [0.5, 0.0, 0.7],
        "constantposition": till_jonah.Constant(2),
        "randomnoise": till_jonah.Random(2),
        "angular": till_jonah.Angular(2),
        "constantvelocity": till_jonah.Velocity(2),
        "constantvelocity2": till_jonah.Velocity2(2),
        "constantturn": till_jonah.ConstantTurn(2)
    },
    "TIP": {
        "color": [0.169, 0.89, 0.725],
        "constantposition": tip_constantposition.Minimal_Variance_Fusion(2),
        "constantvelocity": tip_constantvelocity.KalmanFilter(2),
        "constantvelocity2": tip_constantvelocity2.constantvelocity2_EKF(2),
        "constantturn": tip_constantturn.constantturn_EKF(2),
        "randomnoise": tip_randomnoise.KalmanFilter(2),
        "angular": tip_angular.angular_EKF(2),
    }
}
