import dummy
import nofilter
import tip_constantposition
import tip_randomnoise
import tip_angular
import tip_constantvelocity
import tip_constantvelocity2
import tip_constantturn

# TODO: Add your filters here
filters = {
    "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    },
    "TIP": {
        "color": [0.169, 0.89, 0.725],
        "constantposition": tip_constantposition.Minimal_Variance_Fusion(2),
        "constantvelocity": tip_constantvelocity.KalmanFilter(),
        "constantvelocity2": tip_constantvelocity2.constantvelocity2_EKF(2),
        "constantturn": tip_constantturn.constantturn_EKF(2),
        "randomnoise": tip_randomnoise.KalmanFilter(2),
        "angular": tip_angular.angular_EKF(2),
    },
    # "NO": {
    #     "color": [1, 0.0, 0.0],
    #     "constantposition": nofilter.NoFilter(),
    #     "constantvelocity": nofilter.NoFilter(),
    #     "constantvelocity2": nofilter.NoFilter(),
    #     "constantturn": nofilter.NoFilter(),
    #     "randomnoise": nofilter.NoFilter(),
    #     "angular": nofilter.NoFilter(),
    # }
}
