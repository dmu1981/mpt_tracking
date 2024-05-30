import numpy as np


def evaluate(target_states, results):
    scores = {}
    # Iterate all teams
    for teams in results.keys():
        # Calculate errors in state predictions
        errors = target_states - results[teams]

        # Calculate scroes
        scores[teams] = {
            "errors": np.sqrt(np.mean(np.power(errors, 2.0), axis=1)),
            "rmse": np.sqrt(np.mean(np.power(errors, 2.0))),
        }

    return scores
