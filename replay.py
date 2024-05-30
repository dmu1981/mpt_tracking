import numpy as np


def replay(filters, mode, timeseries):
    measurements = timeseries["measurements"]
    timestamps = timeseries["timestamps"]

    # Initialize results
    results = {}
    for teams in filters.keys():
        results[teams] = None

    # Now feed all measurements to filters
    last_t = 0
    index = 0
    for t in timestamps:
        if t == 0:
            # Reset all filters on first timestep
            try:
                for teams in filters.keys():
                    result = filters[teams][mode].reset(measurements[0, :])
                    if type(result) != np.ndarray:
                        raise Exception(
                            f"Your filter must return numpy array but it returned {type(result)}"
                        )
                    if result.shape != (2,):
                        raise Exception(
                            f"Your filter must return a 2-dimensional vector but it returned this shape: {result.shape}!"
                        )
                    results[teams] = result.reshape(1, -1)
            except Exception as e:
                print(
                    f"Error while reseting filter {filters[teams][mode]} of team {teams}"
                )
                print("Your filtered returned ", result)
                print("Exception was ", e)
                exit()
        else:
            # Update filters on subsequent timesteps
            for teams in filters.keys():
                try:
                    result = filters[teams][mode].update(
                        t - last_t, measurements[index, :]
                    )
                    if type(result) != np.ndarray:
                        raise Exception(
                            f"Your filter must return numpy array but it returned {type(result)}"
                        )
                    if result.shape != (2,):
                        raise Exception(
                            f"Your filter must return a 2-dimensional vector but it returned this shape: {result.shape}!"
                        )
                    res = result.reshape(1, -1)
                    results[teams] = np.concatenate([results[teams], res], axis=0)
                except Exception as e:
                    print(
                        f"Error while updating filter {filters[teams][mode]} of team {teams}"
                    )
                    print("Your filtered returned ", result)
                    print("Exception was ", e)
                    exit()

        last_t = t
        index += 1

    return results
