
import numba as nb
import numpy as np

N_COEFFICIENTS = 5
N_PARAMETERS = 21


@nb.njit()
def cy_orthogonal_poly_fit_equidistant(binnos: nb.float64[:], y_values: nb.float64[:], y_errors: nb.float64[:]):
    weights = np.empty(y_errors.shape[0])
    parameters = np.empty((N_PARAMETERS, N_COEFFICIENTS))

    for j in range(y_errors.shape[0]):
        weights[j] = 1.0 / (y_errors[j] * y_errors[j])

    n_degrees = fit_orthogonal_poly_(binnos, y_values, weights, parameters)

    n_significant_parameters = significant_parameters_(parameters, n_degrees, len(y_values))
    n_degrees = reduce_to_signifcant_parameters(n_significant_parameters, parameters, n_degrees, len(y_values))
    return parameters, n_degrees


@nb.njit()
def fit_orthogonal_poly_(
    x: nb.float64[:],
    y: nb.float64[:],
    weights: nb.float64[:],
    parameters: nb.float64[:],
) -> nb.float64:
    n_supporting_points = x.shape[0]
    alpha = 0.0
    beta = 0.0
    gamma = 0.0
    delta = 0.0
    scratch = np.empty((2, x.shape[0]))

    for j in range(n_supporting_points):
        delta += weights[j] * y[j]
        gamma += weights[j]

    n_degrees = min(N_PARAMETERS, n_supporting_points)

    if gamma != 0.0:
        gamma = 1.0 / np.sqrt(gamma)

    delta *= gamma

    sum_squared = 0.0

    for k in range(n_supporting_points):
        tmp = y[k] - gamma * delta
        sum_squared += weights[k] * tmp * tmp
        scratch[0, k] = gamma
        scratch[1, k] = 0.0

    parameters[0, 0] = alpha
    parameters[0, 1] = beta
    parameters[0, 2] = gamma
    parameters[0, 3] = delta
    parameters[0, 4] = sum_squared

    # recursive loop for higher terms
    ii = 1
    for k in range(1, n_degrees):
        ii = 3 - ii
        alpha = 0.0
        beta = 0.0
        for j in range(0, n_supporting_points):
            alpha += weights[j] * x[j] * scratch[3 - ii - 1, j] * scratch[3 - ii - 1, j]
            beta += weights[j] * x[j] * scratch[ii - 1, j] * scratch[3 - ii - 1, j]

        gamma = 0.0
        for j in range(0, n_supporting_points):
            scratch[ii - 1, j] = (x[j] - alpha) * scratch[3 - ii - 1, j] - beta * scratch[ii - 1, j]
            gamma += weights[j] * scratch[ii - 1, j] * scratch[ii - 1, j]

        if gamma != 0.0:
            gamma = 1.0 / np.sqrt(gamma)

        delta = 0.0
        for j in range(0, n_supporting_points):
            scratch[ii - 1, j] *= gamma
            delta += weights[j] * scratch[ii - 1, j] * y[j]

        sum_squared = max(0.0, sum_squared - delta * delta)
        parameters[k, 0] = alpha
        parameters[k, 1] = beta
        parameters[k, 2] = gamma
        parameters[k, 3] = delta
        parameters[k, 4] = sum_squared
    parameters[0, 1] = 1.0
    parameters[0, 0] = np.nan
    return n_degrees


@nb.njit()
def significant_parameters_(parameters, n_degrees, n_supporting_points):
    result = n_degrees
    npp = 0

    if result < 3:
        return result

    for k in range(2, n_degrees):
        result = k + 1
        # chi^2 < ndf
        if parameters[k, 4] < n_supporting_points - (k + 1):
            break
        # 3.84 = chi2(parameters=0.95, ndf=1)
        if parameters[k, 3] ** 2 < 3.84:
            npp += 1
        # 5.99 = chi2(parameter_arr=0.95,ndf=2)
        if parameters[k, 3] ** 2 + parameters[k - 1, 3] ** 2 < 6.0:
            npp += 1
        if npp >= 4:
            break
    return result


@nb.njit()
def custom_clip(value, lower, upper):
    if value >= lower and value <= upper:
        return value
    elif value < lower:
        return lower
    elif value > upper:
        return upper


@nb.njit()
def reduce_to_signifcant_parameters(n_signi_par, parameters, n_degrees, n_supporting_points):
    n_degrees = min(n_degrees, n_signi_par)
    if n_degrees == n_supporting_points:
        if n_supporting_points > 2:
            n_degrees = custom_clip(int(n_supporting_points * 0.2), 2, N_PARAMETERS)
        else:
            n_degrees = 1

    # this line may have divisions by zero, which are also present in the
    # fortran equivalent
    diff = n_supporting_points - n_degrees
    if diff > 0.0:
        parameters[0, 1] = parameters[n_degrees - 1, 4] / diff

    return n_degrees


@nb.njit()
def cy_apply_orthogonal_poly_fit_equidistant(binnos, parameters, n_degrees):
    n_supporting_points = binnos.shape[0]
    result = np.empty(n_supporting_points)
    values = np.empty(2)

    for i in range(n_supporting_points):
        values[0] = parameters[0, 2]
        values[1] = 0.0
        y_estimate = parameters[0, 2] * parameters[0, 3]
        j = 1
        for k in range(1, n_degrees):
            j = 3 - j
            values[j - 1] = (
                (binnos[i] - parameters[k, 0]) * values[2 - j] - parameters[k, 1] * values[j - 1]
            ) * parameters[k, 2]
            y_estimate += parameters[k, 3] * values[j - 1]
        result[i] = y_estimate
    return result
