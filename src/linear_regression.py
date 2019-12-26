import numpy as np


def least_squares_weights(input_x, target_y):
    """Calculate linear regression least squares weights.

    Positional arguments:
        input_x -- matrix of training input data
        target_y -- vector of training output values

        The dimensions of X and y will be either p-by-n and 1-by-n
        Or n-by-p and n-by-1

    Example:
        import numpy as np
        training_y = np.array([[208500, 181500, 223500,
                                140000, 250000, 143000,
                                307000, 200000, 129900,
                                118000]])
        training_x = np.array([[1710, 1262, 1786,
                                1717, 2198, 1362,
                                1694, 2090, 1774,
                                1077],
                               [2003, 1976, 2001,
                                1915, 2000, 1993,
                                2004, 1973, 1931,
                                1939]])
        weights = least_squares_weights(training_x, training_y)

        print(weights)  #--> np.array([[-2.29223802e+06],
                           [ 5.92536529e+01],
                           [ 1.20780450e+03]])

        print(weights[1][0]) #--> 59.25365290008861

    Assumptions:
        -- target_y is a vector whose length is the same as the
        number of observations in training_x
    """
    # Check rows > columns and if not transpose data
    if input_x.shape[0] < input_x.shape[1]:
        input_x = input_x.T

    if target_y.shape[0] < target_y.shape[1]:
        target_y = target_y.T

    # create a n x 1 column of ones to prepend to input_x
    new_col = np.ones((input_x.shape[0]))
    # prepend the 'new-col' to the zero vector in input_x
    input_x = np.insert(input_x, 0, new_col, axis=1)

    # calculate weights - Next you'll implement the equation above for $w_{LS}$ using the inverse matrix function.
    # %w_{LS} = (X^T X)^{−1}X^Ty
    wls1 = np.linalg.inv(np.matmul(input_x.T, input_x))
    wls2 = np.matmul(input_x.T, target_y)
    weights = np.matmul(wls1, wls2)

    return weights
