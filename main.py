import numpy as np
from numpy import random


# returns a random line of weights w0 + w1x + w2y = 0 in column vector (3 x 1) form
# The line is drawn between two random points [-1, 1]x[-1,1], fixing w2=1
def create_line():
    a = np.array([
                    [1, random.uniform(-1, 1), random.uniform(-1, 1)],
                    [1, random.uniform(-1, 1), random.uniform(-1, 1)],
                    [0, 0, 1]
                ])
    b = [[0], [0], [1]]
    weights = np.linalg.solve(a, b)
    return weights


# draws n points from a uniform distribution [-1, 1]x[-1,1] of dimension d
# fills an extra column of dummy 1 values for multiplication by w0 constant
def generate_sample(n, d):
    x = np.random.uniform(low=-1, high=1, size=(n, d + 1))
    x[:, 0] = 1
    return x


# classifies points as +1 or -1 based on sign of their product with weight vector
# n points with dim=m should be in (n x m) format.
# Weight should be in column vector (m x 1) format.
# Returns (n x 1) classifications
def classify_points(weight, points):
    y = np.sign(points @ weight)
    y[y == 0] = 1   # if any point is exactly 0, count as 1
    return y


def theta_func(weight, points):
    signal = points @ weight
    return np.exp(signal)/(1 + np.exp(signal))


# given w0, desired learning rate, and our data, returns w1 by stochastic gradient descent
def sgd_move(w, lr, x, y):

    # What do we do here?

    return w_new


# coordinates sgd algorithm
def gradient_descent_test(data, true_weights, true_probabilities, learning_rate, success_threshold, max_iterations):
    logistic_weights = np.array([[0], [0], [0]])
    # will always run at least once
    stop_condition = success_threshold + 1
    iterations = 0
    while stop_condition > success_threshold and iterations < max_iterations:
        rp_indices = np.random.choice(data.shape[0], size=data.shape[0], replace=False)
        original_weights = logistic_weights

        # for each point in array
        for rp_index in rp_indices:

            # What do we do here?

        stop_condition = np.linalg.norm(logistic_weights - original_weights)
        iterations += 1

    # calculate in_sample_error
    in_sample_error = 0
    for i in range(data.shape[0]):
        x = data[i, :]
        y = true_probabilities[i]
        in_sample_error += np.log(1 + np.exp(-y * np.transpose(logistic_weights) @ x))
    in_sample_error = in_sample_error / data.shape[0]
    return logistic_weights, in_sample_error, iterations


def e_out(w, true_weights, n):
    x = generate_sample(n, 2)
    y = classify_points(true_weights, x)
    error = 0
    for i in range(n):

        # What do we do here?

    error = error / n
    # # Uncomment to see error in terms of raw classification ability:
    # proposed_answers = theta_func(w, x)
    # error = np.count_nonzero(np.sign(proposed_answers - 0.5) != y) / n
    return error


def problem9():
    error_sum = 0
    epochs = 0
    trials = 10
    for x in range(trials):
        data = generate_sample(100, 2)
        true_weights = create_line()
        true_probabilities = classify_points(true_weights, data)
        learning_rate, success_threshold, max_iterations = 0.01, 0.01, 1000
        logistic_weights, in_sample_error, iterations = gradient_descent_test(
            data, true_weights, true_probabilities, learning_rate, success_threshold, max_iterations)
        out_of_sample_error = e_out(logistic_weights, true_weights, 1000)
        error_sum += out_of_sample_error
        epochs += iterations
    average_error = error_sum / trials
    average_epochs = epochs / trials
    print(f"Out of sample error: {average_error}, In sample error: {in_sample_error} \nEpochs to stopping condition: {average_epochs}")


if __name__ == '__main__':
    random.seed()
    problem9()