import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

"""
Proper credits to:
https://medium.com/@okanyenigun/step-by-step-guide-to-bayesian-optimization-a-python-based-approach-3558985c6818

"""

def blackbox_function(x):
    y = np.sin(x) + np.cos(2*x)
    return y

def expected_improvement(x, gp_model, best_y):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    z = (y_pred - best_y) / y_std
    ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

def upper_confidence_bound(x, gp_model, beta):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    ucb = y_pred + beta * y_std
    return ucb

def probability_of_improvement(x, gp_model, best_y):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    z = (y_pred - best_y) / y_std
    pi = norm.cdf(z)
    return pi

x_range = np.linspace(-2*np.pi, 2*np.pi, 100)
blackbox_output = blackbox_function(x_range)

num_samples = 10
sample_x = np.random.choice(x_range, size=num_samples)
sample_y = blackbox_function(sample_x)

kernel = RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel)

gp_model.fit(sample_x.reshape(-1, 1), sample_y)

y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)

best_idx = np.argmax(sample_y)
best_x = sample_x[best_idx]
best_y = sample_y[best_idx]

ei = expected_improvement(x_range, gp_model, best_y)

beta = 2.0

ucb = upper_confidence_bound(x_range, gp_model, beta)

pi = probability_of_improvement(x_range, gp_model, best_y)

num_iterations = 10

plt.figure(figsize=(10, 6))

""" UCB Method """
# for i in range(num_iterations):
#     gp_model.fit(sample_x.reshape(-1, 1), sample_y)
#     best_idx = np.argmax(sample_y)
#     best_x = sample_x[best_idx]
#     best_y = sample_y[best_idx]
    
#     beta = 2.0

#     ucb = upper_confidence_bound(x_range, gp_model, beta)

#     plt.plot(x_range, blackbox_function(x_range), color='black', label='Blackbox Function')
#     plt.plot(x_range, ucb, color='red', linestyle='dashed', label='Surrogate Function')
#     plt.scatter(sample_x, sample_y, color='blue', label='Prev Points')
#     if i < num_iterations - 1:
#         new_x = x_range[np.argmax(ucb)]
#         new_y = blackbox_function(new_x)
#         sample_x = np.append(sample_x, new_x)
#         sample_y = np.append(sample_y, new_y)
#         plt.scatter(new_x, new_y, color='green', label='New Points')
    
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(f'Iteration #{i+1}')
#     plt.legend()
#     plt.show()

""" EI Method """
for i in range(num_iterations):
    gp_model.fit(sample_x.reshape(-1, 1), sample_y)
    best_idx = np.argmax(sample_y)
    best_x = sample_x[best_idx]
    best_y = sample_y[best_idx]
    
    ei = expected_improvement(x_range, gp_model, best_y)
    
    surrogate_func = gp_model.predict(x_range.reshape(-1, 1)) + ei
    
    plt.plot(x_range, blackbox_function(x_range), color='black', label='Blackbox Function')
    plt.plot(x_range, surrogate_func, color='red', linestyle='dashed', label='Surrogate Function')
    plt.plot(x_range, ei, color='yellow', linestyle='dashed', label='Acquisition Function')
    plt.scatter(sample_x, sample_y, color='blue', label='Previous Samples')
    
    if i < num_iterations - 1:
        new_x = x_range[np.argmax(ei)]
        new_y = blackbox_function(new_x)
        
        if new_x not in sample_x:
            sample_x = np.append(sample_x, new_x)
            sample_y = np.append(sample_y, new_y)
            plt.scatter(new_x, new_y, color='green', label='New Sample')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Iteration #{i+1}')
    plt.legend()
    plt.show()

""" PI Method """
# for i in range(num_iterations):
#     gp_model.fit(sample_x.reshape(-1, 1), sample_y)
#     best_idx = np.argmax(sample_y)
#     best_x = sample_x[best_idx]
#     best_y = sample_y[best_idx]
    
#     pi = probability_of_improvement(x_range, gp_model, best_y)

#     plt.plot(x_range, blackbox_function(x_range), color='black', label='Blackbox Function')
#     plt.plot(x_range, pi, color='red', linestyle='dashed', label='Surrogate Function')
#     plt.scatter(sample_x, sample_y, color='blue', label='Prev Points')
#     if i < num_iterations - 1:
#         new_x = x_range[np.argmax(pi)]
#         new_y = blackbox_function(new_x)
#         sample_x = np.append(sample_x, new_x)
#         sample_y = np.append(sample_y, new_y)
#         plt.scatter(new_x, new_y, color='green', label='New Points')
    
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(f'Iteration #{i+1}')
#     plt.legend()
#     plt.show()