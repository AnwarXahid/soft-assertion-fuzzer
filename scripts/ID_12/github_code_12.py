import numpy as np
from sklearn.metrics import root_mean_squared_log_error, mean_squared_log_error

"""
Source: https://github.com/scikit-learn/scikit-learn/issues/29678
"""

y_true = [0, 0.25, 0.1]
y_pred = [1, 0.5, 0.9]

# Hand calculation of RMSLE is valid
RMSLE = (1 / len(y_pred) * sum([(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2 for i in range(len(y_pred))])) ** 0.5

# Hand calculation of MSLE is valid
MSLE = 1 / len(y_pred) * sum([(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2 for i in range(len(y_pred))])

# Error is raised with sklearn
print(root_mean_squared_log_error(y_true, y_pred))

# Error is raised with sklearn
print(mean_squared_log_error(y_true, y_pred))