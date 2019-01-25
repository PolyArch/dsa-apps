import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time

# Fetch dataset using sklearn
cov = fetch_covtype()
#numpy array of NxM
X = cov.data
#numpy array of Nx1
y = cov.target

# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                    random_state=42)

# Specify sufficient boosting iterations to reach a minimum
num_round = 10

# Leave most parameters as default
param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 8, # Number of possible output classes
         'tree_method': 'approx', # Use GPU accelerated algorithm
         'nthread': 2
         }

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

gpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))
text_file = open("Output.txt", "w")
text_file.write("GPU Training Time: %s seconds" % (str(time.time() - tmp)) + '\n')

# Repeat for CPU algorithm
#tmp = time.time()
#param['tree_method'] = 'hist'
#cpu_res = {}
#xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
#print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
#text_file.write("CPU Training Time: %s seconds" % (str(time.time() - tmp)))

text_file.close()
