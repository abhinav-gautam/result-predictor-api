import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

results_train = pd.read_csv('Results_Train.csv')
results_test = pd.read_csv('Results_Test.csv')

X_train = results_train[['semester1_result', 'isBackLogPresent1',
                          'semester2_result', 'isBackLogPresent2']]
y_train = results_train.drop('name_index',axis=1)['overall_result']

#Model Training

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(X_train, y_train)
print('Model Trained!')

# Save your model
import joblib
joblib.dump(dtc, 'model1.pkl')
print("Model dumped!")

# Load the model that you just saved
dtc = joblib.load('model1.pkl')

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model1_columns.pkl')
print("Models columns dumped!")