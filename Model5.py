import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

results_train = pd.read_csv('Results_Train.csv')
results_test = pd.read_csv('Results_Test.csv')

X_train = results_train[['semester1_result', 'isBackLogPresent1',
                          'semester2_result', 'isBackLogPresent2',
                          'semester3_result', 'isBackLogPresent3',
                          'semester4_result', 'isBackLogPresent4',
                          'semester5_result', 'isBackLogPresent5',
                          'semester6_result', 'isBackLogPresent6']]
y_train = results_train.drop('name_index',axis=1)['overall_result']

#Model Training

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(X_train, y_train)
print('Model Trained!')

# Save your model
import joblib
joblib.dump(dtc, 'model5.pkl')
print("Model dumped!")

# Load the model that you just saved
dtc = joblib.load('model5.pkl')

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model5_columns.pkl')
print("Models columns dumped!")