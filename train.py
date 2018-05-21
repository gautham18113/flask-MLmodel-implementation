from preprocess import PreProcessing
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import warnings
import os
import dill as pickle
warnings.filterwarnings("ignore")
debug = False
filename = "model_v1.pk"


def train():
    data = pd.read_csv('{}\\data\\train.csv'.format(os.getcwd()))
    if debug:
        print(data.head())
    pred_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'],
                                                        test_size=0.25, random_state=42)
    y_train = y_train.replace({'Y': 1, 'N': 0}).as_matrix()
    y_test = y_test.replace({'Y': 1, 'N': 0}).as_matrix()
    pipe = make_pipeline(PreProcessing(), RandomForestClassifier())
    if debug:
        print(pipe)
    param_grid = {"randomforestclassifier__n_estimators": [10, 20, 30],
                  "randomforestclassifier__max_depth": [None, 6, 8, 10],
                  "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20],
                  "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)
    if debug:
        print("Validation set score: {:.2f}".format(grid.score(X_test, y_test)))

    # Load test dataset
    test_df = pd.read_csv('{}\\data\\test.csv'.format(os.getcwd()), encoding="utf-8")
    test_df = test_df.head()
    if debug:
        print(test_df)
    print(grid.predict(test_df))
    with open('%s\\model\\%s' % (os.getcwd(), filename),'wb') as file:
        pickle.dump(grid, file)
    with open('%s\\model\\%s' % (os.getcwd(), filename),'rb') as file:
        loaded_model = pickle.load(file)

    print(loaded_model.predict(test_df))