import copy

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import base64
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
class classification:
    def __init__(self):
        self.models = [LogisticRegression(), DecisionTreeClassifier(), XGBClassifier()]


    def convert_to_dataframe(self, response):

        """Converts the response from the API to a pandas dataframe"""
        # get the binary text
        decodedBytes = base64.b64decode(response.text.split(',')[1]).split(b'\r\n')
        columns = [column.split('_')[-1] for column in decodedBytes[0].decode('ascii').split(',')][1:]
        list_of_rows = []

        # Start from the next row
        for row in decodedBytes[1:]:
            # drop the first column corresponding the data ID
            decoded_row = row.decode('ascii').split(',')[1:]
            list_row = []
            for item in decoded_row:
                try:
                    item = float(item)
                except:
                    pass
                list_row.append(item)
            if len(list_row) == len(columns):
                list_of_rows.append(list_row)
        return pd.DataFrame(list_of_rows, columns=columns)

    def best_model(self):
        scoring = ['roc_auc']

        scores_dict = [cross_validate(model, self.X_train, self.y_train.values.ravel(), scoring=scoring) for model in self.models]
        scores = [item['test_' + scoring[0]].mean() for item in scores_dict]

        return np.argmax(scores)


    def feature_importance(self, data_X, target, chosen_model):
        categorical_columns = data_X.select_dtypes(include=['object', 'category']).columns

        categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        preprocessing = ColumnTransformer([("cat", categorical_encoder, categorical_columns),])

        # check if there are any categorical variables
        if categorical_columns.shape[0]:
            rf = Pipeline([("preprocess", preprocessing), ("classifier", chosen_model),])
        else:
            rf = chosen_model

        rf.fit(data_X, target)
        result = permutation_importance(rf, data_X, target, n_repeats=5, n_jobs=2)
        self.sorted_importances_idx = result.importances_mean.argsort()

        importances = pd.DataFrame(
            result.importances[self.sorted_importances_idx].T,
            columns=data_X.columns[self.sorted_importances_idx],
        )
        ax = importances.plot.box(vert=False, whis=10)
        ax.set_title("Prediction")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Importance in Prediction")
        ax.figure.tight_layout()
        ax.grid();
        ax.figure.savefig('feature_importance.png')
        return self.sorted_importances_idx

    def train(self, df, target):
        self.X = df.drop([target, 'id'], axis=1)
        self.y = df[target].replace({'Yes': 1, 'No': 0})

        featureX, self.X_test, featureY, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        self.X_train, self.y_train = pd.get_dummies(featureX, drop_first=True), pd.get_dummies(featureY, drop_first=True)
        self.X_test, self.y_test = pd.get_dummies(self.X_test, drop_first=True), pd.get_dummies(self.y_test, drop_first=True)

        self.X_train, self.X_test = StandardScaler().fit_transform(self.X_train), StandardScaler().fit_transform(self.X_test)

        index_best_model = self.best_model()

        inference_model = self.models[index_best_model]
        importance_model = copy.deepcopy(inference_model)

        inference_model.fit(self.X_train, self.y_train.values.ravel())
        self.feature_importance(featureX, featureY.values.ravel(), importance_model)
        return [{"Best Model is": "{}".format(inference_model)}, classification_report(self.y_test, inference_model.predict(self.X_test), output_dict=True)]

