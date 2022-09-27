import pickle
from io import BytesIO

from fastapi import FastAPI
import uvicorn
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from flaml import AutoML
from pydantic import BaseModel
import numpy as np
# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    age: int
    Ca: int
    Chol: int
    Cp: int
    Exang: int
    Fbs: int
    OldPeak: float
    RestEcg: int
    sex: int
    Slope: int
    TrestBps: float
    Thal: int
    Thalac: int

class request_body_train(BaseModel):
    target_col: str
    train_time: int

# load data
import requests

url = "https://api.ignatius.io/api/report/export?reportId=ti2coyqg1&tableId=2363&exportType=csv&size=-1&tblName=1"

payload={}


headers={"Authorization": "Bearer X3p-Hum47YMiY8dBEw-OsQpSnVPcZRFdqtSRpx9eEdY"}
response = requests.request("GET", url, headers=headers, data=payload)

print(response.text.encode('utf8'))
# dt = np.dtype([('a', 'i4'), ('b', 'i4'), ('c', 'i4'), ('d', 'f4'), ('e', 'i4'),
#                ('f', 'i4'), ('g', 'i4'), ('h', 'i4'), ('i', 'i4'), ('j', 'i4'),
#                ('k', 'i4'), ('l', 'i4'), ('m', 'i4'), ('n', 'i4')])

# data = np.genfromtxt(response.content, delimiter=',', skip_header=1, dtype=dt)

# headers={"Authorization": "Bearer wgxt87joyqsMlB4AHUD9Zd0V3ITdFkEEzM5htGFnDkk"}


response = requests.request("GET", url, headers={"Authorization": "Bearer wgxt87joyqsMlB4AHUD9Zd0V3ITdFkEEzM5htGFnDkk"})

# df = pd.read_csv(BytesIO(response.text))
# print(df)


df = pd.read_csv('Table_2358_Report_5350.csv')
# initialize AutoML
automl = AutoML()


# define X and y
# Defining path operation for /name endpoint
@app.post('/train')
def train(data : request_body_train):
    target_col = data.target_col
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    automl.fit(X_train=X_train, y_train=y_train, time_budget=data.train_time, task='classification', verbose=0)
    pickle_out = open("automlmodel.pkl", "wb")
    pickle.dump(automl, pickle_out)
    pickle_out.close()
    return {'Model trained successfully. Best model accuracy is ' + str(accuracy_score(y_test, automl.predict(X_test)))}



# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.age,
        data.Ca,
        data.Chol,
        data.Cp,
        data.Exang,
        data.Fbs,
        data.OldPeak,
        data.RestEcg,
        data.sex,
        data.Slope,
        data.TrestBps,
        data.Thal,
        data.Thalac
    ]]
    
    df = pd.DataFrame(test_data, columns=['Age', 'ca Value', 'Chol', 'CP Value', 'exang', 'FBS Value', 'oldpeak',
                                'Rest CG', 'Sex', 'slope', 'Test BPS', 'thal', 'thalach'])

    # Predicting the Class
    trained_model = pickle.load(open('automlmodel.pkl', 'rb'))
    class_idx = trained_model.predict(df)
    # Return the Result
    return {'This person has heart disease.'} if class_idx else {'This person does not have heart disease.'}

# {
#   "age": 63,
#   "Ca": 0,
#   "Chol": 233,
#   "Cp": 3,
#   "Exang": 0,
#   "Fbs": 1,
#   "OldPeak": 2.3,
#   "RestEcg": 0,
#   "sex": 1,
#   "Slope": 0,
#   "TrestBps": 145,
#   "Thal": 1,
#   "Thalac": 150
# }