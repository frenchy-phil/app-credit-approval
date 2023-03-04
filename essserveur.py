from flask import Flask, jsonify, request
import joblib
import numpy as np
import shap
import pickle
import pandas as pd
from lightgbm import LGBMClassifier
import json
from json import JSONEncoder

app = Flask(__name__)


model = pickle.load(open('model.pkl','rb'))
data=pd.read_csv('X_ressampled.csv')
valid_x=pd.read_csv('valid_x.csv')
listid=data['SK_ID_CURR'].tolist()


@app.route('/shap', methods=['POST'])
def graph_data():

     rowshap = request.get_json()
     
     shap_val_all=np.load('shap-values.npy')

    

     needed_shap_val=shap_val_all[rowshap].data.tolist()

  


     return jsonify({'shap_values':needed_shap_val})


@app.route('/predict', methods=['POST'])
def predict():
    iddic=request.get_json(force=True)
    idval = iddic.values()
    id = int(list(idval)[0])
    x=data.loc[data['SK_ID_CURR'] == id]
    y=model.predict_proba(x, num_iteration=model.best_iteration_)[:, 1]
    return jsonify(y.tolist())


if __name__ == '__main__':
    app.run(port=6000,debug=True)