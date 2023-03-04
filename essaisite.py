import streamlit as st
import requests
import shap
import streamlit.components.v1 as components
import pandas as pd
import json
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 

data=pd.read_csv('X_resampled3.csv')
listid=data['SK_ID_CURR'].tolist()
valid_x=pd.read_csv('valid_x2.csv')

endpoint='http://frenchyphil.pythonanywhere.com/'

def score(id):
    response = requests.post(endpoint+'/api/predict', json={'text': id})
    score = response.json()
    return score



st.title("credit prediction")
st.image('shap_summary.png')
id_input = st.selectbox("Choisissez l'identifiant d'un client", data.SK_ID_CURR)

result=score(id_input)
st.metric(label= 'probabilite de remboursement', value=1-result[0])

response_shapley = requests.post(endpoint+'/api/shap', json = data.query(f'SK_ID_CURR == {id_input}').index.values.tolist()[0])
decodedArrays = json.loads(response_shapley.text)



shap_v=np.array(decodedArrays['shap_values'])
shap_v_1= np.array(shap_v, dtype=float)
shap_v_2= np.reshape(shap_v_1, (368,1))

expected_values=np.load('expected-val.npy').item()

st.title('Graphe de decision')
st.set_option('deprecation.showPyplotGlobalUse', False)
p=shap.decision_plot(expected_values, shap_v_1, valid_x, ignore_warnings=True)
st.pyplot(p)

enumeration=['NAME_INCOME_TYPE_Working','CODE_GENDER_M', 'NAME_FAMILY_STATUS_Married', 'REGION_RATING_CLIENT_W_CITY', 'AMT_CREDIT' ]
fig, ax = plt.subplots()
sns.boxplot(data=data[enumeration], ax = ax, flierprops={"marker": "x"}, color='skyblue', showcaps=True)
plt.setp(ax.get_xticklabels(), rotation=90)
client_data=data.query(f'SK_ID_CURR == {id_input}')
  
for k in  enumeration:
    ax.scatter(k, client_data[k].values, marker='X', s=100, color = 'black', label = 'Client selectionné')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
st.pyplot(fig)

