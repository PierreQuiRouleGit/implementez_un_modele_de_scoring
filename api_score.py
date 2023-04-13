import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for

import pickle


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '11111111111'


#data characteristics
def load_df():
    df = pd.read_csv("df_merge_sample.csv")
    return df

df = load_df()


#data predict
def load_df_test_copy():
    df_test_copy = pd.read_csv("df_test_sample.csv")
    return df_test_copy

df_test_copy = load_df_test_copy()


def load_df_test():
    df_test = pd.read_csv("df_test_transformed_sample.csv")
    df_test.drop(columns=['Unnamed: 0'],inplace=True)
    return df_test

df_test = load_df_test()



#best model
def load_model():
    pickle_model = open('model.pkl', 'rb') 
    clf = pickle.load(pickle_model)
    return clf

clf = load_model()


list_id = []
for i in range(len(df)):
    list_id.append(df['SK_ID_CURR'][i])

#target 0.5 data test
df.loc[df['TARGET'].isna(),'TARGET'] = 0.5


proba = clf.predict_proba(df_test)

proba_data = pd.DataFrame(proba).round(2)

proba_data.drop(columns=[0],inplace=True)

id = pd.DataFrame(df_test_copy['SK_ID_CURR'])

proba_id = [id,proba_data]

proba_id_concat = pd.concat(proba_id, axis=1, join='inner')


@app.route('/')
def home():
    return 'Hello world!'




@app.route('/credit/<id_client>')
def credit(id_client):
    
    return jsonify(SK_ID_CURR=id_client,
                score =proba_id_concat.loc[proba_id_concat['SK_ID_CURR']==int(id_client),1].values[0])
    

if __name__ == "__main__":
    app.run(debug=True)

