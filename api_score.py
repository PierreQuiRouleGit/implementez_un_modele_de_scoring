import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time

import streamlit as st


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pickle

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope

import mlflow
from pyngrok import ngrok

import shap
import lime
from lime import lime_tabular


#data characteristics
@st.cache_data
def load_df():
    df = pd.read_csv("df_merge_sample.csv")
    return df

df = load_df()


#data predict
@st.cache_data
def load_df_test_copy():
    df_test_copy = pd.read_csv("df_test_sample.csv")
    return df_test_copy

df_test_copy = load_df_test_copy()


@st.cache_data
def load_df_test():
    df_test = pd.read_csv("df_test_transformed_sample.csv")
    df_test.drop(columns=['Unnamed: 0'],inplace=True)
    return df_test

df_test = load_df_test()



#best model
@st.cache_resource
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

proba = pd.DataFrame(proba)

proba.drop(columns=[0],inplace=True)

id = pd.DataFrame(df_test_copy['SK_ID_CURR'])

proba_id = [id,proba]

proba_id_concat = pd.concat(proba_id, axis=1, join='inner')

st.table(proba_id_concat)





