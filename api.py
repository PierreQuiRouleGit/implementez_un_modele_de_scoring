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

#id
input_id = st.number_input('Write SK_ID_CURR',format="%i")

if input_id in list_id:
    if (df.loc[df['SK_ID_CURR']==int(input_id),'TARGET'].values[0] == 1.0) or (df.loc[df['SK_ID_CURR']==int(input_id),'TARGET'].values[0] == 0.0) :
        st.subheader('Load Validation(0 good - 1 bad)')
        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(df.loc[df['SK_ID_CURR']==int(input_id),'TARGET'].values[0],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = { 'axis': {'range': [0, 1]},
            'bar' :{'color': "black"},
            'steps' : [
                {'range': [0, 0.33], 'color': "green"},
                {'range': [0.33, 0.66], 'color': "yellow"},
                {'range': [0.66, 1], 'color': "red"}]
        
        },
        title = {'text': "Score"}))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Status')
            

            st.write("Gender : ", df.loc[df['SK_ID_CURR']==int(input_id),'CODE_GENDER'].values[0])
            st.write("Age : {:.0f}".format(int(df.loc[df['SK_ID_CURR']==int(input_id),'DAYS_BIRTH'].values[0]/-365)))
            st.write("Family status : ", df.loc[df['SK_ID_CURR']==int(input_id),'NAME_FAMILY_STATUS'].values[0])
            st.write("Number of children : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'CNT_CHILDREN'].values[0]))



            data_age = round((df["DAYS_BIRTH"]/-365), 2)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, bins=20)
            ax.axvline(int(df.loc[df['SK_ID_CURR']==int(input_id),'DAYS_BIRTH'].values[0]/-365), color="green", linestyle='--')
            ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
            st.pyplot(fig)

        with col2:
            st.subheader("Income (USD)")
            st.write("Income total : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_INCOME_TOTAL'].values[0]))
            st.write("Credit amount : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_CREDIT_x'].values[0]))
            st.write("Credit annuities : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_ANNUITY_x'].values[0]))
            st.write("Amount of property for credit : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_GOODS_PRICE_x'].values[0]))
            

            df_income = df.loc[df['AMT_INCOME_TOTAL'] < 500000, :]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_income["AMT_INCOME_TOTAL"], bins=25)
            ax.axvline(int(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_INCOME_TOTAL'].values[0]), color="green", linestyle='--')
            ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
            st.pyplot(fig)

        st.subheader("----"*20)

        colonnes = [
            'None','TARGET',
            'NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY',
            'CNT_CHILDREN',
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT_x',
            'AMT_ANNUITY_x',
            'AMT_GOODS_PRICE_x',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'REGION_POPULATION_RELATIVE',
            'DAYS_BIRTH',
            'DAYS_EMPLOYED',
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
            'OWN_CAR_AGE',
            'FLAG_MOBIL',
            'FLAG_EMP_PHONE',
            'FLAG_WORK_PHONE',
            'FLAG_CONT_MOBILE',
            'FLAG_PHONE',
            'FLAG_EMAIL',
            'OCCUPATION_TYPE',
            'CNT_FAM_MEMBERS',
            'REGION_RATING_CLIENT',
            'REGION_RATING_CLIENT_W_CITY',
            'WEEKDAY_APPR_PROCESS_START',
            'HOUR_APPR_PROCESS_START_x',
            'REG_REGION_NOT_LIVE_REGION',
            'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY',
            'ORGANIZATION_TYPE',
            'EXT_SOURCE_1',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3',
            'APARTMENTS_AVG',
            'BASEMENTAREA_AVG',
            'YEARS_BEGINEXPLUATATION_AVG',
            'YEARS_BUILD_AVG',
            'COMMONAREA_AVG',
            'ELEVATORS_AVG',
            'ENTRANCES_AVG',
            'FLOORSMAX_AVG',
            'FLOORSMIN_AVG',
            'LANDAREA_AVG',
            'LIVINGAPARTMENTS_AVG',
            'LIVINGAREA_AVG',
            'NONLIVINGAPARTMENTS_AVG',
            'NONLIVINGAREA_AVG',
            'APARTMENTS_MODE',
            'BASEMENTAREA_MODE',
            'YEARS_BEGINEXPLUATATION_MODE',
            'YEARS_BUILD_MODE',
            'COMMONAREA_MODE',
            'ELEVATORS_MODE',
            'ENTRANCES_MODE',
            'FLOORSMAX_MODE',
            'FLOORSMIN_MODE',
            'LANDAREA_MODE',
            'LIVINGAPARTMENTS_MODE',
            'LIVINGAREA_MODE',
            'NONLIVINGAPARTMENTS_MODE',
            'NONLIVINGAREA_MODE',
            'APARTMENTS_MEDI',
            'BASEMENTAREA_MEDI',
            'YEARS_BEGINEXPLUATATION_MEDI',
            'YEARS_BUILD_MEDI',
            'COMMONAREA_MEDI',
            'ELEVATORS_MEDI',
            'ENTRANCES_MEDI',
            'FLOORSMAX_MEDI',
            'FLOORSMIN_MEDI',
            'LANDAREA_MEDI',
            'LIVINGAPARTMENTS_MEDI',
            'LIVINGAREA_MEDI',
            'NONLIVINGAPARTMENTS_MEDI',
            'NONLIVINGAREA_MEDI',
            'FONDKAPREMONT_MODE',
            'HOUSETYPE_MODE',
            'TOTALAREA_MODE',
            'WALLSMATERIAL_MODE',
            'EMERGENCYSTATE_MODE',
            'OBS_30_CNT_SOCIAL_CIRCLE',
            'DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE',
            'DEF_60_CNT_SOCIAL_CIRCLE',
            'DAYS_LAST_PHONE_CHANGE',
            'FLAG_DOCUMENT_2',
            'FLAG_DOCUMENT_3',
            'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5',
            'FLAG_DOCUMENT_6',
            'FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_8',
            'FLAG_DOCUMENT_9',
            'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11',
            'FLAG_DOCUMENT_12',
            'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14',
            'FLAG_DOCUMENT_15',
            'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_18',
            'FLAG_DOCUMENT_19',
            'FLAG_DOCUMENT_20',
            'FLAG_DOCUMENT_21',
            'AMT_REQ_CREDIT_BUREAU_HOUR',
            'AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT',
            'AMT_REQ_CREDIT_BUREAU_YEAR',
            'DAYS_EMPLOYED_PERC',
            'INCOME_CREDIT_PERC',
            'INCOME_PER_PERSON',
            'ANNUITY_INCOME_PERC',
            'PAYMENT_RATE',
            'PREVIOUS_LOANS_COUNT',
            'DAYS_CREDIT',
            'CREDIT_DAY_OVERDUE',
            'DAYS_CREDIT_ENDDATE',
            'DAYS_ENDDATE_FACT',
            'AMT_CREDIT_MAX_OVERDUE',
            'CNT_CREDIT_PROLONG',
            'AMT_CREDIT_SUM',
            'AMT_CREDIT_SUM_DEBT',
            'AMT_CREDIT_SUM_LIMIT',
            'AMT_CREDIT_SUM_OVERDUE',
            'DAYS_CREDIT_UPDATE',
            'AMT_ANNUITY_y',
            'MONTHS_BALANCE_MEAN',
            'PREVIOUS_APPLICATION_COUNT',
            'SK_ID_PREV',
            'AMT_ANNUITY',
            'AMT_APPLICATION',
            'AMT_CREDIT_y',
            'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE_y',
            'HOUR_APPR_PROCESS_START_y',
            'NFLAG_LAST_APPL_IN_DAY',
            'RATE_DOWN_PAYMENT',
            'RATE_INTEREST_PRIMARY',
            'RATE_INTEREST_PRIVILEGED',
            'DAYS_DECISION',
            'SELLERPLACE_AREA',
            'CNT_PAYMENT',
            'DAYS_FIRST_DRAWING',
            'DAYS_FIRST_DUE',
            'DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_LAST_DUE',
            'DAYS_TERMINATION',
            'NFLAG_INSURED_ON_APPROVAL',
            'MONTHS_BALANCE_x',
            'AMT_BALANCE',
            'AMT_CREDIT_LIMIT_ACTUAL',
            'AMT_DRAWINGS_ATM_CURRENT',
            'AMT_DRAWINGS_CURRENT',
            'AMT_DRAWINGS_OTHER_CURRENT',
            'AMT_DRAWINGS_POS_CURRENT',
            'AMT_INST_MIN_REGULARITY',
            'AMT_PAYMENT_CURRENT',
            'AMT_PAYMENT_TOTAL_CURRENT',
            'AMT_RECEIVABLE_PRINCIPAL',
            'AMT_RECIVABLE',
            'AMT_TOTAL_RECEIVABLE',
            'CNT_DRAWINGS_ATM_CURRENT',
            'CNT_DRAWINGS_CURRENT',
            'CNT_DRAWINGS_OTHER_CURRENT',
            'CNT_DRAWINGS_POS_CURRENT',
            'CNT_INSTALMENT_MATURE_CUM',
            'SK_DPD_x',
            'SK_DPD_DEF_x',
            'NUM_INSTALMENT_VERSION',
            'NUM_INSTALMENT_NUMBER',
            'DAYS_INSTALMENT',
            'DAYS_ENTRY_PAYMENT',
            'AMT_INSTALMENT',
            'AMT_PAYMENT',
            'MONTHS_BALANCE_y',
            'CNT_INSTALMENT',
            'CNT_INSTALMENT_FUTURE',
            'SK_DPD_y',
            'SK_DPD_DEF_y'
        ]

        option = st.selectbox(
        'Select a variable ?',
        options = colonnes)
        
        st.write('You selected:', option)
        
        if option == 'None':
            st.write('Waiting for you')
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[str(option)], bins=20)
            ax.axvline(df.loc[df['SK_ID_CURR']==int(input_id),str(option)].values[0], color="green", linestyle='--')
            ax.set(title=str(option), xlabel=str(option), ylabel='')
            st.pyplot(fig)

        st.subheader("----"*20)

        option2 = st.selectbox(
        'Select a first variable ?',
        options = colonnes)

        option3 = st.selectbox(
        'Select a second variable ?',
        options = colonnes)

        if option2 == 'None' or option3 == 'None':
            st.write('Waiting for you')
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df,x=str(option3), y=str(option2))
            ax.axvline(df.loc[df['SK_ID_CURR']==int(input_id),str(option3)].values[0], color="green", linestyle='--')
            ax.axhline(df.loc[df['SK_ID_CURR']==int(input_id),str(option2)].values[0], color="green", linestyle='--')
            ax.set(title='Representation of ' + str(option2) + ' by ' + str(option3), xlabel=str(option3), ylabel=str(option2))
            st.pyplot(fig)
        

    else:
        
        st.subheader('Load Validation(0 good - 1 bad)')
        proba = clf.predict_proba(df_test)

        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(proba[df_test_copy.loc[df_test_copy['SK_ID_CURR']==int(input_id)].index.item()][0],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = { 'axis': {'range': [0, 1]},
            'bar' :{'color': "black"},
            'steps' : [
                {'range': [0, 0.33], 'color': "green"},
                {'range': [0.33, 0.66], 'color': "yellow"},
                {'range': [0.66, 1], 'color': "red"}]
        
        },
        title = {'text': "Score"}))
        st.plotly_chart(fig, use_container_width=True)
        

        st.subheader('Feature importances')
        feature_importance = (
                    pd.DataFrame(
                    {
                        'variable': df_test.columns,
                        'coefficient' : clf.coef_[0]
                    }
                    )
                    .round(decimals=2)  \
                    .sort_values('coefficient',ascending=False)  \
                    .style.bar(color=['red','green'],align='zero')
                    )
        st.write(feature_importance)

        explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(df_test),feature_names=df_test.columns,class_names=['target'],mode='regression')
        exp = explainer.explain_instance(data_row=df_test.iloc[df_test_copy.loc[df_test_copy['SK_ID_CURR']==int(input_id)].index.item()], predict_fn=clf.predict_proba)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        exp.as_pyplot_figure()
        st.pyplot()


        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Status')
            

            st.write("Gender : ", df.loc[df['SK_ID_CURR']==int(input_id),'CODE_GENDER'].values[0])
            st.write("Age : {:.0f}".format(int(df.loc[df['SK_ID_CURR']==int(input_id),'DAYS_BIRTH'].values[0]/-365)))
            st.write("Family status : ", df.loc[df['SK_ID_CURR']==int(input_id),'NAME_FAMILY_STATUS'].values[0])
            st.write("Number of children : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'CNT_CHILDREN'].values[0]))



            data_age = round((df["DAYS_BIRTH"]/-365), 2)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, bins=20)
            ax.axvline(int(df.loc[df['SK_ID_CURR']==int(input_id),'DAYS_BIRTH'].values[0]/-365), color="green", linestyle='--')
            ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
            st.pyplot(fig)

        with col2:
            st.subheader("Income (USD)")
            st.write("Income total : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_INCOME_TOTAL'].values[0]))
            st.write("Credit amount : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_CREDIT_x'].values[0]))
            st.write("Credit annuities : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_ANNUITY_x'].values[0]))
            st.write("Amount of property for credit : {:.0f}".format(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_GOODS_PRICE_x'].values[0]))
            

            df_income = df.loc[df['AMT_INCOME_TOTAL'] < 500000, :]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_income["AMT_INCOME_TOTAL"], bins=25)
            ax.axvline(int(df.loc[df['SK_ID_CURR']==int(input_id),'AMT_INCOME_TOTAL'].values[0]), color="green", linestyle='--')
            ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
            st.pyplot(fig)

        st.subheader("----"*20)

        colonnes = [
            'None','TARGET',
            'NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY',
            'CNT_CHILDREN',
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT_x',
            'AMT_ANNUITY_x',
            'AMT_GOODS_PRICE_x',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'REGION_POPULATION_RELATIVE',
            'DAYS_BIRTH',
            'DAYS_EMPLOYED',
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
            'OWN_CAR_AGE',
            'FLAG_MOBIL',
            'FLAG_EMP_PHONE',
            'FLAG_WORK_PHONE',
            'FLAG_CONT_MOBILE',
            'FLAG_PHONE',
            'FLAG_EMAIL',
            'OCCUPATION_TYPE',
            'CNT_FAM_MEMBERS',
            'REGION_RATING_CLIENT',
            'REGION_RATING_CLIENT_W_CITY',
            'WEEKDAY_APPR_PROCESS_START',
            'HOUR_APPR_PROCESS_START_x',
            'REG_REGION_NOT_LIVE_REGION',
            'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY',
            'ORGANIZATION_TYPE',
            'EXT_SOURCE_1',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3',
            'APARTMENTS_AVG',
            'BASEMENTAREA_AVG',
            'YEARS_BEGINEXPLUATATION_AVG',
            'YEARS_BUILD_AVG',
            'COMMONAREA_AVG',
            'ELEVATORS_AVG',
            'ENTRANCES_AVG',
            'FLOORSMAX_AVG',
            'FLOORSMIN_AVG',
            'LANDAREA_AVG',
            'LIVINGAPARTMENTS_AVG',
            'LIVINGAREA_AVG',
            'NONLIVINGAPARTMENTS_AVG',
            'NONLIVINGAREA_AVG',
            'APARTMENTS_MODE',
            'BASEMENTAREA_MODE',
            'YEARS_BEGINEXPLUATATION_MODE',
            'YEARS_BUILD_MODE',
            'COMMONAREA_MODE',
            'ELEVATORS_MODE',
            'ENTRANCES_MODE',
            'FLOORSMAX_MODE',
            'FLOORSMIN_MODE',
            'LANDAREA_MODE',
            'LIVINGAPARTMENTS_MODE',
            'LIVINGAREA_MODE',
            'NONLIVINGAPARTMENTS_MODE',
            'NONLIVINGAREA_MODE',
            'APARTMENTS_MEDI',
            'BASEMENTAREA_MEDI',
            'YEARS_BEGINEXPLUATATION_MEDI',
            'YEARS_BUILD_MEDI',
            'COMMONAREA_MEDI',
            'ELEVATORS_MEDI',
            'ENTRANCES_MEDI',
            'FLOORSMAX_MEDI',
            'FLOORSMIN_MEDI',
            'LANDAREA_MEDI',
            'LIVINGAPARTMENTS_MEDI',
            'LIVINGAREA_MEDI',
            'NONLIVINGAPARTMENTS_MEDI',
            'NONLIVINGAREA_MEDI',
            'FONDKAPREMONT_MODE',
            'HOUSETYPE_MODE',
            'TOTALAREA_MODE',
            'WALLSMATERIAL_MODE',
            'EMERGENCYSTATE_MODE',
            'OBS_30_CNT_SOCIAL_CIRCLE',
            'DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE',
            'DEF_60_CNT_SOCIAL_CIRCLE',
            'DAYS_LAST_PHONE_CHANGE',
            'FLAG_DOCUMENT_2',
            'FLAG_DOCUMENT_3',
            'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5',
            'FLAG_DOCUMENT_6',
            'FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_8',
            'FLAG_DOCUMENT_9',
            'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11',
            'FLAG_DOCUMENT_12',
            'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14',
            'FLAG_DOCUMENT_15',
            'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_18',
            'FLAG_DOCUMENT_19',
            'FLAG_DOCUMENT_20',
            'FLAG_DOCUMENT_21',
            'AMT_REQ_CREDIT_BUREAU_HOUR',
            'AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT',
            'AMT_REQ_CREDIT_BUREAU_YEAR',
            'DAYS_EMPLOYED_PERC',
            'INCOME_CREDIT_PERC',
            'INCOME_PER_PERSON',
            'ANNUITY_INCOME_PERC',
            'PAYMENT_RATE',
            'PREVIOUS_LOANS_COUNT',
            'DAYS_CREDIT',
            'CREDIT_DAY_OVERDUE',
            'DAYS_CREDIT_ENDDATE',
            'DAYS_ENDDATE_FACT',
            'AMT_CREDIT_MAX_OVERDUE',
            'CNT_CREDIT_PROLONG',
            'AMT_CREDIT_SUM',
            'AMT_CREDIT_SUM_DEBT',
            'AMT_CREDIT_SUM_LIMIT',
            'AMT_CREDIT_SUM_OVERDUE',
            'DAYS_CREDIT_UPDATE',
            'AMT_ANNUITY_y',
            'MONTHS_BALANCE_MEAN',
            'PREVIOUS_APPLICATION_COUNT',
            'SK_ID_PREV',
            'AMT_ANNUITY',
            'AMT_APPLICATION',
            'AMT_CREDIT_y',
            'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE_y',
            'HOUR_APPR_PROCESS_START_y',
            'NFLAG_LAST_APPL_IN_DAY',
            'RATE_DOWN_PAYMENT',
            'RATE_INTEREST_PRIMARY',
            'RATE_INTEREST_PRIVILEGED',
            'DAYS_DECISION',
            'SELLERPLACE_AREA',
            'CNT_PAYMENT',
            'DAYS_FIRST_DRAWING',
            'DAYS_FIRST_DUE',
            'DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_LAST_DUE',
            'DAYS_TERMINATION',
            'NFLAG_INSURED_ON_APPROVAL',
            'MONTHS_BALANCE_x',
            'AMT_BALANCE',
            'AMT_CREDIT_LIMIT_ACTUAL',
            'AMT_DRAWINGS_ATM_CURRENT',
            'AMT_DRAWINGS_CURRENT',
            'AMT_DRAWINGS_OTHER_CURRENT',
            'AMT_DRAWINGS_POS_CURRENT',
            'AMT_INST_MIN_REGULARITY',
            'AMT_PAYMENT_CURRENT',
            'AMT_PAYMENT_TOTAL_CURRENT',
            'AMT_RECEIVABLE_PRINCIPAL',
            'AMT_RECIVABLE',
            'AMT_TOTAL_RECEIVABLE',
            'CNT_DRAWINGS_ATM_CURRENT',
            'CNT_DRAWINGS_CURRENT',
            'CNT_DRAWINGS_OTHER_CURRENT',
            'CNT_DRAWINGS_POS_CURRENT',
            'CNT_INSTALMENT_MATURE_CUM',
            'SK_DPD_x',
            'SK_DPD_DEF_x',
            'NUM_INSTALMENT_VERSION',
            'NUM_INSTALMENT_NUMBER',
            'DAYS_INSTALMENT',
            'DAYS_ENTRY_PAYMENT',
            'AMT_INSTALMENT',
            'AMT_PAYMENT',
            'MONTHS_BALANCE_y',
            'CNT_INSTALMENT',
            'CNT_INSTALMENT_FUTURE',
            'SK_DPD_y',
            'SK_DPD_DEF_y'
        ]

        option = st.selectbox(
        'Select a variable ?',
        options = colonnes)
        
        st.write('You selected:', option)
        
        if option == 'None':
            st.write('Waiting for you')
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[str(option)], bins=20)
            ax.axvline(df.loc[df['SK_ID_CURR']==int(input_id),str(option)].values[0], color="green", linestyle='--')
            ax.set(title=str(option), xlabel=str(option), ylabel='')
            st.pyplot(fig)

        st.subheader("----"*20)

        option2 = st.selectbox(
        'Select a first variable ?',
        options = colonnes)

        option3 = st.selectbox(
        'Select a second variable ?',
        options = colonnes)

        if option2 == 'None' or option3 == 'None':
            st.write('Waiting for you')
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df,x=str(option3), y=str(option2))
            ax.axvline(df.loc[df['SK_ID_CURR']==int(input_id),str(option3)].values[0], color="green", linestyle='--')
            ax.axhline(df.loc[df['SK_ID_CURR']==int(input_id),str(option2)].values[0], color="green", linestyle='--')
            ax.set(title='Representation of ' + str(option2) + ' by ' + str(option3), xlabel=str(option3), ylabel=str(option2))
            st.pyplot(fig)

else:
    st.header('Wrong ID, Try again !')