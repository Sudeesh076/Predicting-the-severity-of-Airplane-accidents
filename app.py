import streamlit as st
from num2words import num2words
import numpy as np
import pickle
import pandas as pd
import numpy as np
import os
import glob
import joblib
from scipy.sparse import hstack
import xgboost
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
import shap


st.title("The Severity of Airplane Accidents")
st.header('Prediction')
st.subheader("By Sudeesh Reddy")
st.image("1.jpg")


# loading models

arr = os.listdir("models_pre_processing/")

path = glob.glob("models_pre_processing/*")
lst = []
for i in range(len(arr)):
    arr1 = (arr[i].replace(".joblib", ""))
    lst.append(arr1)

for i in range(len(lst)):
    a = lst[i]
    b = joblib.load(path[i])
    globals()[a] = b

XGBT = pickle.load(open("xgbt_model/pima.pickle.dat", "rb"))










st.subheader('Enter the Airline details:')



a = 1

with st.form("my_form"):


    Safety_Score = st.number_input('Safety Score [0 - 100]:', min_value=0, max_value=100, value=50)

    Days_Since_Inspection = st.number_input('Days Since Inspection:', min_value=1, max_value=30, step=1, value=14)

    Total_Safety_Complaints = st.number_input('Total Safety Complaints:', min_value=0, max_value=54, step=1, value=22)

    Control_Metric = st.number_input('Control Metric:[0-100]', min_value=0, max_value=100, step=1, value=71)

    Turbulence_In_gforces = st.number_input('Turbulence In gforces [0-1]:',value=0.207)

    Cabin_Temperature = st.number_input('Cabin Temperature:', min_value=70, max_value=100, value=74)

    Accident_Type_Code = st.radio('Accident_Type_Code:', (1,2,3,4,5,6,7), index=2)

    Violations = st.radio('Violations:', (0,1,2,3,4,5), index=3)

    Max_Elevation = st.number_input('Max_Elevation: ', min_value=800, max_value=70000, step=1, value=31335)

    Adverse_Weather_Metric = st.number_input('Adverse Weather Metric:',value=0.424352)

    submitted = st.form_submit_button("Predict the severity")

    input = [Safety_Score,Days_Since_Inspection,Total_Safety_Complaints,Control_Metric,Turbulence_In_gforces,Cabin_Temperature,Accident_Type_Code,Violations,Max_Elevation,Adverse_Weather_Metric]


    if submitted:
        a=0

if a==0:

    df_test = pd.DataFrame([input])

    df_test.columns = ["Safety_Score","Days_Since_Inspection","Total_Safety_Complaints","Control_Metric","Turbulence_In_gforces","Cabin_Temperature","Accident_Type_Code","Violations","Max_Elevation","Adverse_Weather_Metric"]



    # feature enigneerring methods
    df_test['FE1'] = df_test["Safety_Score"] * df_test["Days_Since_Inspection"]
    df_test['FE2'] = df_test["Safety_Score"] / df_test["Accident_Type_Code"]
    df_test['FE3'] = np.log(df_test["Adverse_Weather_Metric"])
    df_test['FE4'] = np.sqrt(df_test["Total_Safety_Complaints"])
    df_test['FE5'] = np.log(df_test["Max_Elevation"] / df_test["Adverse_Weather_Metric"])
    df_test['FE6'] = (df_test["Control_Metric"] * df_test["Turbulence_In_gforces"])
    df_test['FE7'] = np.log(df_test["Adverse_Weather_Metric"] / (df_test["Accident_Type_Code"]))
    df_test['FE8'] = (df_test["Safety_Score"] + (df_test["Control_Metric"]) / 2)
    df_test['FE9'] = df_test[["Safety_Score", "Control_Metric"]].min(axis=1)
    df_test['FE10'] = df_test[["Safety_Score", "Control_Metric"]].min(axis=1)

    X1 = df_test


    # converting numerical data to cat data
    a = []

    for i in X1.Accident_Type_Code:
        a.append(num2words(i, to='ordinal'))
    X1['Accident_Type_Code_cat'] = pd.DataFrame({'Accident_Type_Code': a})


    # data pre-processing

    train_Safety_Score_feature_StandardScaler = scaler1.transform(X1['Safety_Score'].values.reshape(-1, 1))
    train_Days_Since_Inspection_feature_StandardScaler = scaler2.transform(
        X1['Days_Since_Inspection'].values.reshape(-1, 1))
    train_Total_Safety_Complaints_feature_StandardScaler = scaler3.transform(
        X1['Total_Safety_Complaints'].values.reshape(-1, 1))
    train_Control_Metric_feature_StandardScaler = scaler4.transform(X1['Control_Metric'].values.reshape(-1, 1))
    train_Turbulence_In_gforces_feature_StandardScaler = scaler5.transform(
        X1['Turbulence_In_gforces'].values.reshape(-1, 1))
    train_Cabin_Temperature_feature_StandardScaler = scaler6.transform(X1['Cabin_Temperature'].values.reshape(-1, 1))
    train_Max_Elevation_feature_StandardScaler = scaler7.transform(X1['Max_Elevation'].values.reshape(-1, 1))
    train_Adverse_Weather_Metric_feature_StandardScaler = scaler8.transform(
        X1['Adverse_Weather_Metric'].values.reshape(-1, 1))
    train_Accident_Type_Code_feature_StandardScaler = scaler9.transform(X1['Accident_Type_Code'].values.reshape(-1, 1))
    train_Violations_feature_StandardScaler = scaler10.transform(X1['Violations'].values.reshape(-1, 1))
    train_FE1_feature_StandardScaler = scaler11.transform(X1['FE1'].values.reshape(-1, 1))
    train_FE2_feature_StandardScaler = scaler12.transform(X1['FE2'].values.reshape(-1, 1))
    train_FE3_feature_StandardScaler = scaler13.transform(X1['FE3'].values.reshape(-1, 1))
    train_FE4_feature_StandardScaler = scaler14.transform(X1['FE4'].values.reshape(-1, 1))
    train_FE5_feature_StandardScaler = scaler15.transform(X1['FE5'].values.reshape(-1, 1))
    train_FE6_feature_StandardScaler = scaler16.transform(X1['FE6'].values.reshape(-1, 1))
    train_FE7_feature_StandardScaler = scaler17.transform(X1['FE7'].values.reshape(-1, 1))
    train_FE8_feature_StandardScaler = scaler18.transform(X1['FE8'].values.reshape(-1, 1))
    train_FE9_feature_StandardScaler = scaler19.transform(X1['FE9'].values.reshape(-1, 1))
    train_FE10_feature_StandardScaler = scaler20.transform(X1['FE10'].values.reshape(-1, 1))
    train_Accident_Type_Code_feature_onehotCoding = vectorizer1.transform(X1['Accident_Type_Code_cat'])


    # data stacking

    X_train_pp1 = hstack((train_Safety_Score_feature_StandardScaler,
                          train_Days_Since_Inspection_feature_StandardScaler,
                          train_Total_Safety_Complaints_feature_StandardScaler,
                          train_Control_Metric_feature_StandardScaler,
                          train_Turbulence_In_gforces_feature_StandardScaler,
                          train_Cabin_Temperature_feature_StandardScaler,
                          train_Max_Elevation_feature_StandardScaler,
                          train_Adverse_Weather_Metric_feature_StandardScaler,
                          train_Accident_Type_Code_feature_StandardScaler,
                          train_Violations_feature_StandardScaler,
                          train_FE1_feature_StandardScaler,
                          train_FE2_feature_StandardScaler,
                          train_FE3_feature_StandardScaler,
                          train_FE4_feature_StandardScaler,
                          train_FE5_feature_StandardScaler,
                          train_FE6_feature_StandardScaler,
                          train_FE7_feature_StandardScaler,
                          train_FE8_feature_StandardScaler,
                          train_FE9_feature_StandardScaler,
                          train_FE10_feature_StandardScaler,
                          train_Accident_Type_Code_feature_onehotCoding,
                          )).tocsr()

    # model prediction

    y_predict_train = XGBT.predict(X_train_pp1)

    output = le.inverse_transform(y_predict_train)

    o1 = "The Severity of airplane: " + output[0]




    explainer = shap.TreeExplainer(XGBT)
    shap_values = explainer.shap_values(X_train_pp1)

    header = ['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints', 'Control_Metric',
              'Turbulence_In_gforces', 'Cabin_Temperature', 'Max_Elevation', 'Adverse_Weather_Metric',
              'Accident_Type_Code', 'Violations', 'FE1', 'FE2', 'FE3', 'FE4', 'FE5', 'FE6', 'FE7', 'FE8', 'FE9', 'FE10']
    for i in range(1, 8):
        header.append("Accident_Type_Code_" + str(i))

    list1, list2 = zip(*sorted(zip(shap_values[0][0], header)))
    top5 = list(list2[-5:])[::-1]



    st.error(o1)

    st.write("These are top Feature 5 that causes this accident")

    st.warning(top5)

    st.write("Reference of Feature")
    st.image("2.png")
















