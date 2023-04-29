import streamlit as st 
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def preprocessing(data):
    #Nan values
    if(data.isna().sum().sum() !=0):
         data.dropna(axis = 0 , inplace=True)
    #duplicated values 
    if(data.duplicated().sum()!=0):
         data.drop_duplicates(inplace=True)
    #categorical values to num ..normalisation
    encoder  = LabelEncoder()
    for col in data.columns : 
       if data[col].dtype == "object" or data[col].dtype == "category":
        data[col] = encoder.fit_transform(data[col])
    #standarisation 
    scaler= StandardScaler()
    data = scaler.fit_transform(data)

    return data
         
         

uploded_file = st.file_uploader("Upload a file ",type="csv")

#display data 
def display_dataset(uploded_file):
   if uploded_file is not None :
     df = pd.read_csv(uploded_file)
     return df
   else :
     return "No dataset selected"

df = display_dataset(uploded_file)
st.write(df)
if df is not None :
    st.write("Pre-Processing phase :")
    st.write(df.dtypes)
    st.write(preprocessing(df))
    



add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)
with st.sidebar:
      add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )


