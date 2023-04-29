import streamlit as st 
from PIL import Image
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


         
         

#sideBar---------------------------------------------------------

with st.sidebar:
    #File uploader
    uploded_file = st.file_uploader("Upload a file ",type="csv")
    #SelectBox for Dataset
    add_selectbox = st.selectbox(
    "Select a dataSet :",
    ("Breast Cancer","Colic","Diabetes", "Drug200","hepatitis")
    )
    #Radio for clustering method
    add_radio = st.radio(
    "Choose a Clustering method",
    ("K-Means", "K-Medoids","Agnes","Diana"),
    index = 0,)
    #Submit buttom
    button_color = 'color:blue'  # red
    button_style = f'background-color: {button_color};'
    submit = st.button("Starts")



#display data---------------------------------------------------
def display_dataset(uploded_file):
   if uploded_file is not None :
     df = pd.read_csv(uploded_file)
     return df
   else :
     return "No dataset selected"
   

#Page------------------------------------------------------------
img = Image.open("image.jpg")


import streamlit as st

# Define some CSS to set the background image
import streamlit as st

# Define some CSS to set the background image
page_bg_img = '''
<style>
body {
background-image: url("https://cdn.pixabay.com/photo/2021/09/03/16/59/fisherman-6600665_1280.jpg");
background-size: cover;
}
</style>
'''

# Add the custom CSS to the page
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add some content to the page
st.write('Hello, world!')


# Add the custom CSS to the page
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add some content to the page
st.write('Hello, world!')

st.title("It's summer!")
st.sidebar.header("Configuration")

df = display_dataset(uploded_file)
st.write(df)
if type(df)is not str:
    st.write("Pre-Processing phase :")
    st.write(df.dtypes)
    st.write(preprocessing(df))
    






