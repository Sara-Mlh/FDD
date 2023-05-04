#pip install kneed
import streamlit as st 
from PIL import Image
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator


#Pre-processing ----------------------------------------------------
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

#K_Means---------------------------------------------------------
def k_Means(data):
   kmeans_kwargs = {
      "init":"random",
      "n_init": 10 ,
      "max_iter" : 300,
      "random_state": 42,
   }
   sse = {}
   for k in range(1,11):
      k_means = KMeans(n_clusters=k , **kmeans_kwargs) #second prtmr  only random state =1
      k_means.fit(data)
      sse[k] = k_means.inertia_
   return sse 

#Elbow_Method : SSE Curve----------------------------------------------------
def plot_elbow(data):
   sse = k_Means(data)
   plt.style.use("fivethirtyeight")
   plt.plot(list(sse.keys()),list(sse.values()), 'bx-',linewidth=1.5,color='red')
   #plt.xticks(range(1, 11))
   plt.title("Elbow Method ")
   plt.xlabel("Number of Clusters")
   plt.ylabel("SSE")
   #sns.pointplot(x=list(sse.keys()), y=list(sse.values()),color = "blue")
   #plt.gca().collections[0].set_sizes([50])
   #plt.plot([], [], linewidth=2)
   plt.show()
   return plt
#The Optimal K number of clusters -------------------------------------------------
def optimal_K(data): 
   sse = k_Means(data)
   k=  KneeLocator(list(sse.keys()), list(sse.values()), curve="convex", direction="decreasing")
   return k.elbow

#Application of the K means algorithm-----------------------------------------------
def perform_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids
#K means plot---------------------
def plot_kmeans(df,labels,centroids):
   plt.scatter(df[:, 0], df[:, 1], c=labels)
   plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', linewidths=3, color='r')
   plt.title('Clusters (k = {})'.format(optimal_K(df)))
   plt.show()
   return plt

         
         

#sideBar---------------------------------------------------------

with st.sidebar:
    #File uploader
    uploded_file = st.file_uploader("Upload a file ",type="csv")
    #SelectBodf for Dataset
    add_selectbodf = st.selectbox(
    "Select a dataSet :",
    ("","Breast Cancer","Colic","Diabetes", "Drug200","hepatitis")
    )
    #Radio for clustering method
    add_radio = st.radio(
    "Choose a Clustering method",
    ("","K-Means", "K-Medoids","Agnes","Diana"),
    index = 0,)
    #Submit buttom
    button_color = 'color:blue'  # red
    button_style = f'background-color: {button_color};'


    submit = st.button("Start Test")

#display data---------------------------------------------------
def display_dataset(uploded_file):
   if uploded_file is not None :
     df = pd.read_csv(uploded_file)
     return df
   else :
     st.markdown("<h5 style='tedft-align: left ; margin-top:5em ; color: red;'>No dataset selected</h1>", unsafe_allow_html=True)
     return None

#Page------------------------------------------------------------

# Define some CSS to set the background image
page_bg_img = '''
<style>
body {
background-image: url("https://cdn.pidfabay.com/photo/2021/09/03/16/59/fisherman-6600665_1280.jpg");
background-size: cover;
}
</style>
'''
# Add the custom CSS to the page
st.markdown(page_bg_img, unsafe_allow_html=True)
# Add some content to the page
st.title('Clustering Test Application \n')
st.header('Welcome , ')

dataset = display_dataset(uploded_file)
if dataset is not None:
  st.write(dataset)
  st.write(dataset.dtypes)
  st.write("Pre-Processing phase :")
  df = preprocessing(dataset)
  st.write(df)
  st.write("K-Means :")
# Call the function to plot the SSE curve
  fig = plot_elbow(df)
# Display the plot in the Streamlit app
  st.pyplot(fig)
# Display the K value 
  k = optimal_K(df)
  st.write("The optimal K is : ",k)
  labels,centroids = perform_kmeans(df,k)
  #st.pyplot(plot_kmeans(df,labels,centroids))


  
