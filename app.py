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
from sklearn.metrics import pairwise_distances



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
def k_Means(data): #data after pre-processing
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

#Elbow_Method : SSE Curve-----------------------------------------
def plot_elbow(data):
   sse = k_Means(data)
   plt.style.use("fivethirtyeight")
   fig = plt.figure()
   plt.plot(list(sse.keys()),list(sse.values()), 'bx-',linewidth=1.5,color='red')
   #plt.xticks(range(1, 11))
   plt.title("Elbow Method ")
   plt.xlabel("Number of Clusters")
   plt.ylabel("SSE")
   #sns.pointplot(x=list(sse.keys()), y=list(sse.values()),color = "blue")
   #plt.gca().collections[0].set_sizes([50])
   #plt.plot([], [], linewidth=2)
   #plt.show()
   st.pyplot(fig)
#The Optimal K number of clusters --------------------------------
def optimal_K(data): 
   sse = k_Means(data)
   k=  KneeLocator(list(sse.keys()), list(sse.values()), curve="convex", direction="decreasing")
   return k.elbow

#Application of the K means algorithm-----------------------------
def perform_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids
#K means plot-----------------------------------------------------
def plot_kmeans(df):
   k = optimal_K(df)
   labels, centroids = perform_kmeans(df,k)
   fig, ax = plt.subplots()
   sns.scatterplot(x=df[:, 0], y=df[:, 1], c=labels, ax=ax)
   sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='x', label="centroid", linewidths=3, color='r', ax=ax)
   plt.title('Clusters (k = {})'.format(k))
   st.pyplot(fig)
# Intraclasse calcul --------------------------------------------
def calculate_intracluster_distance(data, metric='euclidean'):
    kmeans = KMeans(n_clusters=optimal_K(data))
    kmeans.fit(data)
    labels = kmeans.labels_
    sum_dist = 0
    for i in range(len(set(labels))):
        sum_dist += pairwise_distances(data[labels == i], metric=metric).sum()
    return sum_dist

# Interclasse calcul --------------------------------------------
def calculate_intercluster_distance(data, metric='euclidean'):
    kmeans = KMeans(n_clusters=optimal_K(data))
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    return pairwise_distances(centroids, centroids, metric=metric).sum()


         
         

#sideBar---------------------------------------------------------

with st.sidebar:
    #File uploader
    uploded_file = st.file_uploader("Upload a file ",type="csv")
    #SelectBox for Dataset
    selectbox = st.selectbox(
    "Select a dataSet :",
    ("","Breast Cancer","Colic","Diabetes", "Drug200","hepatitis")
    )
    #Radio for clustering method
    radio = st.radio(
    "Choose a Clustering method",
    ("","K-Means", "K-Medoids","Agnes","Diana"),
    index = 0,)
    #Submit buttom
    button_color = 'color:blue'  # red
    button_style = f'background-color: {button_color};'
    submit = st.button("Start Test")
#if radio == "K-Means" :
   #data_header(df)

    

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
  #fig = plot_elbow(df)
# Display the plot in the Streamlit app
  #st.pyplot(fig)
# Display the K value 
  plot_elbow(df)
  st.write("The optimal K is : ",optimal_K(df))
  plot_kmeans(df)
  st.write(" Interclasse :", calculate_intercluster_distance(df))
  st.write(" Interclasse :", calculate_intracluster_distance(df))
  


  
