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
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist


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

#Elbow method implementation ---------------------------------------------------------
def elbow(data,method): #data after pre-processing
   kmeans_kwargs = {
      "init":"k-means++",
      "n_init": 10 ,
      "max_iter" : 300,
      "random_state": 42,
   }

   sse = {}
   model = None
   for k in range(1,11):
      if method == "K-Means" : 
         model = KMeans(n_clusters=k,**kmeans_kwargs) #second prtmr  only random state =1
         model.fit(data)
      elif method == "Agnes":
         dist_matrix = pdist(data, metric='euclidean')
         model= AgglomerativeClustering(n_clusters=k)
         try:
           model.fit(dist_matrix.reshape(-1, 1))
         except MemoryError:
                print("MemoryError occurred while fitting AgglomerativeClustering model for k =", k)
                continue
      if model is not None:
        sse[k] = model.inertia_
   return sse 

#Elbow_Method plot : SSE Curve-----------------------------------------
def plot_elbow(data,method):
   sse = elbow(data,method)
   plt.style.use("fivethirtyeight")
   fig = plt.figure(figsize=(10, 5))
   plt.plot(list(sse.keys()),list(sse.values()), 'bx-',linewidth=1.5,color='green')
   plt.title("Elbow Method ")
   plt.xlabel("Number of Clusters")
   plt.ylabel("SSE")
   st.pyplot(fig)
#The Optimal K number of clusters --------------------------------
def optimal_K(data,method): 
   sse = elbow(data,method)
   if not sse:
       return None
   k = KneeLocator(list(sse.keys()), list(sse.values()), curve="convex", direction="decreasing")
   return k.knee
   #k=  KneeLocator(list(sse.keys()), list(sse.values()), curve="convex", direction="decreasing")
   #return k.elbow

#Application of the K means algorithm-----------------------------
def perform_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids
#K means scatter plot-----------------------------------------------------
def plot_kmeans(df,method):
   k = optimal_K(df,method)
   if method == "K-Means" :
     labels, centroids = perform_kmeans(df,k)
   fig, ax = plt.subplots(figsize=(10, 5))
   sns.scatterplot(x=df[:, 0], y=df[:, 1], c=labels, ax=ax)
   sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='x', label="centroid", linewidths=3, color='r', ax=ax)
   plt.title('Clusters (k = {})'.format(k))
   st.pyplot(fig)
# Intraclasse calcul --------------------------------------------
def calculate_intracluster_distance(data,method, metric='euclidean'):
    k = optimal_K(data,method)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    intra_cluster_distances = np.zeros(k)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i in range(k):
        points_in_cluster = data[labels == i]
        centroid = centroids[i]
        intra_cluster_distances[i] = np.mean(pairwise_distances(points_in_cluster, centroid.reshape(1, -1)))
    return intra_cluster_distances.sum()

# Interclasse calcul --------------------------------------------
def calculate_intercluster_distance(data,method,metric='euclidean'):
    kmeans = KMeans(n_clusters=optimal_K(data,method))
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    return pairwise_distances(centroids).sum()

# Agnes test ----------------------------------------------------
def Agnes_dendogram(data):
   # Compute the linkage matrix
   fig = plt.figure(figsize=(30,20))
   sch.dendrogram(sch.linkage(data, method='ward'))
   plt.title('Dendrogram')
   plt.xlabel('Cluster Size')
   plt.ylabel('Distance')
   st.write("Agnes Dendogram :")
   st.pyplot(fig)


         
         

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
    ("","K-Means", "K-Medoids","Agnes","Diana","DBScan"),
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
st.set_option('deprecation.showPyplotGlobalUse', False)
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
  #st.write(radio)
  if radio == "K-Means": #K-Means
    #print("inside k means ")
    st.write(radio,":")
    plot_elbow(df,radio) # Call the function to plot the SSE curve
    st.write("The optimal K is : ",optimal_K(df,radio)) # Display the K value 
    plot_kmeans(df,radio)    
    #Display intraclass and interclass values 
    st.write(" Interclasse :", calculate_intercluster_distance(df,radio))
    st.write(" Intraclasse :", calculate_intracluster_distance(df,radio))
  elif radio == "Agnes" :
       Agnes_dendogram(df)

    #elif radio == "K-Medoids":
    #elif radio == "Agnes":
    #elif radio == "Diana":
    


  
