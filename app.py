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
#pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
#from pyclustering.cluster.diana import diana
from sklearn.decomposition import PCA

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
    dataa =scaler.fit_transform(data)
    newdata = pd.DataFrame(dataa,columns=data.columns)


    return newdata

#Elbow method implementation ---------------------------------------------------------
def elbow(data): #data after pre-processing
   kmeans_kwargs = {
      "init":"k-means++",
      "n_init": 10 ,
      "max_iter" : 300,
      "random_state": 42,
   }

   sse = {}
   model = None
   for k in range(1,11):
         model = KMeans(n_clusters=k,**kmeans_kwargs) #second prtmr  only random state =1
         model.fit(data)
      #elif method == "Agnes":
         #dist_matrix = pdist(data, metric='euclidean')
         #model= AgglomerativeClustering(n_clusters=k)
         #try:
           #model.fit(dist_matrix.reshape(-1, 1))
         #except MemoryError:
             #   print("MemoryError occurred while fitting AgglomerativeClustering model for k =", k)
              #  continuess
         if model is not None:
           sse[k] = model.inertia_
   return sse 

#Elbow_Method plot : SSE Curve-----------------------------------------
def plot_elbow(data):
   sse = elbow(data)
   plt.style.use("fivethirtyeight")
   fig = plt.figure(figsize=(10, 5))
   plt.plot(list(sse.keys()),list(sse.values()), 'bx-',linewidth=1.5,color='green')
   plt.title("Elbow Method ")
   plt.xlabel("Number of Clusters")
   plt.ylabel("SSE")
   st.pyplot(fig)
#The Optimal K number of clusters --------------------------------
def optimal_K(data): 
   sse = elbow(data)
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
def plot_kmeans(df):
   k = optimal_K(df)
   
   labels, centroids = perform_kmeans(df,k)
   fig, ax = plt.subplots(figsize=(10, 5))
   pca = PCA(n_components=2)
   dfpca = pca.fit_transform(df)
   sns.scatterplot(x=dfpca[:, 0], y=dfpca[:, 1], c=labels, ax=ax)
   sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='x', label="centroid", linewidths=3, color='r', ax=ax)
   plt.title('Clusters (k = {})'.format(k))
   st.pyplot(fig)
# Intraclasse calcul --------------------------------------------
def calculate_intracluster_distance_kmeans(data, metric='euclidean'):
    k = optimal_K(data)
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
def calculate_intercluster_distance_kmeans(data,metric='euclidean'):
    kmeans = KMeans(n_clusters=optimal_K(data))
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
#Agnes clustes --------------------------------------------
def agglomerative_clustering_with_centroids(data):
    num_clusters = optimal_K(data,method="K-Means")
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = clustering.fit_predict(data)

    clusters = []
    for cluster_id in range(num_clusters):
        cluster_points = data[cluster_labels == cluster_id]
        cluster_centroid = np.mean(cluster_points, axis=0)
        clusters.append({
            'points': cluster_points,
            'centroid': cluster_centroid
        })

    return clusters
# intraclasse agnes -------------------------------------
def calculate_intraclass_distanceagnes(data, distance_metric='euclidean'):
    clusters = agglomerative_clustering_with_centroids(data)
    num_clusters = len(clusters)
    intraclass_distances = np.zeros(num_clusters)
    
    for i in range(num_clusters):
        cluster_points = clusters[i]['points']
        intraclass_distances[i] = np.mean(pairwise_distances(cluster_points, metric=distance_metric)) 
    intraclass_distance = np.mean(intraclass_distances)
    return intraclass_distance

#interclasse agnes -------------------------------------
def calculate_interclass_distanceagnes(data, distance_metric='euclidean'):
    clusters = agglomerative_clustering_with_centroids(data)
    num_clusters = len(clusters)
    interclass_distances = np.zeros((num_clusters, num_clusters))
    
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            cluster_points_i = clusters[i]['points']
            cluster_points_j = clusters[j]['points']
            distances = pairwise_distances(cluster_points_i, cluster_points_j, metric=distance_metric)
            interclass_distances[i, j] = np.max(distances)
            interclass_distances[j, i] = np.max(distances)
    
    interclass_distance = np.max(interclass_distances)
    return interclass_distance

# dendogram diana----------------------------------------
def diana_dendrogram(data):
    # Calculate the dissimilarity matrix
    dissimilarity_matrix = squareform(pdist(data))

    # Perform hierarchical clustering
    linkage_matrix = linkage(dissimilarity_matrix, method='ward')

    # Plot the dendrogram
    

    # Show the plot
    fig = plt.figure(figsize=(30,20))
    dendrogram(linkage_matrix)
    plt.xlabel('Data points')
    plt.ylabel('Dissimilarity')
    plt.title('DIANA Dendrogram')
    st.pyplot(fig)
def kmedoid_clustering(data):
    # Create k-medoids model
    k = optimal_K(data)
    kmedoids = KMedoids(n_clusters=k, random_state=0)

    # Fit model to preprocessed data
    kmedoids.fit(data)

    # Get cluster labels and centers
    cluster_labels = kmedoids.labels_
    cluster_centers = kmedoids.cluster_centers_

    # Plot clusters
    fig, ax = plt.subplots(figsize=(8, 8))
    #colors = plt.cm.tab20(np.linspace(0, 1, k))
    for i in range(k):
        cluster_data = data[cluster_labels == i]
        ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {i+1}')
        #ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c =cluster_labels,cmap ='viridis' ) fouad ves
        ax.scatter(cluster_centers[i][0], cluster_centers[i][1], marker='x', s=200)
    ax.set_title(f'K-medoids Clustering with k={k}')
    ax.legend()
    st.pyplot(fig)

def intra_class_distance_kmedoids(data):
    kmedoids = KMedoids(n_clusters=optimal_K(data), metric='euclidean').fit(data)
    labels = kmedoids.labels_
    centers = kmedoids.cluster_centers_
    distances = []
    for i in range(len(centers)):
        cluster_data = data[labels == i]
        center = centers[i]
        dist = np.sum(np.square(cluster_data - center))
        distances.append(dist)
    return np.sum(distances)

def inter_class_distance_kmedoids(data):
    kmedoids = KMedoids(n_clusters=optimal_K(data), metric='euclidean').fit(data)
    centers = kmedoids.cluster_centers_
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.sum(np.square(centers[i] - centers[j]))
            distances.append(dist)
    return np.sum(distances)
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
    plot_elbow(df) # Call the function to plot the SSE curve
    st.write("The optimal K is : ",optimal_K(df)) # Display the K value 
    plot_kmeans(df)    
    #Display intraclass and interclass values 
    st.write(" Interclasse :", calculate_intercluster_distance_kmeans(df))
    st.write(" Intraclasse :", calculate_intracluster_distance_kmeans(df))
  elif radio == "Agnes" :
       Agnes_dendogram(df)
       st.write("interclasse :",calculate_interclass_distanceagnes(df))
       st.write("intraclasse :",calculate_intraclass_distanceagnes(df))
  elif radio == "Diana" :
       st.write("Dendogram Diana")
       diana_dendrogram(df)
  elif radio == "K-Medoids":
      kmedoid_clustering(df)
      st.write("interclasse :",inter_class_distance_kmedoids(df))
      st.write("intraclasse :",intra_class_distance_kmedoids(df))


    #elif radio == "K-Medoids":
    #elif radio == "Agnes":
    #elif radio == "Diana":
    


  
