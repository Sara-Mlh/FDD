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
from sklearn.metrics import pairwise_distances , silhouette_score
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

#Pre-processing ----------------------------------------------------
def preprocessing(data):
    #Nan values
    if(data.isna().sum().sum() !=0):
       # Replace missing values with the mean
        for col in data.columns:
            if data[col].dtype != "object" and data[col].dtype != "category":
                col_mean = data[col].mean()
                data[col].fillna(col_mean, inplace=True)
            else :
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
         model = KMeans(n_clusters=k,**kmeans_kwargs) 
         model.fit(data)
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
   fig, ax = plt.subplots(figsize=(8, 5))
   pca = PCA(n_components=2)
   dfpca = pca.fit_transform(df)
   sns.scatterplot(x=dfpca[:, 0], y=dfpca[:, 1], c=labels, ax=ax)
   sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], label="centroid", linewidths=3, color='r', ax=ax)
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
        intra_cluster_distances[i] = np.mean(np.linalg.norm(points_in_cluster - centroid, axis=1))
    return intra_cluster_distances.mean()

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
    num_clusters = optimal_K(data)
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
    fig = plt.figure(figsize=(30,20))
    dendrogram(linkage_matrix)
    plt.xlabel('Data points')
    plt.ylabel('Dissimilarity')
    plt.title('DIANA Dendrogram')
    st.pyplot(fig)
#inter intra diana ----------------------------------------
def calculate_intraclass_distanceadiana(data, distance_metric='euclidean'):
    clusters = agglomerative_clustering_with_centroids(data)
    num_clusters = len(clusters)
    intraclass_distances = np.zeros(num_clusters)
    
    for i in range(num_clusters):
        cluster_points = clusters[i]['points']
        intraclass_distances[i] = np.mean(pairwise_distances(cluster_points, metric=distance_metric)) 
        intraclass_distance = np.mean(intraclass_distances)
    return intraclass_distance

#interclasse agnes -------------------------------------
def calculate_interclass_distancediana(data, distance_metric='euclidean'):
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

def kmedoids_clustering(data):
    # Create k-medoids model
    k = optimal_K(data)
    kmedoids = KMedoids(n_clusters=k, random_state=0)

    # Fit model to data
    kmedoids.fit(data)

    # Get cluster labels and centers
    cluster_labels = kmedoids.labels_
    cluster_centers = kmedoids.cluster_centers_

    # Apply PCA for visualization (assuming data is a DataFrame)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Plot clusters
    fig= plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], c=cluster_labels)
    sns.scatterplot(x=cluster_centers[:, 0], y=cluster_centers[:, 1], label="centroid", linewidths=7, color='r')
    plt.title(f'K-medoids Clustering with k={k}')
    plt.legend()
    plt.show()
    st.pyplot(fig)
#Kmedoids -------------------------------------------------------------------
def intra_class_distance_kmedoids(data):
    k = optimal_K(data)
    kmedoids = KMedoids(n_clusters=k, metric='euclidean').fit(data)
    labels = kmedoids.labels_
    medoids = kmedoids.cluster_centers_
    intra_cluster_distances = np.zeros(k)
    for i in range(k):
        cluster_points = data[labels == i]
        medoid = medoids[i]
        intra_cluster_distances[i] = np.mean(pairwise_distances(cluster_points, [medoid], metric='euclidean'))
    return intra_cluster_distances.mean()

def inter_class_distance_kmedoids(data):
    kmedoids = KMedoids(n_clusters=optimal_K(data), metric='euclidean').fit(data)
    centers = kmedoids.cluster_centers_
    return pairwise_distances(centers, metric='euclidean').sum()
# DBSCAN ---------------------------------------------------------------------
def DBScan(data):

    # Define values of minPts and epsilon to test
    minPts_list = [5, 10, 15]
    epsilon_list = [0.5, 1, 1.5]

    # Initialize list to store the number of clusters obtained for each combination of minPts and epsilon
    n_clusters_list = []

    # Iterate over different values of minPts and epsilon
    for minPts in minPts_list:
        for epsilon in epsilon_list:

            # Create an instance of DBSCAN with the current minPts and epsilon values
            dbscan = DBSCAN(eps=epsilon, min_samples=minPts)

            # Apply DBSCAN to the standardized data
            y_pred = dbscan.fit_predict(data)

            # Count the number of clusters
            labels = np.unique(y_pred)
            n_clusters = len(labels) - (1 if -1 in labels else 0)

            # Append the number of clusters to the list
            n_clusters_list.append(n_clusters)

            # Print the number of clusters and noise points for the current minPts and epsilon values
            n_noise = list(y_pred).count(-1)
            print(f"Pour MinPts={minPts} et Epsilon={epsilon}, le nombre de clusters est de {n_clusters} et le nombre de points de bruit est de {n_noise}.")

    # Reshape the list of cluster counts into a 2D array
    n_clusters_array = np.array(n_clusters_list).reshape(len(minPts_list), len(epsilon_list))

    # Plot the number of clusters as a function of minPts and epsilon
    fig = plt.figure(figsize=(30,20))
    plt.imshow(n_clusters_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(epsilon_list)), epsilon_list)
    plt.yticks(np.arange(len(minPts_list)), minPts_list)
    plt.xlabel('Epsilon')
    plt.ylabel('MinPts')
    plt.title('Nombre de clusters en fonction de MinPts et Epsilon')
    plt.show()
    st.pyplot(fig)

# comparaison plot 
def compare_intra_interclass(algorithms, intra_scores, inter_scores):
    """
    Compare intra and interclass differences between different algorithm methods and generate a bar plot.

    Args:
        algorithms (list): List of algorithm names.
        intra_scores (list): List of intra-class scores for each algorithm.
        inter_scores (list): List of inter-class scores for each algorithm.
    """

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(intra_scores))
    r2 = [x + bar_width for x in r1]

    # Create the bar plot
    fig, ax = plt.subplots()
    ax.bar(r1, intra_scores, color='b', width=bar_width, edgecolor='white', label='Intra-class')
    ax.bar(r2, inter_scores, color='g', width=bar_width, edgecolor='white', label='Inter-class')

    # Add labels and title
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.set_title('Intra and Inter-class Differences')

    # Add x-axis tick labels
    ax.set_xticks([r + bar_width/2 for r in range(len(intra_scores))])
    ax.set_xticklabels(algorithms)

    # Add legend
    ax.legend()

    # Display the plot
    st.pyplot(fig)
#sideBar---------------------------------------------------------

with st.sidebar:
    #File uploader
    uploded_file = st.file_uploader("Upload a file ",type="csv")
    #SelectBox for Dataset
    selectbox = st.selectbox(
    "OR Select a dataSet here :",
    ("","Breast Cancer","Colic","Diabetes","hepatitis", "DNA Sequences")
    )
    #Radio for clustering method
    radio = st.radio(
    "Choose a Clustering method",
    ("None","K-Means", "K-Medoids","Agnes","Diana","DBScan"),
    index = 0,)


#display data---------------------------------------------------
def display_dataset(uploded_file):
   if uploded_file is not None :
     df = pd.read_csv(uploded_file)
     return df
   else :
     st.markdown("<h5 style='tedft-align: left ; margin-top:5em ; color: red;'>Selection of a dataset</h1>", unsafe_allow_html=True)
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
#upload a file
dataset = display_dataset(uploded_file)
if dataset is None:
    if selectbox !="" :
     if selectbox == "Diabetes":
        dataset = pd.read_csv("C:/Users/ASUS/Desktop/datasets/diabetes.csv")
     elif selectbox=="Breast Cancer": 
        dataset = pd.read_csv("C:/Users/ASUS/Desktop/datasets/cancer.csv") 
     elif selectbox=="Colic":  
        dataset = pd.read_csv("C:/Users/ASUS/Desktop/datasets/colic.csv")
     elif selectbox=="hepatitis":
        dataset = pd.read_csv("C:/Users/ASUS/Desktop/datasets/HepatitisCdata.csv")
     elif selectbox=="DNA Sequences":
        dataset = pd.read_csv("C:/Users/ASUS/Desktop/datasets/dna.txt")
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
  elif radio == "Agnes" :  #AGNES
       Agnes_dendogram(df)
       st.write("interclasse :",calculate_interclass_distanceagnes(df))
       st.write("intraclasse :",calculate_intraclass_distanceagnes(df))
  elif radio == "Diana" :  #DIANA
       st.write("Dendogram Diana")
       diana_dendrogram(df)
       #st.write("interclasse :",calculate_interclass_distancediana(df))
       #st.write("intraclasse :",calculate_intraclass_distanceadiana(df))
  elif radio == "K-Medoids":
      kmedoids_clustering(df)
      st.write("interclasse :",inter_class_distance_kmedoids(df))
      st.write("intraclasse :",intra_class_distance_kmedoids(df))
  elif radio == "DBScan":
      st.write("DBSCAN  :")
      DBScan(df)
  if st.sidebar.button("Comparer les methodes"):
      st.write("comparaison")
         # Calculate intra-class and inter-class scores for each algorithm
      algorithms = ['K-Means', 'K-Medoids','Agnes','DIANA']
      intra_scores = [2.44, 2.50, 3.3514,4.59]
      inter_scores = [44.41, 34.80, 12.39,11.39489]

      # Compare and plot the intra and inter-class differences
      compare_intra_interclass(algorithms, intra_scores, inter_scores)

  
