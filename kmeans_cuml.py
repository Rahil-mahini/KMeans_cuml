# -*- coding: utf-8 -*-

import os
from cuml.cluster import KMeans
import pandas as pd
from dask.distributed import Client 
from dask_cuda import  LocalCUDACluster
from cuml.metrics.cluster import silhouette_score 
from scipy.spatial.distance import euclidean
import cudf
import dask_cudf

#This code computes the feature selection for a very large dataset of features of about 30 million features.
#It calculate the KMeans using cuml.cluster on both features (X) and endpoints (Y) to find the distance of each feature that are in the same cluster as endpoint to select the closest features to endpoints.
#First, load the features csv file (X) and endpoints csv file (Y) into cudf dataframe.
#Next, combine both datasets (X and Y). It assume endpoint (Y) as one feature.
#Next, perform KMeans, fit and predict the cluster labels for each feature (X) and endpoint (Y).
#Next, find the cluster label of the endpoint (Y). 
#Next, find the Euclidean distance of the features(X) that are in the same cluster as endpoint (Y), from the endpoint (Y). 
#Then, sort the distances and find the k closet distance to the endpoint (Y).
#Finally, extract the closest distance features (X) to the endpoint(Y) as selected features.	


# Load data from X_file_path csv file and return cudf dataframe  
def load_X_data(X_file_path):
    
    try:
        # Load data from the CSV file  including the header to cudf dataframe
        df_X = pd.read_csv(X_file_path, sep=',' )  
        print ( "X  type ", (type(df_X)))       
        
        X_transposed = df_X.transpose() 
        print ( "X_transposed sliced dataframe shape", X_transposed.shape)
        
        X_cudf = cudf.DataFrame(X_transposed) 
        print ( "X_cudf", type(X_cudf))
       
        
        return X_cudf

    except Exception as e:
        print("Error occurred while loading features CSV data:", e)
        
        
        
# Load data from y_file_path csv file and return cudf dataframe       
def load_y_data(y_file_path):
    
    try:
       
        # Load data from the CSV file  including the header to cudf dataframe  
        df_Y = pd.read_csv(y_file_path, sep= ',' )
        print ( "y  shape ", df_Y.shape)
        
        # exclude the first column of samples'names 
        df_Y = df_Y.iloc[:, 1:]      
        print ( "y sliced dataframe shape", df_Y.shape)
        print ( "y sliced dataframe", df_Y)
        
        Y_transposed = df_Y.transpose()
        print ( "Y_transposed sliced dataframe shape", Y_transposed.shape)
        
        Y_cudf = cudf.DataFrame(Y_transposed)        
        print ( "Y_cudf", type(Y_cudf))                             
        
        return  Y_cudf

    except Exception as e:
        print("Error occurred while loading labels CSV data:", e)
        
        

# Combine X and Y datasets to dask_cudf
def combine_datasets(X_file_path, y_file_path, num_partitions):
    
    X = load_X_data(X_file_path)
    Y = load_y_data(y_file_path)
    
    
    combined_data = cudf.concat([X, Y], axis=0)
    print ( "combined_data shape", combined_data.shape)
    print ( "combined_data ", combined_data)
    
    # Create dask_cudf from cudf
    combined_data = dask_cudf.from_cudf(combined_data, npartitions= num_partitions)
    print ("combined_data type ", type (combined_data))
    
    return combined_data


        
# Perform KMeans clustering on combined dataset
def perform_clustering(combined_data, num_clusters):
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    cluster_labels = kmeans.fit_predict(combined_data)
    print ( "cluster_labels ", type (cluster_labels))    
    
    return cluster_labels


                              
# Find the cluster containing Y
def find_y_cluster(cluster_labels):
    
    # Assuming Y is the last row
    y_cluster = cluster_labels.iloc[len(cluster_labels) - 1]
    print ( "y_cluster ", y_cluster)
    
    return y_cluster 

                              

# Calculate distances between Y and features in the same cluster
def calculate_distances(combined_data,  cluster_labels, cluster_id):
    
    
    cluster_labels = cluster_labels.to_numpy()
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    print ( "cluster_indices includes  ", cluster_indices)
    print ( "cluster_indices type ", type(cluster_indices))
    
    featuresInClusterY  = combined_data.index[cluster_labels == cluster_id].to_numpy()
    print ( "featuresInClusterY type ", type(featuresInClusterY))
    print ( "featuresInClusterY  ", featuresInClusterY)
    
    distances = []
    
    for idx in cluster_indices:
        
        distances.append((idx, euclidean(combined_data.iloc[idx].to_numpy(), combined_data.iloc[len(cluster_labels) - 1].to_numpy())))

        
    sorted_distances = sorted(distances, key=lambda x: x[1])    
    print ( "sorted_distances ", sorted_distances)
    print ( "sorted_distances type ", type(sorted_distances))
    print ( "sorted_distances length", len(sorted_distances))
    
    
    return sorted_distances



# Select features closest to Y
def select_closest_features(X_file_path, distances, num_features):
     
    X = load_X_data(X_file_path)
    selected_features = []    
    
    for idx, _ in distances[:min(len(distances), num_features)]:
             
        if 0 <= idx < len(X):
        
            selected_features.append(X.iloc[idx])
            
        else:
            print(f"Index {idx} is out of bounds for DataFrame X")
            
    
    print ("selected_features type", type(selected_features))
     
    # Concatenate the DataFrames into a single cuDF DataFrame
    concat_features = cudf.concat(selected_features, axis= 1 )
    
    print ("concat_features type", type(concat_features))

    # Convert the combined cuDF DataFrame to a Pandas DataFrame
    selected_features = concat_features.to_pandas()

    print ( "selected_features type ", type(selected_features))
    print ( "selected_features shape ", selected_features.shape)
    
    selected_features = selected_features.transpose()
    print ( "transposed selected_features  shape ", selected_features.shape)
        
    return selected_features

                    

# Function gets the Pandas DataFrame as input and  write Pandas DataFrame to csv and returns file path dictionaty    
def write_to_csv(X_selected, output_path):
    
    print ("X_selected 2 type: ", type(X_selected))   
    print ("X_selected 2 shape : ", X_selected.shape) 
          
    
    # Create a separate directory for the output file
    try:
        
      # Create the output directory if it doesn't exist                                                        
       os.makedirs(output_path, exist_ok = True)     
       file_name = 'cuml_kmeans.csv'
       file_path = os.path.join(output_path, file_name)
                
       X_selected.to_csv(file_path, sep = ',', header =True, index = True ) 

       file_path_dict = {'cuml_kmeans': file_path}
       print("CSV file written successfully.")
       print ("CSV file size is  " , os.path.getsize(file_path))
       print ("CSV file column number is  " , X_selected.shape[1])
       print ("file_path_dictionary is  " , file_path_dict)
       return file_path 
   
    except Exception as e:
       print("Error occurred while writing matrices to CSV:", e)
       
    

    
        
if __name__ == '__main__': 
                 
    
    # Create Lucalluster with specification for each dask worker to create dask scheduler     
    cluster = LocalCUDACluster ()   
    
    #Create the Client using the cluster
    client = Client(cluster) 

       
    X_file_path = r'/features.csv' 
    y_file_path = r'/endpoint.csv'  
    output_path = r'output' 
    
    # Determine the number of clusters 
    num_clusters = 2
   
    # Change the npartitions based on how big is your data
    num_partitions =100
    
    combined_data = combine_datasets(X_file_path , y_file_path, num_partitions )
  
    # Perform KMeans clustering
    cluster_labels = perform_clustering(combined_data, num_clusters)
    
    # Find the cluster containing Y
    y_cluster = find_y_cluster(cluster_labels)
    
    # Calculate distances between Y and data points in the same cluster
    distances = calculate_distances(combined_data,  cluster_labels, y_cluster)
    
    # Select features closest to Y
    num_closest_features =10000

    closest_features = select_closest_features(X_file_path, distances, num_closest_features)
    
    path = write_to_csv(closest_features, output_path)    
    print ('path', path )
 
    scheduler_address = client.scheduler.address
    print("Scheduler Address:", scheduler_address)
    print(cluster.workers)
    print(cluster.scheduler)
    print(cluster.dashboard_link)
    print(cluster.status)
   
    client.close()
    cluster.close()       
    

            
