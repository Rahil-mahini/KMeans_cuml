# KMeans_cuml
This code computes the feature selection for a very large dataset of features of about 30 million features.
It calculate the KMeans using cuml.cluster on both features (X) and endpoints (Y) to find the distance of each feature that are in the same cluster as endpoint to select the closest features to endpoints.
First, load the features csv file (X) and endpoints csv file (Y) into cudf dataframe.
Next, combine both datasets (X and Y). It assume endpoint (Y) as one feature.
Next, perform KMeans, fit and predict the cluster labels for each feature (X) and endpoint (Y).
Next, find the cluster label of the endpoint (Y). 
Next, find the Euclidean distance of the features(X) that are in the same cluster as endpoint (Y), from the endpoint (Y). 
Then, sort the distances and find the k closet distance to the endpoint (Y).
Finally, extract the closest distance features (X) to the endpoint(Y) as selected features.	