#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:59:57 2019

@author: mohityadav
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#Using elbows method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Applying Kmeans to Mall Dataset
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10,max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the Clusters
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans ==0,1], s=50, color ='red', label='Careful')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans ==1,1], s=50, color ='blue', label='Standard')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans ==2,1], s=50, color ='green', label='Target')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans ==3,1], s=50, color ='cyan', label='Careless')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans ==4,1], s=50, color ='magenta', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, color ='yellow', label='Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score 1-100')
plt.legend()
plt.show()