#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:16:28 2019

@author: mohityadav
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#Using Dendrogram to find optimal numbers of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')

#Fitting Hierarchical Clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualizing the results
plt.scatter(X[y_hc == 0,0], X[y_hc ==0,1], s=50, color ='red', label='Careful')
plt.scatter(X[y_hc == 1,0], X[y_hc ==1,1], s=50, color ='blue', label='Standard')
plt.scatter(X[y_hc == 2,0], X[y_hc ==2,1], s=50, color ='green', label='Target')
plt.scatter(X[y_hc == 3,0], X[y_hc ==3,1], s=50, color ='cyan', label='Careless')
plt.scatter(X[y_hc == 4,0], X[y_hc ==4,1], s=50, color ='magenta', label='Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score 1-100')
plt.legend()
plt.show()