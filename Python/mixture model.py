# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:39:59 2023

@author: Gbenga Agunbiade
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the mixture model
n_samples = 10000
mean1, mean2 = [-4, 4]
variance1, variance2 = [1, 3]
p1 = 0.5  # proportion of samples from first cluster

# Generate the data
samples1 = np.random.normal(mean1, np.sqrt(variance1), int(n_samples * p1))
samples2 = np.random.normal(mean2, np.sqrt(variance2), int(n_samples * (1 - p1)))
data = np.concatenate([samples1, samples2])

# Plot the data
plt.hist(data, bins=50)
plt.show()

# Define the number of clusters
k = 2

# Initialize the means
np.random.seed(0)
means = np.random.uniform(-4, 4, k)

# Define the stopping criterion
tolerance = 1e-3

# Run the algorithm
while True:
    # Assign each data point to its nearest mean
    distances = np.abs(data[:, np.newaxis] - means)
    cluster_assignments = np.argmin(distances, axis=1)
    
    # Update the means
    new_means = np.array([data[cluster_assignments == i].mean() for i in range(k)])
    
    # Check for convergence
    if np.all(np.abs(new_means - means) < tolerance):
        break
        
    means = new_means
    
# Plot the results
plt.scatter(data, np.zeros_like(data), c=cluster_assignments)
plt.scatter(means, np.zeros_like(means), c='red', marker='x', s=100)
plt.show()