


import numpy as np
import random
import matplotlib.pyplot as plt 



# Generate 10 random 2-dimensional vectors.
# Each vector is a list of 2 random floats between 0 and 1.
vectors = [ [random.random(), random.random()] for _ in range(10) ]

# Find the Frechet mean of the vectors.
# The Frechet mean is the point that minimizes the sum of squared distances to all points.

mean_vector = np.mean(vectors, axis=0)
# print("Mean vector:", mean_vector)

# Plot the vectors.
plt.figure(figsize=(8, 8))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title('Random Vectors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.scatter(*zip(*vectors), color='blue', label='Vectors')

# Plot the mean vector.
plt.scatter(mean_vector[0], mean_vector[1], color='red', label='Mean Vector', s=100, edgecolor='black')


plt.legend()
plt.show()