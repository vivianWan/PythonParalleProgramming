import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def parzen_estimation(x_samples, point_x, h):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        x_sample: training sample, 'd x 1' -dimensional numpy  array
        x: point x for density estimation, 'd x 1' -dimensional numpy array
        h: window width

    Returns the predicted pdf (probability density function) as float. 

    """
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:,np.newaxis]) / (h)
        for row in x_i:
            if np.abs(row) > (1/2):
                break
        else:  # "completion-else" 
                k_n += 1
    
    return (k_n /len(x_samples))/(h**point_x.shape[1])

fig = plt.figure(figsize= (7,7))
ax = fig.gca(projection = '3d')
ax.set_aspect("equal")

# Plot Points

# samples within th ecub
X_inside = np.array([[0,0,0],[0.2,0.2,0.2],[0.1,-0.1,-0.3]])

X_outside = np.array([[-1.2,0.3,-0.3],[0.8,-0.82,-0.9],[1,0.6,-0.7],[0.8,0.7,0.2],[0.7,-0.8,-0.45],[-0.3,0.6,0.9],[0.7,-0.6,-0.8]])

for row in X_inside:
    ax.scatter(row[0],row[1],row[2], color = 'r', s=50, marker = '^')

for row in X_outside:
    ax.scatter(row[0], row[1], row[2], color = 'k', s = 50)

# Plot Cube
h = [-0.5, 0.5]
for s, e in combinations(np.array(list(product(h,h,h))),2):
    if np.sum(np.abs(s-e)) == h[1] -h[0]:
        ax.plot3D(*zip(s,e), color = 'g')

ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(-1.5,1.5)

plt.show()

point_x = np.array([[0],[0],[0]])
X_all = np.vstack((X_inside, X_outside))

print('p(x) = ', parzen_estimation(X_all, point_x, h=1))