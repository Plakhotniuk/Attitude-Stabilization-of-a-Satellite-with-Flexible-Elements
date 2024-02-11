import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt("../tests/saved_data/point_potential_traj.txt")

times = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]
vx = data[:, 4]
vy = data[:, 5]
vz = data[:, 6]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

plt.show()
