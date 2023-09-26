import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from random import random

shape = (500,500)
scale = 200.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
z = random() * scale

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise3(i/scale, 
                                    j/scale,
                                    z,
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=42)

plt.figure(figsize=(8, 8))
plt.imshow(world,cmap='terrain')
plt.savefig('topological_map.png')
plt.show()

lin_x = np.linspace(0,1,shape[0],endpoint=False)
lin_y = np.linspace(0,1,shape[1],endpoint=False)
x,y = np.meshgrid(lin_x,lin_y)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x,y,world,cmap='terrain')
plt.savefig('3d_vizualization.png')
plt.show()