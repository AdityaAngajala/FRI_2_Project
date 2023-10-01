import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from random import random
from matplotlib.colors import ListedColormap, BoundaryNorm
from random import randint

shape = (500,500)
scale = 200.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
z = random()

num_slots = int(1 / 0.05)
random_colors = ['#%02X%02X%02X' % (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(num_slots)]
values = [i * .05 - .5 for i in range(num_slots)]
cmap = ListedColormap(colors=random_colors)
norm = BoundaryNorm(values, cmap.N)

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise3(i/500, 
                                    j/500,
                                    z,
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=42)

world *= 2
world = world.round(1)
world /= 2

plt.figure(figsize=(8, 8))
plt.imshow(world, cmap="terrain")#cmap=cmap, norm=norm)
plt.savefig('topological_map.png')
plt.show()

lin_x = np.linspace(0,1,shape[0],endpoint=False)
lin_y = np.linspace(0,1,shape[1],endpoint=False)
x,y = np.meshgrid(lin_x,lin_y)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, world, cmap="terrain")#cmap=cmap, norm=norm)
plt.savefig('3d_vizualization.png')
plt.show()