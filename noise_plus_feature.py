import random
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import noise

land_size = 120

def generate_hill(radius, max_height, cy, cx, land):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] += max_height * ((radius - distance) / radius)

def generate_noise(scale=0.5, noise_freq=60.0, octaves=6, persistence=0.5, lacunarity=2.0):
    z = random.random() * land_size
    terrain_noise = np.zeros((land_size, land_size))
    for x in range(land_size):
        for y in range(land_size):
            terrain_noise[x][y] = noise.pnoise3(x / noise_freq, 
                                                y / noise_freq,
                                                z / noise_freq,
                                                octaves=octaves,
                                                persistence=persistence, 
                                                lacunarity=lacunarity, 
                                                repeatx=1024, 
                                                repeaty=1024, 
                                                base=42)
            terrain_noise[x][y] = (terrain_noise[x][y] + 1.0) * scale

    print(np.max(terrain_noise))
    return terrain_noise

def round_to_interval(value, interval):
    return interval * round(value / interval)

if __name__ == '__main__':
    land = np.zeros((land_size, land_size))

    generate_hill(30, 1.0, 50, 50, land)
    generate_hill(30, 1.0, 0, 0, land)
    generate_hill(30, 1.0, 20, 10, land)
    generate_hill(30, 1.0, 75, 90, land)
    land = land / np.max(land)
    
    land += generate_noise()
    land = land / np.max(land) * 255

    for i in range(land_size):
        for j in range(land_size):
            land[i][j] = round_to_interval(land[i][j], 16)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(land, cmap='terrain')

    # Add hover annotations
    cursor = mplcursors.cursor(im, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        x, y = int(sel.target[0]), int(sel.target[1])
        value = land[x][y]
        sel.annotation.set_text(f'Value: {value}')
    plt.show()

    lin_x = np.linspace(0, 1, land_size, endpoint=False)
    lin_y = np.linspace(0, 1, land_size, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, land, cmap='terrain')
    plt.savefig('3d_vizualization.png')
    plt.show()
