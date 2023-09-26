import random
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

land_size = 100

def generate_hill(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] += round_to_interval(round(max_height * ((radius - distance) / radius)), 16);

def round_to_interval(value, interval):
    return interval * round((value / interval));

if __name__ == '__main__':
    land = np.zeros((land_size, land_size))

    generate_hill(20, 255, 50, 50)
    generate_hill(20, 255, 0, 0)
    generate_hill(20, 255, 20, 10)
    generate_hill(20, 255, 75, 90)


    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(land, cmap='nipy_spectral')

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
