import random
import matplotlib.pyplot as plt
from matplotlib import colors
import mplcursors
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import noise
import cv2
import math

land_size = 128
land = np.zeros((land_size, land_size))
posXOptions = ["Left", "Center", "Right"]
posYOptions = ["Top", "Middle", "Bottom"]


def generate_hill(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] += max_height * quadratic_scaling(distance, radius)


def generate_basin(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] -= max_height * quadratic_scaling(distance, radius)


def generate_noise(noise_freq=60.0, octaves=6, persistence=0.5, lacunarity=2.0):
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
            terrain_noise[x][y] = (terrain_noise[x][y])

    print(np.max(terrain_noise))
    return terrain_noise


def round_to_interval(value, interval):
    return interval * round(value / interval)


def linear_scaling(distance, radius):
    return (radius - distance) / radius


def quadratic_scaling(distance, radius):
    return 1 - ((distance / radius) ** 2)


def gen_rand_attributes(radius_min=10, radius_max=30, min_height=50, max_height=90):
    random_coord = random.randint(0, land_size - 1), random.randint(0, land_size - 1)
    random_radius = random.randint(radius_min, radius_max)
    random_height = random.randint(min_height, max_height)
    return random_coord, random_radius, random_height


def set_poses(random_coord):
    return posXOptions[int(random_coord[0] // (land_size / 3))], posYOptions[int(random_coord[1] // (land_size / 3))]


def original_2d_vis(name):
    #plt.rcParams['figure.dpi'] = 12
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    #bounds = np.arange(0,257,32)
    #norm = colors.Normalize(vmin=-160, vmax=160)
    #norm = colors.BoundaryNorm(bounds, 8)

    ax.imshow(land, cmap='terrain')
    #ax.imshow(land, cmap='plasma', norm=norm)

    cursor = mplcursors.cursor(hover=True)

    # Save the image to a file
    plt.savefig('images/' + str(name) + '.png', bbox_inches='tight', pad_inches=0)

    @cursor.connect("add")
    def on_add(sel):
        # Get the coordinates of the selected point
        x, y = int(sel.target[0]), int(sel.target[1])
        value = land[y][x]
        text = f"Point: ({x}, {y})\nValue: {value}"
        sel.annotation.set_text(text)

def generate_2d_visualization(name, num_intervals, min_elevation, max_elevation):
    random_colors = [
        np.array([67, 185, 147]),
        np.array([124, 155, 213]),
        np.array([180, 184, 241]),
        np.array([87, 56, 39]),
        np.array([83, 105, 208]),
        np.array([38, 246, 175]),
        np.array([26, 20, 31]),
        np.array([113,  30, 174]),
        np.array([188, 109, 143]),
        np.array([36, 216,  26]),
        np.array([163, 102,  42]),
        np.array([98, 151, 220]),
        np.array([20, 144,  44]),
        np.array([156,  92, 228]),
        np.array([167,  98, 220]),
        np.array([21,  92, 3]),
        np.array([95,  74, 138]),
        np.array([35,  234, 73]),
        np.array([92,  64, 54]),
        np.array([82,  100, 83]),
        np.array([20,  160, 160]),
        np.array([34,  234, 56]),
        np.array([95,  74, 26]),
        np.array([37,  26, 94])
    ]
    num_vals_per_interval = math.ceil((max_elevation - min_elevation + 1) / num_intervals)
    image = np.zeros((land_size, land_size, 3))
    for i in range(land_size):
        for j in range(land_size):
            # print(int((land[i][j] + abs(min_elevation)) // num_vals_per_interval))
            image[i][j] = random_colors[int((land[i][j] + abs(min_elevation)) // num_vals_per_interval)]
    cv2.imwrite('images/' + str(name) + '.png', image)


def generate_3d_visualization(name):
    lin_x = np.linspace(0, 1, land_size, endpoint=False)
    lin_y = np.linspace(0, 1, land_size, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, land, cmap='terrain')
    plt.savefig('images/' + str(name) + '_3d_visualisation.png')
    #plt.show()

def generate_terrain(name, min_hills=0, max_hills=3, min_basins=0, max_basins=3, min_height=-192, max_height=191, num_intervals=24):
    global land
    land = np.zeros((land_size, land_size))

    land += generate_noise()
    land = land / np.max(np.abs(land)) * 50

    hillCount = random.randint(min_hills, max_hills)
    basinCount = random.randint(min_basins, max_basins)

    f = open('labels/' + str(name) + ".txt", "w")

    for _ in range(hillCount):
        rand_coord, randomRadius, randomHeight = gen_rand_attributes()
        generate_hill(randomRadius, randomHeight, rand_coord[0], rand_coord[1])
        posX, posY = set_poses(rand_coord)

        f.write("There is a Hill in the " + str(posY) + "-" + str(posX) + "\n")

    for _ in range(basinCount):
        rand_coord, randomRadius, randomHeight = gen_rand_attributes()
        generate_basin(randomRadius, randomHeight, rand_coord[0], rand_coord[1])
        posX, posY = set_poses(rand_coord)

        f.write("There is a Basin in the " + str(posY) + "-" + str(posX)+ "\n")

    f.close()

    for i in range(land_size):
        for j in range(land_size):
            land[i][j] = round_to_interval(land[i][j], int((max_height - min_height) / num_intervals))
    land = np.clip(land, min_height, max_height)
    print("min: ", np.min(land))
    print("max: ", np.max(land))

    generate_2d_visualization(name, num_intervals, min_height, max_height)
    generate_3d_visualization(name)

if __name__ == '__main__':
    for count in range(10):
        generate_terrain(count)
