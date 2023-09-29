import random
import matplotlib.pyplot as plt
from matplotlib import colors
import mplcursors
import numpy as np
import noise
import cv2
import math
import pickle

land_size = 128
posXOptions = ["Left", "Center", "Right"]
posYOptions = ["Top", "Middle", "Bottom"]
colorOptions = []
colorOutput = []


def generate_hill(radius, max_height, cy, cx):
    for x in range(max(0, int(cx - radius)), min(land_size, int(cx + radius))):
        for y in range(max(0, int(cy - radius)), min(land_size, int(cy + radius))):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] += max_height * quadratic_scaling(distance, radius)


def generate_basin(radius, max_height, cy, cx):
    for x in range(max(0, int(cx - radius)), min(land_size, int(cx + radius))):
        for y in range(max(0, int(cy - radius)), min(land_size, int(cy + radius))):
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


def initialize_colors(pastel=False):
    global colorOptions, colorOutput

    if pastel:
        # Start with a base of 20 colors
        cmap = plt.get_cmap('tab20')
        colorOptions = cmap.colors
        # Add 4 more colors to the base
        colorOptions += 'darkblue', 'black', 'maroon', 'white'
        # Convert to a colormap in RGBA format
        cmap = colors.ListedColormap(colorOptions)
        colorOptions = colors.ListedColormap(cmap(np.linspace(0, 1, cmap.N))).colors

    else:
        # Start with a base of 20 colors taken from the nipy_spectral colormap
        cmap = colors.ListedColormap(plt.get_cmap('nipy_spectral')(np.linspace(0, 1, 20)))
        # Add 6 more colors to the base
        cmap2 = colors.ListedColormap(
            colors.ListedColormap(['chocolate', 'darkgreen', 'maroon',
                                   'slategray', 'fuchsia', 'hotpink'])(np.linspace(0, 1, 6)))
        colorOptions = np.concatenate([cmap.colors, cmap2.colors])

        # Remove the colors that are too similar to each other and shuffle the list
        colorOptions = list(colorOptions)
        colorOptions.pop(10)
        colorOptions.pop(11)
        colorOptions.pop(12)
        random.shuffle(colorOptions)

    cmap = colors.ListedColormap(colorOptions)

    # Convert the colors to RGB values usable by opencv
    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))

    # Save the colormap to a file
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(1, 1, figsize=(5, 1))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    plt.savefig('images/' + 'colorsPlot.png', bbox_inches='tight', pad_inches=0)


def reinitialize_color_order():
    with open("colordump", "rb") as f:
        global colorOptions
        colorOptions = pickle.load(f)

    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))


def save_color_order():
    with open("colordump", "wb") as f:
        pickle.dump(colorOptions, f)


def generate_2d_plot(name, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Create a colormap based on the colorOptions
    bounds = np.arange(-192, 192, 16)
    cmap = colors.ListedColormap(colorOptions)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(land, cmap=cmap, norm=norm)

    # Save the image to a file
    if save:
        plt.savefig('images/' + str(name) + 'Plot.png', bbox_inches='tight', pad_inches=0)

    # Cursor Annotations
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Get the coordinates of the selected point
        x, y = int(sel.target[0]), int(sel.target[1])
        value = land[y][x]
        text = f"Point: ({x}, {y})\nValue: {value}"
        sel.annotation.set_text(text)


def generate_2d_visualization(name, num_intervals, min_elevation, max_elevation):
    num_vals_per_interval = math.ceil((max_elevation - min_elevation + 1) / num_intervals)
    image = np.zeros((land_size, land_size, 3))
    for i in range(land_size):
        for j in range(land_size):
            image[i][j] = colorOutput[int((land[i][j] + abs(min_elevation)) // num_vals_per_interval)]

    cv2.imwrite('images/' + str(name) + '.png', np.array(image)[..., ::-1])


def generate_3d_visualization(name):
    lin_x = np.linspace(0, 1, land_size, endpoint=False)
    lin_y = np.linspace(0, 1, land_size, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(y, x, land, cmap='terrain')
    plt.savefig('images/' + str(name) + '_3d_visualisation.png')
    plt.close()
    # plt.show()


def generate_terrain(name, min_hills=0, max_hills=3, min_basins=0, max_basins=3, min_height=-192, max_height=191,
                     num_intervals=24):
    global land

    # Initialize the land array with zeros
    land = np.zeros((land_size, land_size))

    # Generate noise and normalize it before generating the hills and basins
    # to make it look more natural
    land += generate_noise()
    land = land / np.max(np.abs(land)) * 50

    # Generate random number of hills and basins within the given range
    hillCount = random.randint(min_hills, max_hills)
    basinCount = random.randint(min_basins, max_basins)

    # Open text file to save the labels
    f = open('labels/' + str(name) + ".txt", "w")

    # Generate hills and save their labels
    for _ in range(hillCount):
        rand_coord, randomRadius, randomHeight = gen_rand_attributes()
        generate_hill(randomRadius, randomHeight, rand_coord[0], rand_coord[1])
        posX, posY = set_poses(rand_coord)
        f.write("There is a Hill in the " + str(posY) + "-" + str(posX) + "\n")

    # Generate basins and save their labels
    for _ in range(basinCount):
        rand_coord, randomRadius, randomHeight = gen_rand_attributes()
        generate_basin(randomRadius, randomHeight, rand_coord[0], rand_coord[1])
        posX, posY = set_poses(rand_coord)
        f.write("There is a Basin in the " + str(posY) + "-" + str(posX) + "\n")

    # Close text file
    f.close()

    # Round the heights to the nearest interval and clip them to the given height range
    for i in range(land_size):
        for j in range(land_size):
            land[i][j] = round_to_interval(land[i][j], int((max_height - min_height) / num_intervals))
    land = np.clip(land, min_height, max_height)

    # Generate and save 2D and 3D visualizations of the terrain
    #generate_2d_plot(name, save=False)
    generate_2d_visualization(name, num_intervals, min_height, max_height)
    #generate_3d_visualization(name)


if __name__ == '__main__':
    initialize_colors(pastel=False)
    for count in range(10):
        generate_terrain(count)
    save_color_order()
