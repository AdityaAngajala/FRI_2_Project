import random
import matplotlib.pyplot as plt
from matplotlib import colors
import mplcursors
import numpy as np
import noise
import cv2
import math
import pickle
import re


class Const:
    LAND_SIZE = 128
    MIN_ELEVATION = -192
    MAX_ELEVATION = 191
    NUM_INTERVALS = 24
    NUM_VALS_PER_INTERVAL = math.ceil((MAX_ELEVATION - MIN_ELEVATION + 1) / NUM_INTERVALS)
    UPSCALE = 4


posXOptions = ["left", "center", "right"]
posYOptions = ["top", "middle", "bottom"]
colorOptions = []
colorOutput = []
basinHeightOptions = ["shallow", "", "deep"]
hillHeightOptions = ["short", "", "tall"]
widthOptions = ["narrow", "", "wide"]

land = []


def generate_hill(data, radius, max_height, cy, cx, land_size=Const.LAND_SIZE):
    for x in range(max(0, int(cx - radius)), min(land_size, int(cx + radius))):
        for y in range(max(0, int(cy - radius)), min(land_size, int(cy + radius))):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                data[x, y] += max_height * quadratic_scaling(distance, radius)

    return data


def generate_basin(data, radius, max_height, cy, cx, land_size=Const.LAND_SIZE):
    for x in range(max(0, int(cx - radius)), min(land_size, int(cx + radius))):
        for y in range(max(0, int(cy - radius)), min(land_size, int(cy + radius))):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                data[x, y] -= max_height * quadratic_scaling(distance, radius)

    return data


def generate_noise(noise_freq=60.0, octaves=6, persistence=0.5, lacunarity=2.0, size=Const.LAND_SIZE):
    z = random.random() * size
    terrain_noise = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            terrain_noise[x][y] = noise.pnoise3(x / noise_freq,
                                                y / noise_freq,
                                                z / noise_freq,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=1024,
                                                repeaty=1024,
                                                base=42)
            terrain_noise[x][y] = (terrain_noise[x][y] + 0.25)

    print(np.max(terrain_noise))
    return terrain_noise


def round_to_interval(value, interval):
    return interval * round(value / interval)


def linear_scaling(distance, radius):
    return (radius - distance) / radius


def quadratic_scaling(distance, radius):
    return 1 - ((distance / radius) ** 2)


def generate_rand_from_levels(min_val: int, max_val: int, level, num_levels, padding_ratio):
    range_per_level = (max_val - min_val) // num_levels
    return random.randint(0, int(range_per_level * (1 - 2 * padding_ratio))) + int(
        (level + padding_ratio) * range_per_level) + min_val


def gen_rand_attributes(quad_x, quad_y, height_opt, width_opt, radius_min=15, radius_max=35, min_height=50,
                        max_height=120, land_size=Const.LAND_SIZE):
    random_coord = generate_rand_from_levels(0, land_size - 1, quad_x, 3, 1 / 3), \
                   generate_rand_from_levels(0, land_size - 1, quad_y, 3, 1 / 3)
    random_radius = generate_rand_from_levels(radius_min, radius_max, width_opt, len(widthOptions), 0.1)
    random_height = generate_rand_from_levels(min_height, max_height, height_opt, len(hillHeightOptions), 0.1)
    return random_coord, random_radius, random_height


def set_poses(random_coord, land_size=Const.LAND_SIZE):
    return posXOptions[int(random_coord[0] // (land_size / 3))], \
        posYOptions[int(random_coord[1] // (land_size / 3))]


def initialize_colors(pastel=False):
    global colorOptions, colorOutput

    if pastel:
        # Start with a base of 20 colors, then add 4 more
        cmap = plt.get_cmap('tab20')
        colorOptions = cmap.colors
        colorOptions += 'darkblue', 'black', 'maroon', 'white'

        # Convert to a colormap in RGBA format
        cmap = colors.ListedColormap(colorOptions)
        colorOptions = colors.ListedColormap(cmap(np.linspace(0, 1, cmap.N))).colors

    else:
        # Start with a base of 20 colors takn from the nipy_spectral colormap, then add 6 more colors to the base
        cmap = colors.ListedColormap(plt.get_cmap('nipy_spectral')(np.linspace(0, 1, 20)))
        cmap2 = colors.ListedColormap(
            colors.ListedColormap(['chocolate', 'darkgreen', 'maroon',
                                   'slategray', 'fuchsia', 'hotpink', 'goldenrod'])(np.linspace(0, 1, 7)))
        colorOptions = np.concatenate([cmap.colors, cmap2.colors])

        # Remove the colors that are too similar to each other and shuffle the list
        colorOptions = list(colorOptions)
        colorOptions.pop(10)
        colorOptions.pop(11)
        colorOptions.pop(12)
        # random.shuffle(colorOptions)

    cmap = colors.ListedColormap(colorOptions)

    # Convert the colors to RGB values usable by opencv
    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))

    # Save the colormap to a file
    gradient = np.linspace(0, 1, 384)
    gradient = np.vstack((gradient, gradient))
    _, ax = plt.subplots(1, 1, figsize=(5, 1))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    plt.savefig('images/' + 'colorsPlot.png', bbox_inches='tight', pad_inches=0)


def reinitialize_color_order():
    global colorOutput
    with open("Save/colordump", "rb") as f:
        global colorOptions
        colorOptions = pickle.load(f)

    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))

    cmap = colors.ListedColormap(colorOptions)

    # Save the colormap to a file
    gradient = np.linspace(0, 1, 384)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(1, 1, figsize=(5, 1))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    plt.savefig('images/' + 'colorsPlot.png', bbox_inches='tight', pad_inches=0)


def save_color_order():
    with open("colordump", "wb") as f:
        pickle.dump(colorOptions, f)


def generate_2d_plot(data, name, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Create a colormap based on the colorOptions
    bounds = np.arange(Const.MIN_ELEVATION - 1, Const.MAX_ELEVATION + 1, Const.NUM_VALS_PER_INTERVAL)
    cmap = colors.ListedColormap(colorOptions)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(data, cmap=cmap, norm=norm)

    # ax.imshow(land, cmap='rainbow')

    # Save the image to a file
    if save:
        plt.savefig('images/' + str(name) + 'Plot.png', bbox_inches='tight', pad_inches=0)

    # Cursor Annotations
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Get the coordinates of the selected point
        x, y = int(sel.target[0]), int(sel.target[1])
        value = data[y][x]
        text = f"Point: ({x}, {y})\nValue: {value}"
        sel.annotation.set_text(text)


def generate_2d_visualization(data, name, upscale=1):
    global colorOutput
    image = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE, 3))
    for i in range(Const.LAND_SIZE):
        for j in range(Const.LAND_SIZE):
            image[i][j] = colorOutput[int((data[i][j] + abs(Const.MIN_ELEVATION)) // Const.NUM_VALS_PER_INTERVAL)]
            image[i][j] = np.flip(image[i][j])  # Flipping color channel from RGB to BGR

    image_upscale = np.zeros((Const.LAND_SIZE * upscale, Const.LAND_SIZE * upscale, 3))
    for a in range(upscale):
        for b in range(upscale):
            image_upscale[a::upscale, b::upscale] = image

    cv2.imwrite('images/' + str(name) + '.png', image_upscale)


def generate_3d_visualization(data, name):
    lin_x = np.linspace(0, 1, Const.LAND_SIZE, endpoint=False)
    lin_y = np.linspace(0, 1, Const.LAND_SIZE, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(y, x, data, cmap='terrain')
    plt.savefig('images/' + str(name) + '_3d_visualisation.png')
    # plt.close()
    # plt.show()


def generate_feature(data, generator, height_opt_arr, taken, f, name, land_size=Const.LAND_SIZE,
                     radius_min=15, radius_max=35, min_height=50, max_height=120):
    rand_idx = random.randrange(len(taken))
    rand_quad = taken[rand_idx]
    taken.pop(rand_idx)
    quad_x, quad_y = rand_quad // 3, rand_quad % 3
    height_opt = random.randrange(len(hillHeightOptions))
    width_opt = random.randrange(len(widthOptions))
    rand_coord, randomRadius, randomHeight = gen_rand_attributes(quad_x, quad_y, height_opt, width_opt,
                                                                 radius_min=radius_min, radius_max=radius_max,
                                                                 min_height=min_height, max_height=max_height, )
    data = generator(data, randomRadius, randomHeight, rand_coord[0], rand_coord[1], land_size=land_size)
    posX, posY = posXOptions[quad_x], posYOptions[quad_y]
    label = f"A {height_opt_arr[height_opt]}, {widthOptions[width_opt]} {name} in the {posY}-{posX}"
    label = re.sub(" , ", ' ', label)
    label = re.sub(",  ", ' ', label)
    label = re.sub(r'\s{2,}', ' ', label)
    f.write(label + '\n')
    return data


def generate_terrain(name, min_hills=0, max_hills=3, min_basins=0, max_basins=3, noise_normalize=30,
                     MAX_ELEVATION=Const.MAX_ELEVATION, MIN_ELEVATION=Const.MIN_ELEVATION,
                     NUM_VALS_PER_INTERVAL=Const.NUM_VALS_PER_INTERVAL, LAND_SIZE=Const.LAND_SIZE,
                     radius_min=15, radius_max=35, min_height=50, max_height=120):
    # Initialize the land array with zeros
    data = np.zeros((LAND_SIZE, LAND_SIZE))

    # Generate noise and normalize it before generating the hills and basins
    # to make it look more natural
    data += generate_noise(size=LAND_SIZE)
    data = data / np.max(np.abs(data)) * noise_normalize

    hillCount = random.randint(min_hills, max_hills)
    basinCount = random.randint(min_basins, max_basins)

    f = open('labels/' + str(name) + ".txt", "w")
    f.write("hbEncStyle\n")

    taken = [*range(9)]

    # Generate hills and save their labels
    for _ in range(hillCount):
        data = generate_feature(data, generate_hill, hillHeightOptions, taken, f, "hill", land_size=LAND_SIZE,
                                radius_min=radius_min, radius_max=radius_max,
                                min_height=min_height, max_height=max_height)

    for _ in range(basinCount):
        data = generate_feature(data, generate_basin, basinHeightOptions, taken, f, "basin", land_size=LAND_SIZE,
                                radius_min=radius_min, radius_max=radius_max,
                                min_height=min_height, max_height=max_height)

    if hillCount == 0 and basinCount == 0:
        f.write("There are no features.\n")

    f.close()

    # Round the heights to the nearest interval and clip them to the given height range
    for i in range(LAND_SIZE):
        for j in range(LAND_SIZE):
            data[i][j] = round_to_interval(data[i][j], NUM_VALS_PER_INTERVAL)
    data = np.clip(data, MIN_ELEVATION, MAX_ELEVATION)

    return data


if __name__ == '__main__':
    initialize_colors(pastel=False)
    for count in range(10):
        land = generate_terrain(count)
        generate_2d_visualization(land, count, upscale=Const.UPSCALE)
        generate_2d_plot(land, count, save=True)
        # generate_3d_visualization(land, count)

    save_color_order()
