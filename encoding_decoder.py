import tkinter
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import colors
import mplcursors
import numpy as np
import cv2
import pickle
from tkinter import filedialog
import os
from topologyLabelGeneration import Const

land = []
colorOptions = []
colorOptionsHSV = []
colorOutput = []


def get_files(is_dir=False):
    images = []

    if is_dir:
        src = filedialog.askdirectory()
        tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
        if len(src) == 0:
            exit()
        for f in os.listdir(src):
            images.append(cv2.imread(os.path.join(src, f)))
    else:
        src = filedialog.askopenfilename()
        if len(src) == 0:
            exit()
        tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
        images.append(cv2.imread(src))

    return images


def reinitialize_color_order():
    colorOptionsHSV = []
    colorOutput = []

    with open("colordump", "rb") as f:
        colorOptions = pickle.load(f)

    for color in colorOptions:
        colorOptionsHSV.append(colors.rgb_to_hsv(colors.to_rgb(color)))

    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))

    colorOptions.append((1, 1, 1))
    colorOptionsHSV.append(colors.rgb_to_hsv((1, 1, 1)))
    colorOutput.append((255, 255, 255))

    cmap = colors.ListedColormap(colorOptions)

    # Save the colormap to a file
    gradient = np.linspace(0, 1, 384)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(1, 1, figsize=(5, 1))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    plt.savefig('images/' + 'output_colorsPlot.png', bbox_inches='tight', pad_inches=0)

    return colorOptions, colorOptionsHSV, colorOutput


def extract_data(image, hsv=False):
    values = np.arange(Const.MIN_ELEVATION, Const.MAX_ELEVATION + 1, Const.NUM_VALS_PER_INTERVAL)

    size = Const.LAND_SIZE * Const.UPSCALE
    image_mpl = np.zeros((size, size, 3))

    for i in range(size):
        for j in range(size):
            image_mpl[i][j] = cv2_to_mpl(image[i][j])

    image_downscale = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE, 3))

    for a in range(Const.UPSCALE):
        for b in range(Const.UPSCALE):
            image_downscale += image_mpl[a::Const.UPSCALE, b::Const.UPSCALE]

    # Divide by 'upscale^2' to get the average value for each block
    image_downscale /= (Const.UPSCALE ** 2)

    colorOptionChoice = colorOptionsHSV if hsv else colorOptions

    for i in range(Const.LAND_SIZE):
        for j in range(Const.LAND_SIZE):
            land[i][j] = values[get_index_of_closest_color(image_downscale[i][j], colorOptionChoice, hsv=hsv)]


def cv2_to_mpl(color):
    # color = np.flip(color)  # BGR to RGB
    color = [(x / 255) for x in color]  # to Matplot Scheme
    return color


def get_index_of_closest_color(input_color, colorOptionChoice, hsv):
    if hsv:
        input_color = colors.rgb_to_hsv(input_color)

    distances = []
    for color in colorOptionChoice:
        distances.append((color[0] - input_color[0]) ** 2 +
                         (color[1] - input_color[1]) ** 2 +
                         (color[2] - input_color[2]) ** 2)
    return distances.index(min(distances))


def generate_2d_plot(name, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Create a colormap based on the colorOptions
    bounds = np.arange(Const.MIN_ELEVATION - 1, Const.MAX_ELEVATION + 1, Const.NUM_VALS_PER_INTERVAL)
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


def generate_3d_visualization(name):
    lin_x = np.linspace(0, 1, Const.LAND_SIZE, endpoint=False)
    lin_y = np.linspace(0, 1, Const.LAND_SIZE, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(y, x, land, cmap='terrain')
    plt.savefig('images/' + str(name) + '_3d_visualisation.png')
    # plt.close()
    plt.show()


if __name__ == '__main__':
    colorOptions, colorOptionsHSV, colorOutput = reinitialize_color_order()
    land = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE))

    for file in get_files(is_dir=False):
        extract_data(file, hsv=False)
        generate_2d_plot("output", save=True)
        generate_3d_visualization("output")
