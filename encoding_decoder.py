import tkinter
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
colorOutput = []
images = []


def get_files(is_dir=False):
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


def reinitialize_color_order():
    with open("colordump", "rb") as f:
        global colorOptions
        colorOptions = pickle.load(f)

    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))




def extract_data(image):
    values = np.arange(Const.MIN_ELEVATION, Const.MAX_ELEVATION + 1, Const.NUM_VALS_PER_INTERVAL)
    for i in range(Const.LAND_SIZE):
        for j in range(Const.LAND_SIZE):
            land[i][j] = values[color_to_values_index(image[i][j])]


def color_to_values_index(color):
    global colorOutput
    color = np.flip(color)  # BGR to RGB
    return colorOutput.index(tuple(color))


def generate_2d_plot(name, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Create a colormap based on the colorOptions
    bounds = np.arange(Const.MIN_ELEVATION, Const.MAX_ELEVATION + 1, Const.NUM_VALS_PER_INTERVAL)
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
    reinitialize_color_order()
    land = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE))

    get_files(is_dir=False)

    for file in images:
        extract_data(file)
        generate_2d_plot("output", save=True)
        generate_3d_visualization("output")
