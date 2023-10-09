import pickle

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
import numpy as np

colorOptions = []
colorOptionsHSV = []
precomputed_rgb = {}
precomputed_rounding = {}


# Create a function to update the color plot
def update(val, precompute=False):
    if precompute:
        saturation = val
    else:
        saturation = round(saturation_slider.val, 1)
    print(saturation)
    if saturation not in precomputed_rgb:
        h = np.linspace(0, 1, 360)
        l = np.linspace(0, 1, 100)
        H, L = np.meshgrid(h, l)
        S = np.full_like(H, saturation)
        HSV = np.stack((H, S, L), axis=-1)
        RGB = colors.hsv_to_rgb(HSV)
        if hsvModeOn:
            rounded_rgb = round_color_hsv(HSV)
        else:
            rounded_rgb = round_color_rgb(RGB)
        precomputed_rgb[saturation] = RGB
        precomputed_rounding[saturation] = rounded_rgb
        print("CALCULATING")
    else:
        RGB = precomputed_rgb[saturation]
        rounded_rgb = precomputed_rounding[saturation]
        print("PRECOMPUTE")

    if precompute:
        return

    ax1.clear()
    ax2.clear()

    ax1.imshow(RGB, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    ax2.imshow(rounded_rgb, origin='lower', extent=[0, 1, 0, 1], aspect='auto')

    ax1.set_xlabel('Hue')
    ax1.set_ylabel('Lightness')
    ax1.set_title(f'HSL Color Space (Saturation: {saturation})')

    ax2.set_xlabel('Hue')
    ax2.set_ylabel('Lightness')
    mode = 'HSV' if hsvModeOn else 'RGB'
    ax2.set_title(f'Rounded HSL Color Space (Saturation: {saturation}) Mode: {mode}')

    fig.canvas.draw_idle()


def round_color_rgb(input_color_array):
    output_array = np.zeros(input_color_array.shape)
    for i in range(len(input_color_array)):
        for j in range(len(input_color_array[0])):
            input_color = input_color_array[i][j]
            distances = []
            for color in colorOptions:
                distances.append((color[0] - input_color[0]) ** 2 +
                                 (color[1] - input_color[1]) ** 2 +
                                 (color[2] - input_color[2]) ** 2)
            output_array[i][j] = colors.to_rgb(colorOptions[(distances.index(min(distances)))])

    return output_array


def round_color_hsv(input_color_array):
    output_array = np.zeros(input_color_array.shape)
    for i in range(len(input_color_array)):
        for j in range(len(input_color_array[0])):
            input_color = input_color_array[i][j]
            distances = []
            for color in colorOptionsHSV:
                distances.append((color[0] - input_color[0]) ** 2 +
                                 (color[1] - input_color[1]) ** 2 +
                                 (color[2] - input_color[2]) ** 2)
            output_array[i][j] = colors.to_rgb(colorOptions[(distances.index(min(distances)))])

    return output_array


def reinitialize_color_order():
    with open("../colordump", "rb") as f:
        global colorOptions
        colorOptions = pickle.load(f)

    for color in colorOptions:
        colorOptionsHSV.append(colors.rgb_to_hsv(colors.to_rgb(color)))


if __name__ == '__main__':
    hsvModeOn = False

    # Initializes figures + create flider
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    saturation_slider_ax = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    saturation_slider = Slider(saturation_slider_ax, 'Saturation', 0, 1.0, valinit=0.5)
    saturation_slider.on_changed(update)

    reinitialize_color_order()

    # Precompute
    for step in np.arange(0, 1, 0.1):
        update(round(step, 1), precompute=True)

    update(None)

    plt.show()
    plt.close()
