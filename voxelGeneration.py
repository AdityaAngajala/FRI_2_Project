import random
import mayavi.mlab as mlab
import mplcursors
import noise
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pyvista as pv
from pyvista import examples

from topologyLabelGeneration import quadratic_scaling


class Const:
    LAND_SIZE = 128


def generate_noise(noise_freq=60.0, octaves=6, persistence=0.5, lacunarity=2.0):
    z = random.random() * Const.LAND_SIZE
    terrain_noise = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE))
    for x in range(Const.LAND_SIZE):
        for y in range(Const.LAND_SIZE):
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


def generate_hill(radius, max_height, cy, cx, height_map):
    for x in range(max(0, int(cx - radius)), min(Const.LAND_SIZE, int(cx + radius))):
        for y in range(max(0, int(cy - radius)), min(Const.LAND_SIZE, int(cy + radius))):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                height_map[x, y] += max_height * quadratic_scaling(distance, radius)


def gen_rand_attributes(radius_min=10, radius_max=30, min_height=50, max_height=90):
    random_coord = random.randint(0, Const.LAND_SIZE - 1), random.randint(0, Const.LAND_SIZE - 1)
    random_radius = random.randint(radius_min, radius_max)
    random_height = random.randint(min_height, max_height)
    return random_coord, random_radius, random_height


def generate_terrain():
    height_map = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE)).astype('float64')
    height_map += generate_noise()
    height_map = height_map / np.max(np.abs(height_map)) * 30

    hillCount = random.randint(0, 3)
    rand_coord, randomRadius, randomHeight = gen_rand_attributes()

    # for _ in range(hillCount):
    #     generate_hill(randomRadius, randomHeight, rand_coord[0], rand_coord[1], height_map)

    return height_map


def boolean3d_2_points(specs):
    specsPoints = np.column_stack(np.where(specs))
    return specsPoints[:, 0], specsPoints[:, 1], specsPoints[:, 2]


def generate_2d_plot(heightthing):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    ax.imshow(heightthing, cmap='rainbow')

    # Cursor Annotations
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Get the coordinates of the selected point
        x, y = int(sel.target[0]), int(sel.target[1])
        value = heightthing[y][x]
        text = f"Point: ({x}, {y})\nValue: {value}"
        sel.annotation.set_text(text)

def init_colors():
    # Start with a base of 20 colors taken from the nipy_spectral colormap, then add 6 more colors to the base
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

    return colors.ListedColormap(colorOptions)


hmap = generate_terrain()

# prepare some coordinates
x, y, z = np.indices((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
bottom = np.logical_and((x >= 0) & (y >= 0), z < hmap[:, :, np.newaxis])

grid = pv.StructuredGrid(z, y, x)
ok = (np.invert(bottom).flatten())
grid.hide_points(ok)

colorSet = np.empty(x.shape, dtype=object)
colorSet = np.random.random_integers(0, 200, colorSet.shape)

grid['values'] = colorSet.flatten()
print(grid.DataSetFilters.extract_cells())

cmap = init_colors()


grid.plot(show_edges=True, cmap=cmap, interpolate_before_map=False)

# mlab.points3d(*boolean3d_2_points(bottom), color=(1, 0, 0), scale_factor=1.0, mode='cube')
# mlab.show()

