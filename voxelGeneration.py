import mayavi.mlab as mlab
import mplcursors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pyvista as pv
from topologyLabelGeneration import generate_noise


class Const:
    LAND_SIZE = 128 + 1


def generate_terrain():
    height_map = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE)).astype('float64')
    height_map += generate_noise(size=Const.LAND_SIZE)
    height_map = height_map / np.max(np.abs(height_map)) * 30

    return height_map


def boolean3d_2_points(specs):
    specsPoints = np.column_stack(specs)
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

def generate(old_noise=False):
    # Generate Voxel Space
    x, y, z = np.indices((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
    grid = pv.StructuredGrid(z, y, x)

    if (old_noise):
        # Make Land Generation
        height_map = generate_terrain()
        terrain = np.logical_and(x >= 0, z < height_map[:, :, np.newaxis]) # 3D-bool array based on height map
        grid.hide_points(np.invert(terrain).flatten())
    else:
        freq = (3, 3, 3)
        noise3D = pv.perlin_noise(1, freq, (0, 0, 0))
        grid = pv.sample_function(noise3D, [-1, 1.0, -1, 1.0, -1, 1.0], dim=(Const.LAND_SIZE - 1, Const.LAND_SIZE - 1, Const.LAND_SIZE - 1))
        grid = grid.threshold(0.2)

    # Randomize Color of Each Block
    colorSet = np.random.randint(0, 200, np.arange(0, grid.GetNumberOfCells(), 1).shape)
    grid.cell_data['Colors'] = colorSet

    grid.plot(show_edges=True, cmap=init_colors(), interpolate_before_map=False, scalars='Colors')

    # mlab.points3d(*boolean3d_2_points(bottom), color=(1, 0, 0), scale_factor=1.0, mode='cube')
    # mlab.show()

if __name__ == '__main__':
    generate(old_noise=True)