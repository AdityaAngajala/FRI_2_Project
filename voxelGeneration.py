import mayavi.mlab as mlab
import mplcursors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pyvista as pv
from pyvista.plotting.opts import PickerType
from topologyLabelGeneration import generate_noise


class Const:
    LAND_SIZE = 10 + 1


def generate_terrain():
    height_map = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE)).astype('float64')
    height_map += generate_noise(size=Const.LAND_SIZE)
    height_map = height_map / np.max(np.abs(height_map)) * Const.LAND_SIZE * (30 / 128)
    return height_map


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
        terrain = np.logical_and(x >= 0, z < height_map[:, :, np.newaxis])  # 3D-bool array based on height map
        grid.hide_points(np.invert(terrain).flatten())

    else:
        freq = (3, 3, 3)
        noise3D = pv.perlin_noise(1, freq, (0, 0, 0))
        grid = pv.sample_function(noise3D, [-1, 1.0, -1, 1.0, -1, 1.0],
                                  dim=(Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
        grid = grid.threshold(0.2)

    # Randomize Color of Each Block
    colorSet = np.random.randint(0, 200, np.arange(0, grid.GetNumberOfCells(), 1).shape)
    grid.cell_data['Colors'] = colorSet
    grid.cell_data['cell_ind'] = np.arange(grid.GetNumberOfCells())

    def printInfo(ok):
        if pl.picked_cell:
            coords = np.floor(pl.picked_cell.center).astype(int)
            print('X: ', coords[0], ' Y: ', coords[1], ' Z: ', coords[2])
            index = (pl.picked_cell['cell_ind'])[0]
            print("Color: ", (grid.cell_data['Colors'])[index])

    pl = pv.Plotter()
    pl.add_mesh(grid, show_edges=True, cmap=init_colors(), interpolate_before_map=False, scalars='Colors')
    pl.enable_element_picking(pickable_window=False, picker=PickerType.CELL, tolerance=0.001, callback=printInfo)

    pl.show(auto_close=False)

    # grid.plot(show_edges=True, cmap=init_colors(), interpolate_before_map=False, scalars='Colors')
    # mlab.points3d(*np.where(terrain), color=(1, 0, 0), scale_factor=1.0, mode='cube')
    # mlab.show()

if __name__ == '__main__':
    generate(old_noise=True)
