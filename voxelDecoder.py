from enum import Enum

import mplcursors
import numpy as np
import pyvista as pv
from matplotlib import colors, pyplot as plt
from pyvista.plotting.opts import PickerType

from encoding_decoder import get_files, reinitialize_color_order, cv2_to_mpl, get_index_of_closest_color
from voxelGeneration import Const, enable_slicing

data = []
colorOptions = []
colorOptionsHSV = []
colorOutput = []


class Mode(Enum):
    STACKED = 1
    INTERLEAVE = 2
    HILBERT = 3
    SLICES = 4


def convert_image(image):
    size = Const.IMAGE_HEIGHT_CAP
    image_mpl = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            image_mpl[i][j] = cv2_to_mpl(image[i][j])

    return image_mpl


def downscale_image(image):
    image_downscale = np.zeros((
        Const.IMAGE_HEIGHT_CAP // Const.VOXEL_DOWNSCALE,
        Const.IMAGE_HEIGHT_CAP // Const.VOXEL_DOWNSCALE, 3))

    for a in range(Const.VOXEL_DOWNSCALE):
        for b in range(Const.VOXEL_DOWNSCALE):
            image_downscale += image[a::Const.VOXEL_DOWNSCALE, b::Const.VOXEL_DOWNSCALE]

    # Divide by 'upscale^2' to get the average value for each block
    image_downscale /= (Const.VOXEL_DOWNSCALE ** 2)

    return image_downscale


def extract_data(image, mode, hsv=False):
    x, y, z = np.indices((Const.LAND_SIZE + 1, Const.LAND_SIZE + 1, Const.LAND_SIZE + 1))
    grid = pv.StructuredGrid(z, y, x)

    image = convert_image(image)

    if mode == Mode.SLICES:
        extract_slices(image, hsv)
    elif mode == Mode.STACKED:
        extract_slices(image)
    elif mode == Mode.INTERLEAVE:
        extract_interleave()
    else:
        extract_hilbert()

    thing = data.flatten()

    grid.cell_data['Colors'] = data.flatten()
    grid.cell_data['cell_ind'] = np.arange(grid.GetNumberOfCells())

    mesh = grid.cast_to_unstructured_grid()
    mesh.remove_cells(np.isnan(data.flatten()), inplace=True)

    return grid, mesh


def extract_slices(image, hsv=False):
    image = downscale_image(image)
    values = np.arange(0.5, Const.NUM_COLORS + 1, 1)
    size = 64 // Const.VOXEL_DOWNSCALE

    colorOptionChoice = colorOptionsHSV if hsv else colorOptions

    for num in range(64):  # Num Slices
        z_slice = np.zeros((size, size))
        for i in range(size):  # Size of Slice
            for j in range(size):
                x = (num % 8) * size
                y = (num // 8) * size
                z_slice[i][j] = values[get_index_of_closest_color(image[x + i][y + j], colorOptionChoice, hsv)]
                if z_slice[i][j] > Const.NUM_COLORS:
                    z_slice[i][j] = np.nan

        z_slice = np.where(np.isnan(z_slice), -1e6, z_slice)
        slice_upscale = np.full((Const.LAND_SIZE, Const.LAND_SIZE), -1)
        for a in range(Const.VOXEL_DOWNSCALE):
            for b in range(Const.VOXEL_DOWNSCALE):
                slice_upscale[a::Const.VOXEL_DOWNSCALE, b::Const.VOXEL_DOWNSCALE] = z_slice
        slice_upscale = np.where(slice_upscale == -1e6, np.nan, slice_upscale)

        data[num] = slice_upscale

        # Create a figure with two subplots

        # cmap = colors.ListedColormap(colorOptions)
        # bounds = np.arange(0, Const.NUM_COLORS + 1, 1)
        # norm = colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad('k', alpha=0)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.imshow(slice_upscale, cmap=cmap, norm=norm)
        # plt.show()



def extract_stacked():
    return


def extract_interleave():
    return


def extract_hilbert():
    return


def display_data(image, old_noise=False, clip=False, mode=Mode.SLICES, hsv=False):
    grid, mesh = extract_data(image, mode=mode, hsv=hsv)

    def printInfo(ok):
        if pl.picked_cell:
            coords = np.floor(pl.picked_cell.center).astype(int)
            print('X: ', coords[0], ' Y: ', coords[1], ' Z: ', coords[2])
            ind = (pl.picked_cell['cell_ind'])[0]  # [0] is because it is returned as array
            indexColor = list(grid.cell_data['cell_ind']).index(ind)
            print("Color: ", (grid.cell_data['Colors'])[indexColor])

    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True, cmap=colors.ListedColormap(colorOptions),
                scalars='Colors', clim=[0, Const.NUM_COLORS])
    enable_slicing(pl, mesh, clip=clip)
    pl.enable_element_picking(pickable_window=True, picker=PickerType.CELL, tolerance=0.001, callback=printInfo)
    pl.show()


if __name__ == '__main__':
    colorOptions, colorOptionsHSV, colorOutput = reinitialize_color_order()


    data = np.full((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE), np.nan)

    for file in get_files(is_dir=False):
        display_data(file, mode=Mode.SLICES, hsv=False)
