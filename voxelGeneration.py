import math
import random

import cv2
import mplcursors
import numpy as np
import vtk
from matplotlib import pyplot as plt
from matplotlib import colors
from bad.hilbert import gen_coords
import pyvista as pv
from pyvista.plotting.opts import PickerType
from topologyLabelGeneration import generate_noise


class Const:
    LAND_SIZE = 64
    NUM_COLORS = 24
    IMAGE_HEIGHT_CAP = 512


data = []
colorOutput = []


def plot_slice(plot, name, save=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    if np.isnan(plot).all():
        return

    cmap = init_colors()
    bounds = np.arange(0, Const.NUM_COLORS + 1, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad('k', alpha=0)
    ax.imshow(plot, cmap=cmap, norm=norm)

    if save:
        plt.savefig('images3D/' + str(name) + 'Plot.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        return

        # Cursor Annotations
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Get the coordinates of the selected point
        x, y = int(sel.target[0]), int(sel.target[1])
        value = plot[y][x]
        text = f"Point: ({x}, {y})\nValue: {value}"
        sel.annotation.set_text(text)


def plot_sparse(input_data):
    outputPlot = np.full((Const.LAND_SIZE ** 2 * 2, Const.LAND_SIZE), np.nan)

    for index in range(len(input_data)):
        colorVals = [val[0] for val in input_data[index]]
        lenVals = [val[1] for val in input_data[index]]

        for index2 in range(len(colorVals)):
            outputPlot[index * 2][index2] = colorVals[index2]
            outputPlot[index * 2 + 1][index2] = lenVals[index2]

    max_len = np.max([len(col) for col in input_data])
    print("MAX LEN: ", max_len)
    outputPlot = np.where(outputPlot < 0, 0, outputPlot)
    outputPlot = np.where(outputPlot > 23.5, 23.5, outputPlot)
    init_colors()

    write_image_interleave(outputPlot, max_len, upscale=1)


def sparse_data(input_data):
    columnSum = 0
    sparseListSum = 0
    prefilterSum = 0

    data_sparse = []
    data_prefilter = []

    for x in range(Const.LAND_SIZE):
        for y in range(Const.LAND_SIZE):
            column = input_data[y, x, :]  # Get Every Z Column

            data_sparse.append(sparse_list(column))
            data_prefilter.append(sparse_list(prefilter_sub(column)))

            # printColInfo(column)

            columnSum += np.count_nonzero(~np.isnan(column))
            sparseListSum += len(sparse_list(column)) * 2 - 2
            prefilterSum += len(sparse_list(prefilter_sub(column))) * 2 - 2

    # printAvgColInfo(columnSum, sparseListSum, prefilterSum, data_sparse, data_prefilter)
    plot_sparse(data_sparse)


def sparse_list(input_list):
    list_sparse = []
    count = 0
    prevVal = input_list[0]
    input_list = np.where(np.isnan(input_list), np.min, input_list)
    for val in input_list:
        if not (val == prevVal):
            if prevVal == np.min:
                prevVal = np.nan
            list_sparse.append((prevVal, count))
            count = 0
        prevVal = val
        count += 1
    if prevVal == np.min:
        prevVal = np.nan
    list_sparse.append((prevVal, count))
    return list_sparse


def printColInfo(column):
    print("Column: ", column)
    print("Column len: ", np.count_nonzero(~np.isnan(column)))
    print("Sparselist: ", sparse_list(column))
    print("Sparselist len: ", len(sparse_list(column)) * 2 - 2)
    print("Prefilter: ", sparse_list(prefilter_sub(column)))
    print("Prefilter Len: ", len(sparse_list(prefilter_sub(column))) * 2 - 2)


def printAvgColInfo(columnSum, sparseListSum, prefilterSum, data_sparse, data_prefilter):
    print("Column: ", columnSum)
    print("Avg: ", columnSum / Const.LAND_SIZE ** 2)
    print("Sparselist: ", sparseListSum)
    print("Avg: ", sparseListSum / Const.LAND_SIZE ** 2)
    print("Max: ", np.max([len(col) for col in data_sparse]))
    print("Prefilter: ", prefilterSum)
    print("Avg: ", prefilterSum / Const.LAND_SIZE ** 2)
    print("Max: ", np.max([len(col) for col in data_prefilter]))


def prefilter_sub(input_list):
    prevVal = 0
    output = []
    for val in input_list:
        output.append(val - prevVal)
        prevVal = val
    return output


def write_image_interleave(outputPlot, max_len, upscale=1):
    height_cap = Const.IMAGE_HEIGHT_CAP
    image = np.full((height_cap, (len(outputPlot) // height_cap) * max_len, 3), np.nan)

    count = 0
    for i in range(len(outputPlot)):
        for j in range(max_len):
            outputI = i % height_cap
            outputJ = (count // height_cap) * max_len + j
            # print("I,J: ", outputI, ", ", outputJ)
            if np.isnan(outputPlot[i][j]):
                image[outputI][outputJ] = (255, 255, 255)
            else:
                if i % 2 == 0:
                    image[outputI][outputJ] = colorOutput[np.floor(outputPlot[i][j]).astype(int)]
                    # Flipping color channel from RGB to BGR
                    image[outputI][outputJ] = np.flip(image[outputI][outputJ])
                else:
                    val = outputPlot[i][j] * 10
                    if val > 255:
                        val = outputPlot[i][j]
                    image[outputI][outputJ] = (val, val, val)
        count += 1

    cv2.imwrite('images3D/' + 'sdkjfsdkfj.png', upscale_image(image, upscale=upscale))


def upscale_image(image, upscale=1):
    shape = (image.shape[0] * upscale, image.shape[1] * upscale, image.shape[2])
    image_upscale = np.full(shape, np.nan)
    for a in range(upscale):
        for b in range(upscale):
            image_upscale[a::upscale, b::upscale] = image
    return image_upscale


# Really only works for 64x64x64 because the next size that get you a square image is 256x256x256 which is massive
def plot_hilbert(input_data):
    hilbert_x, hilbert_y, hilbert_z = gen_coords(dimSize=3, size_exponent=round(math.log2(Const.LAND_SIZE)))
    image = np.full((round(math.sqrt(Const.LAND_SIZE ** 3)), round(math.sqrt(Const.LAND_SIZE ** 3)), 3), np.nan)

    input_data = np.where(input_data < 0, 0, input_data)
    input_data = np.where(input_data > 23.5, 23.5, input_data)

    num = 0
    print(len(image))
    print("HILBER", len(hilbert_x))
    for i in range(len(image)):
        for j in range(len(image)):

            val = input_data[hilbert_z[num]][hilbert_y[num]][hilbert_x[num]]
            if np.isnan(val):
                image[i][j] = (255, 255, 255)
            else:
                try:
                    image[i][j] = colorOutput[np.floor(val).astype(int)]
                except IndexError:
                    print(i, j)
                    print(val)
                    print(np.floor(val).astype(int))
                    print(colorOutput[val])
            num += 1

    image = image.reshape(2048, 128, 3)
    cv2.imwrite('images3D/' + 'hilbert.png', upscale_image(image, upscale=1))


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

    for colorVal in colorOptions:
        colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))

    return colors.ListedColormap(colorOptions)


def toggleAxisSlicer(slicer, axis: str, pl, mesh, clip=True):
    pl.clear()
    if slicer:
        if clip:
            pl.add_mesh_clip_plane(mesh, show_edges=True, cmap=init_colors(),
                                   scalars='Colors', crinkle=True, invert=True, clim=[0, Const.NUM_COLORS],
                                   assign_to_axis=axis, interaction_event=vtk.vtkCommand.InteractionEvent)
        else:
            pl.add_mesh_slice(mesh, show_edges=True, cmap=init_colors(), clim=[0, Const.NUM_COLORS],
                              scalars='Colors', assign_to_axis=axis, interaction_event=vtk.vtkCommand.InteractionEvent)
    else:
        pl.add_mesh(mesh, show_edges=True, cmap=init_colors(), scalars='Colors', clim=[0, Const.NUM_COLORS])

    return not slicer


def enable_slicing(pl, mesh, clip=False):
    sliceDict = {'x': True, 'y': True, 'z': True}

    def toggle_x():
        sliceDict['x'] = toggleAxisSlicer(sliceDict.get('x'), 'x', pl, mesh, clip=clip)

    def toggle_y():
        sliceDict['y'] = toggleAxisSlicer(sliceDict.get('y'), 'y', pl, mesh, clip=clip)

    def toggle_z():
        sliceDict['z'] = toggleAxisSlicer(sliceDict.get('z'), 'z', pl, mesh, clip=clip)

    pl.add_key_event('x', toggle_x)
    pl.add_key_event('y', toggle_y)
    pl.add_key_event('z', toggle_z)


def save_slices(xSlices=False, ySlices=False, zSlices=False):
    if xSlices:
        x_slices = [data[x, :, :] for x in reversed(range(Const.LAND_SIZE))]
        for num in range(len(x_slices)):
            plot_slice(np.rot90(x_slices[num]), "xSlice" + str(num))
    if ySlices:
        y_slices = [data[:, y, :] for y in range(Const.LAND_SIZE)]
        for num in range(len(y_slices)):
            plot_slice(np.rot90(y_slices[num]), "ySlice" + str(num))
    if zSlices:
        z_slices = [data[:, :, z] for z in range(Const.LAND_SIZE)]
        for num in range(len(z_slices)):
            plot_slice(z_slices[num], "zSlice" + str(num))


def generate_terrain():
    height_map = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE)).astype('float64')
    height_map += generate_noise(size=Const.LAND_SIZE)
    height_map = height_map / np.max(np.abs(height_map)) * Const.LAND_SIZE * (40 / 128)
    return height_map


def gen_voxel_colors(grid):
    freq = (random.randrange(150, 300) / 100, random.randrange(150, 300) / 100, random.randrange(150, 300) / 100)
    noise3D = pv.perlin_noise(1, freq, (0, 0, 0))
    grid2 = pv.sample_function(noise3D, [-1, 1.0, -1, 1.0, -1, 1.0],
                               dim=(Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
    grid.cell_data['Colors'] = (np.round(grid2['scalars'] * (Const.NUM_COLORS - 1))) - 0.5
    grid.cell_data['Colors'] = np.where(grid.cell_data['Colors'] > 0, grid.cell_data['Colors'], -0.5)

    # Label index before removing cells to get back original position
    grid.cell_data['cell_ind'] = np.arange(grid.GetNumberOfCells())


def randomize_colors(grid):
    # Randomize Color of Each Block
    colorSet = np.random.randint(0, Const.NUM_COLORS + 1, np.arange(0, grid.GetNumberOfCells(), 1).shape).astype(float)
    colorSet -= 0.5
    grid.cell_data['Colors'] = colorSet


def gen_voxels(old_noise):
    # Generate Voxel Space
    x, y, z = np.indices((Const.LAND_SIZE + 1, Const.LAND_SIZE + 1, Const.LAND_SIZE + 1))
    grid = pv.StructuredGrid(z, y, x)
    x, y, z = np.indices((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
    print("Generated Voxel Space")

    # 3D Noise Sampling for Colors
    gen_voxel_colors(grid)
    init_colors()
    print("Generated Colors")
    # randomize_colors(grid)

    global data
    if old_noise:
        # Make Land Generation
        height_map = generate_terrain()
        terrain = np.logical_and(x >= 0, z < height_map[:, :, np.newaxis])  # 3D-bool array based on height map

        mesh = grid.cast_to_unstructured_grid()
        mesh.remove_cells(np.invert(terrain).flatten(), inplace=True)
        print("Terrain Made")

        data = np.where(terrain.flatten(), grid.cell_data['Colors'], np.nan)
        data = data.reshape((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))

    else:
        grid = grid.threshold(0.2)
        mesh = grid.cast_to_unstructured_grid()

        data = np.full((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE), np.nan).flatten()

        for index in range(len(grid.cell_data['cell_ind'])):
            data[(grid.cell_data['cell_ind'])[index]] = grid.cell_data['Colors'][index]
        data = data.reshape((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
        print("Terrain Made")

    return grid, mesh


def generate(old_noise=False, clip=False, xSlices=False, ySlices=False, zSlices=False):
    grid, mesh = gen_voxels(old_noise)
    print("Generated Voxels")

    def printInfo(ok):
        if pl.picked_cell:
            coords = np.floor(pl.picked_cell.center).astype(int)
            print('X: ', coords[0], ' Y: ', coords[1], ' Z: ', coords[2])
            ind = (pl.picked_cell['cell_ind'])[0]  # [0] is because it is returned as array
            indexColor = list(grid.cell_data['cell_ind']).index(ind)
            print("Color: ", (grid.cell_data['Colors'])[indexColor])

    # pl = pv.Plotter()
    # pl.add_mesh(mesh, show_edges=True, cmap=init_colors(), scalars='Colors', clim=[0, Const.NUM_COLORS])
    # enable_slicing(pl, mesh, clip=clip)
    # pl.enable_element_picking(pickable_window=True, picker=PickerType.CELL, tolerance=0.001, callback=printInfo)
    # save_slices(xSlices, ySlices, zSlices)
    sparse_data(data)
    plot_hilbert(data)
    # pl.show(auto_close=False)


if __name__ == '__main__':
    generate(old_noise=True, clip=False, xSlices=True, zSlices=True)
