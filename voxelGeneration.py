import math
import random
import cv2
import mplcursors
import numpy as np
import vtk
from matplotlib import pyplot as plt
from matplotlib import colors
from pyvista.plotting.opts import PickerType
from FRI_2_Project.utils.hilbert import gen_coords
import pyvista as pv
import os

from topologyLabelGeneration import generate_terrain, save_color_order


class Const:
    LAND_SIZE = 64
    NUM_COLORS = 48
    IMAGE_HEIGHT_CAP = 512
    MIN_ELEVATION = -32
    MAX_ELEVATION = 31
    NUM_INTERVALS = 12
    NUM_VALS_PER_INTERVAL = 1  # math.ceil((MAX_ELEVATION - MIN_ELEVATION + 1) / NUM_INTERVALS)
    VOXEL_DOWNSCALE = 1
    IMAGE_UPSCALE = 1


data = []
colorOutput = []


# noinspection DuplicatedCode
def plot_slice(plot, name, save=True, plotTerrain=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # if np.isnan(plot).all():
    #     return

    if not plotTerrain:
        cmap = init_colors()
        bounds = np.arange(0, Const.NUM_COLORS + 1, 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        cmap.set_bad('k', alpha=0)
        ax.imshow(plot, cmap=cmap, norm=norm)
    else:
        ax.imshow(plot, cmap='terrain')

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


def plot_sparse_interleave(input_data, name=''):
    outputPlot = np.full((Const.LAND_SIZE ** 2 * 2, Const.LAND_SIZE), np.nan)

    for index in range(len(input_data)):
        colorVals = [val[0] for val in input_data[index]]
        lenVals = [val[1] for val in input_data[index]]

        for index2 in range(len(colorVals)):
            outputPlot[index * 2][index2] = colorVals[index2]
            outputPlot[index * 2 + 1][index2] = lenVals[index2]

    # max_len = np.max([len(col) for col in input_data])
    max_len = Const.LAND_SIZE // Const.VOXEL_DOWNSCALE
    print("MAX LEN: ", max_len)

    write_image_interleave(outputPlot, max_len, upscale=1, name=name)


def plot_sparse_stacked(input_data, name=''):
    outputPlot = np.full((Const.LAND_SIZE ** 2, Const.LAND_SIZE * 2), np.nan)
    shouldPlotColor = np.full((Const.LAND_SIZE ** 2, Const.LAND_SIZE * 2), False)

    for index in range(len(input_data)):
        colorVals = [val[0] for val in input_data[index]]
        lenVals = [val[1] for val in input_data[index]]

        for index2 in range(len(colorVals)):
            outputPlot[index][index2] = colorVals[index2]
            shouldPlotColor[index][index2] = True
            outputPlot[index][index2 + len(colorVals)] = lenVals[index2]

    max_len = (Const.LAND_SIZE // Const.VOXEL_DOWNSCALE) * 2
    print("MAX LEN: ", max_len)

    write_image_stacked(outputPlot, shouldPlotColor, max_len, upscale=1, name=name)


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
            # columnSum += np.count_nonzero(~np.isnan(column))
            # sparseListSum += len(sparse_list(column)) * 2 - 2
            # prefilterSum += len(sparse_list(prefilter_sub(column))) * 2 - 2

    # printAvgColInfo(columnSum, sparseListSum, prefilterSum, data_sparse, data_prefilter)
    return data_sparse


def sparse_list(input_list):
    list_sparse = []
    count = 0
    input_list = np.where(np.isnan(input_list), np.min, input_list)
    prevVal = input_list[0]
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
    print("SparseList: ", sparseListSum)
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


def bin_val(val):
    if val <= 16 * Const.VOXEL_DOWNSCALE:
        return colorOutput[(val // Const.VOXEL_DOWNSCALE) - 1]
    else:
        return colorOutput[(((val // Const.VOXEL_DOWNSCALE) - 17) // 2) + 16]  # 2 = Range of Bins after 16 for 32B


def write_image_interleave(outputPlot, max_len, upscale=1, name=""):
    height_cap = Const.IMAGE_HEIGHT_CAP
    image = np.full((height_cap, math.ceil(len(outputPlot) / height_cap) * max_len, 3), np.nan)

    count = 0
    for i in range(len(outputPlot)):
        for j in range(max_len):
            outputI = i % height_cap
            outputJ = (count // height_cap) * max_len + j
            if np.isnan(outputPlot[i][j]):
                image[outputI][outputJ] = (255, 255, 255)
            else:
                val = math.floor(outputPlot[i][j])
                if i % 2 == 0:
                    image[outputI][outputJ] = colorOutput[val]
                else:
                    image[outputI][outputJ] = bin_val(val)

                # Flipping color channel from RGB to BGR
                image[outputI][outputJ] = np.flip(image[outputI][outputJ])
        count += 1

    cv2.imwrite('images3D/interleave/' + 'interleave' + name + '.png', upscale_image(image, upscale=upscale))


def write_image_stacked(outputPlot, shouldPlotColor, max_len, upscale=1, name=''):
    height_cap = Const.IMAGE_HEIGHT_CAP
    image = np.full((height_cap, math.ceil(len(outputPlot) / height_cap) * max_len, 3), np.nan)

    count = 0
    for i in range(len(outputPlot)):
        for j in range(max_len):
            outputI = i % height_cap
            outputJ = (count // height_cap) * max_len + j
            # print("I,J: ", outputI, ", ", outputJ)
            if np.isnan(outputPlot[i][j]):
                image[outputI][outputJ] = (255, 255, 255)
            else:
                val = outputPlot[i][j].astype(int)

                if shouldPlotColor[i][j]:
                    image[outputI][outputJ] = colorOutput[math.floor(val)]
                else:
                    image[outputI][outputJ] = bin_val(val)

            # Flipping color channel from RGB to BGR
            image[outputI][outputJ] = np.flip(image[outputI][outputJ])
        count += 1

    cv2.imwrite('images3D/stacked/' + 'stacked' + name + '.png', upscale_image(image, upscale=upscale))


def upscale_image(image, upscale=1):
    shape = (image.shape[0] * upscale, image.shape[1] * upscale, image.shape[2])
    image_upscale = np.full(shape, np.nan)
    for a in range(upscale):
        for b in range(upscale):
            image_upscale[a::upscale, b::upscale] = image
    return image_upscale


def upscale_data(downscaled_data, size, upscale):
    downscaled_data = np.where(np.isnan(downscaled_data), -1e6, downscaled_data)  # Hack to deal with np.nans
    data_upscale = np.full((size, size), -1).astype(float)
    for a in range(upscale):
        for b in range(upscale):
            data_upscale[a::upscale, b::upscale] = downscaled_data
    data_upscale = np.where(data_upscale == -1e6, np.nan, data_upscale)  # Undoing Hack
    return data_upscale


def upscale_voxel(voxel, upscale=1):
    shape = np.multiply(voxel.shape, upscale)
    voxel_upscale = np.full(shape, np.nan)
    for a in range(upscale):
        for b in range(upscale):
            for c in range(upscale):
                voxel_upscale[a::upscale, b::upscale, c::upscale] = voxel
    return voxel_upscale


# Really only works for 64x64x64 because the next size that get you a square image is 256x256x256 which is massive
def write_hilbert(name=''):
    global hilbert_x, hilbert_y, hilbert_z, hilbertX, hilbertY
    image = np.full((round(math.sqrt(Const.LAND_SIZE ** 3)), round(math.sqrt(Const.LAND_SIZE ** 3)), 3), np.nan)

    vals = []
    num = 0
    print(len(image))
    print("HILBER", len(hilbert_x))
    for i in range(len(image)):
        for j in range(len(image)):

            val = data[hilbert_x[num]][hilbert_y[num]][hilbert_z[num]]
            vals.append(val)
            if np.isnan(val):
                image[hilbertX[num]][hilbertY[num]] = (255, 255, 255)
            else:
                try:
                    image[hilbertX[num]][hilbertY[num]] = colorOutput[np.floor(val).astype(int)]
                    # Flipping color channel from RGB to BGR
                    image[hilbertX[num]][hilbertY[num]] = np.flip(image[hilbertX[num]][hilbertY[num]])
                except IndexError:
                    print(i, j)
                    print(val)
                    print(np.floor(val).astype(int))
                    print(colorOutput[val])
            num += 1

    # image = image.reshape(4096, 64, 3)

    cv2.imwrite('images3D/hilbert/' + 'hilbert' + name + '.png', upscale_image(image, upscale=1))


def write_slices(upscale=1, name=''):
    height_cap = Const.IMAGE_HEIGHT_CAP
    image = np.full((height_cap, height_cap, 3), np.nan)

    z_slices = [data[:, :, z] for z in range(Const.LAND_SIZE)]
    for num in range(len(z_slices)):
        z_slice = z_slices[num]
        for i in range(Const.LAND_SIZE):
            for j in range(Const.LAND_SIZE):
                outputI = ((num % 8) * Const.LAND_SIZE) + i
                outputJ = ((num // 8) * Const.LAND_SIZE) + j
                # print("I,J: ", outputI, ", ", outputJ)
                image[outputI][outputJ] = color_from_val(z_slice[i][j])

    cv2.imwrite('images3D/slices/' + 'slices' + name + '.png', upscale_image(image, upscale=upscale))


def write_expanded_slices(upscale=1, name='', seperated=True):
    image = np.full((Const.LAND_SIZE, Const.LAND_SIZE, 3), np.nan)
    z_slices = [data[:, :, z] for z in range(Const.LAND_SIZE)]
    for num in np.arange(0, len(z_slices), 2):
        z_slice = z_slices[num]
        for i in range(Const.LAND_SIZE):
            for j in range(Const.LAND_SIZE):
                image[i][j] = color_from_val(z_slice[i][j])

        folder_name = 'images3D/layerGroups/' + str(num) if seperated else 'images3D/layerGroups/' + name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        cv2.imwrite(folder_name + '/layer' + str(num) + '_Gen' + name + '.png', upscale_image(image, upscale=upscale))


def color_from_val(val):
    if np.isnan(val):
        return 255, 255, 255

    return np.flip(colorOutput[np.floor(val).astype(int)])


# noinspection DuplicatedCode
def init_colors(num_colors=Const.NUM_COLORS):
    # Start with a base of 20 colors taken from the nipy_spectral colormap, then add 6 more colors to the base

    if num_colors > 24:
        cmap = colors.ListedColormap(plt.get_cmap('nipy_spectral')(np.linspace(0, 1, Const.NUM_COLORS)))
        colorOptions = list(cmap.colors)

    else:
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

    if len(colorOutput) == 0:
        for colorVal in colorOptions:
            colorOutput.append(tuple(int(255 * val) for val in colors.ColorConverter.to_rgb(colorVal)))

    save_color_order(saveThing=colorOptions)

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


def save_slices(data, xSlices=False, ySlices=False, zSlices=False):
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


def gen_voxel_colors(grid, downscale=1):
    freq = (random.randrange(150, 300) / 100, random.randrange(150, 300) / 100, random.randrange(150, 300) / 100)
    noise3D = pv.perlin_noise(1, freq, (0, 0, 0))

    dimShape = (Const.LAND_SIZE // downscale, Const.LAND_SIZE // downscale, Const.LAND_SIZE // downscale)

    grid2 = pv.sample_function(noise3D, [-1, 1.0, -1, 1.0, -1, 1.0], dim=dimShape)

    colorThing = (np.array(grid2['scalars'])).reshape(dimShape)
    colorThing = upscale_voxel(colorThing, downscale).flatten()

    grid.cell_data['Colors'] = (np.round(colorThing * (Const.NUM_COLORS - 1))) - 0.5
    grid.cell_data['Colors'] = np.where(grid.cell_data['Colors'] > 0, grid.cell_data['Colors'], -0.5)

    # Label index before removing cells to get back original position
    grid.cell_data['cell_ind'] = np.arange(grid.GetNumberOfCells())


def split_colors(grid, mesh):
    half_one = (grid.cell_data['Colors'] + 0.5) // 12  # Base 24, for the 24 custom colors we are using
    half_two = (grid.cell_data['Colors'] + 0.5) % 12

    new_mesh_one = mesh.copy()
    new_mesh_two = mesh.copy()

    for mesh_ind in range(len(mesh.cell_data['cell_ind'])):
        new_mesh_one.cell_data['Colors'][mesh_ind] = half_one[new_mesh_one.cell_data['cell_ind'][mesh_ind]]
        new_mesh_two.cell_data['Colors'][mesh_ind] = half_two[new_mesh_two.cell_data['cell_ind'][mesh_ind]]

    return new_mesh_one, new_mesh_two


def randomize_colors(grid):
    # Randomize Color of Each Block
    colorSet = np.random.randint(0, Const.NUM_COLORS + 1, np.arange(0, grid.GetNumberOfCells(), 1).shape).astype(float)
    colorSet -= 0.5
    grid.cell_data['Colors'] = colorSet


# noinspection DuplicatedCode
def gen_voxels(old_noise, name=None, plotTerrain=False):
    # Generate Voxel Space
    x, y, z = np.indices((Const.LAND_SIZE + 1, Const.LAND_SIZE + 1, Const.LAND_SIZE + 1))
    grid = pv.StructuredGrid(z, y, x)
    x, y, z = np.indices((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
    print("Generated Voxel Space")

    # 3D Noise Sampling for Colors
    gen_voxel_colors(grid, Const.VOXEL_DOWNSCALE)
    init_colors()
    print("Generated Colors")
    # randomize_colors(grid)

    global data
    if old_noise:
        # Make Land Generation
        # if random.randrange(0, 10) > 5:
        #     hillsMax = 2
        #     hillsMin = 2
        #     basinMax = 0
        #     basinMin = 0
        # else:
        #     hillsMax = 0
        #     hillsMin = 0
        #     basinMax = 2
        #     basinMin = 2

        hillsMax = 2
        hillsMin = 1
        basinMax = 2
        basinMin = 1

        height_map = generate_terrain(name, MAX_ELEVATION=Const.MAX_ELEVATION, MIN_ELEVATION=Const.MIN_ELEVATION,
                                      NUM_VALS_PER_INTERVAL=1,
                                      LAND_SIZE=Const.LAND_SIZE // Const.VOXEL_DOWNSCALE,
                                      max_hills=hillsMax, min_hills=hillsMin, max_basins=basinMax, min_basins=basinMin,
                                      noise_normalize=(Const.LAND_SIZE // (3 + Const.VOXEL_DOWNSCALE)),
                                      radius_min=8 // Const.VOXEL_DOWNSCALE, radius_max=16 // Const.VOXEL_DOWNSCALE,
                                      min_height=15, max_height=25)
        height_map = upscale_data(height_map, Const.LAND_SIZE, upscale=Const.VOXEL_DOWNSCALE)

        height_map += Const.LAND_SIZE / 2  # Moving min to 0, comes from height of 64 cube
        terrain = np.logical_and(x >= 0, z < height_map[:, :, np.newaxis])  # 3D-bool array based on height map

        if plotTerrain:
            plot_slice(height_map, name, save=False, plotTerrain=plotTerrain)
            plt.show()

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

    data = np.where(data < 0, 0, data)
    data = np.where(data > Const.NUM_COLORS - 0.5, Const.NUM_COLORS - 0.5, data)

    return grid, mesh


def generate(old_noise=False, clip=False, plotTerrain=False, xCuts=False, yCuts=False, zCuts=False, name="", split=False):
    grid, mesh = gen_voxels(old_noise, plotTerrain=plotTerrain, name=name)
    print("Generated Voxels")

    # noinspection DuplicatedCode
    def printInfo(ok):
        if pl.picked_cell:
            coords = np.floor(pl.picked_cell.center).astype(int)
            print('X: ', coords[0], ' Y: ', coords[1], ' Z: ', coords[2])
            ind = (pl.picked_cell['cell_ind'])[0]  # [0] is because it is returned as array
            indexColor = list(grid.cell_data['cell_ind']).index(ind)
            print("Color: ", (grid.cell_data['Colors'])[indexColor])

    pl = pv.Plotter()
    if split:
        mesh2, mesh3 = split_colors(grid, mesh)
        pl.add_mesh(mesh2.translate((80, 0, 0), inplace=True), show_edges=True, cmap=init_colors(num_colors=24),
                    scalars='Colors', clim=[0, 24])
        pl.add_mesh(mesh3.translate((-80, 0, 0), inplace=True), show_edges=True, cmap=init_colors(num_colors=24),
                    scalars='Colors', clim=[0, 24])
    else:
        pl.add_mesh(mesh, show_edges=True, cmap=init_colors(), scalars='Colors', clim=[0, Const.NUM_COLORS])


    enable_slicing(pl, mesh, clip=clip)
    pl.enable_element_picking(pickable_window=True, picker=PickerType.CELL, tolerance=0.001, callback=printInfo)
    save_slices(data, xCuts, yCuts, zCuts)
    # data_sparse = sparse_data(data)
    # plot_sparse_interleave(data_sparse, name=name)
    # plot_sparse_stacked(data_sparse, name=name)
    # write_slices(name=name)
    # write_expanded_slices(name=name, upscale=8, seperated=False)
    # write_hilbert(name=name)

    pl.show(auto_close=False)


if __name__ == '__main__':
    # hilbert_x, hilbert_y, hilbert_z = gen_coords(dimSize=3, size_exponent=round(math.log2(Const.LAND_SIZE)))
    # hilbertX, hilbertY = gen_coords(dimSize=2, size_exponent=9)
    for num in range(50):
        generate(old_noise=True, clip=False, name=str(num), plotTerrain=False, split=False)
        print("PROGRESS: " + str(num + 1) + " / 30")
