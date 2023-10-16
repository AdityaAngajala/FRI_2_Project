import mplcursors
import numpy as np
import vtk
from matplotlib import pyplot as plt
from matplotlib import colors
import pyvista as pv
from pyvista.plotting.opts import PickerType
from topologyLabelGeneration import generate_noise


class Const:
    LAND_SIZE = 64
    NUM_COLORS = 24


data = []


def generate_terrain():
    height_map = np.zeros((Const.LAND_SIZE, Const.LAND_SIZE)).astype('float64')
    height_map += generate_noise(size=Const.LAND_SIZE)
    height_map = height_map / np.max(np.abs(height_map)) * Const.LAND_SIZE * (40 / 128)
    return height_map


def generate_2d_plot(plot, name, save=True):
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


def toggleHeightSlicer(slicer, axis: str, pl, mesh, clip=True):
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
        sliceDict['x'] = toggleHeightSlicer(sliceDict.get('x'), 'x', pl, mesh, clip=clip)

    def toggle_y():
        sliceDict['y'] = toggleHeightSlicer(sliceDict.get('y'), 'y', pl, mesh, clip=clip)

    def toggle_z():
        sliceDict['z'] = toggleHeightSlicer(sliceDict.get('z'), 'z', pl, mesh, clip=clip)

    pl.add_key_event('x', toggle_x)
    pl.add_key_event('y', toggle_y)
    pl.add_key_event('z', toggle_z)


def save_slices(xSlices=False, ySlices=False, zSlices=False):
    if xSlices:
        x_slices = [data[x, :, :] for x in reversed(range(Const.LAND_SIZE))]
        for num in range(len(x_slices)):
            generate_2d_plot(np.rot90(x_slices[num]), "xSlice" + str(num))
    if ySlices:
        y_slices = [data[:, y, :] for y in range(Const.LAND_SIZE)]
        for num in range(len(y_slices)):
            generate_2d_plot(np.rot90(y_slices[num]), "ySlice" + str(num))
    if zSlices:
        z_slices = [data[:, :, z] for z in range(Const.LAND_SIZE)]
        for num in range(len(z_slices)):
            generate_2d_plot(z_slices[num], "zSlice" + str(num))


def randomize_colors(grid):
    # Randomize Color of Each Block
    colorSet = np.random.randint(0, Const.NUM_COLORS + 1, np.arange(0, grid.GetNumberOfCells(), 1).shape).astype(float)
    colorSet -= 0.5
    grid.cell_data['Colors'] = colorSet


def generate(old_noise=False, clip=False, xSlices=False, ySlices=False, zSlices=False):
    # Generate Voxel Space
    x, y, z = np.indices((Const.LAND_SIZE + 1, Const.LAND_SIZE + 1, Const.LAND_SIZE + 1))
    grid = pv.StructuredGrid(z, y, x)
    x, y, z = np.indices((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))

    # 3D Noise Sampling for Colors

    freq = (2, 2, 2)
    noise3D = pv.perlin_noise(1, freq, (0, 0, 0))
    grid2 = pv.sample_function(noise3D, [-1, 1.0, -1, 1.0, -1, 1.0],
                               dim=(Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
    grid.cell_data['Colors'] = (np.round(grid2['scalars'] * Const.NUM_COLORS)) - 0.5

    # Label index before removing cells to get back original position
    grid.cell_data['cell_ind'] = np.arange(grid.GetNumberOfCells())
    # randomize_colors(grid)

    global data
    if old_noise:
        # Make Land Generation
        height_map = generate_terrain()
        terrain = np.logical_and(x >= 0, z < height_map[:, :, np.newaxis])  # 3D-bool array based on height map

        mesh = grid.cast_to_unstructured_grid()
        mesh.remove_cells(np.invert(terrain).flatten(), inplace=True)

        data = np.where(terrain.flatten(), grid.cell_data['Colors'], np.nan)
        data = data.reshape((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))
    else:
        grid = grid.threshold(0.2)
        mesh = grid.cast_to_unstructured_grid()

        data = np.full((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE), np.nan).flatten()

        for index in range(len(grid.cell_data['cell_ind'])):
            data[(grid.cell_data['cell_ind'])[index]] = grid.cell_data['Colors'][index]
        data = data.reshape((Const.LAND_SIZE, Const.LAND_SIZE, Const.LAND_SIZE))

    def printInfo(ok):
        if pl.picked_cell:
            coords = np.floor(pl.picked_cell.center).astype(int)
            print('X: ', coords[0], ' Y: ', coords[1], ' Z: ', coords[2])
            ind = (pl.picked_cell['cell_ind'])[0]  # [0] is because it is returned as array
            indexColor = list(grid.cell_data['cell_ind']).index(ind)
            print("Color: ", (grid.cell_data['Colors'])[indexColor])

    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True, cmap=init_colors(), scalars='Colors', clim=[0, Const.NUM_COLORS])
    enable_slicing(pl, mesh, clip=clip)
    pl.enable_element_picking(pickable_window=True, picker=PickerType.CELL, tolerance=0.001, callback=printInfo)
    save_slices(xSlices, ySlices, zSlices)
    pl.show(auto_close=False)


if __name__ == '__main__':
    generate(old_noise=False, clip=False, zSlices=True)
