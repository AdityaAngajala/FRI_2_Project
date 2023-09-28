import random
import matplotlib.pyplot as plt
from matplotlib import colors
import mplcursors
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import noise


land_size = 128
land = np.full((land_size, land_size), 50)
posXOptions = ["Left", "Center", "Right"]
posYOptions = ["Top", "Middle", "Bottom"]


def generate_hill(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] += max_height * quadratic_scaling(distance, radius)


def generate_basin(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] -= max_height * quadratic_scaling(distance, radius)


def generate_noise(scale=0.5, noise_freq=60.0, octaves=6, persistence=0.5, lacunarity=2.0):
    z = random.random() * land_size
    terrain_noise = np.zeros((land_size, land_size))
    for x in range(land_size):
        for y in range(land_size):
            terrain_noise[x][y] = noise.pnoise3(x / noise_freq,
                                                y / noise_freq,
                                                z / noise_freq,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=1024,
                                                repeaty=1024,
                                                base=42)
            terrain_noise[x][y] = (terrain_noise[x][y] + 1.0) * scale

    print(np.max(terrain_noise))
    return terrain_noise


def round_to_interval(value, interval):
    return interval * round(value / interval)


def linear_scaling(distance, radius):
    return (radius - distance) / radius


def quadratic_scaling(distance, radius):
    return 1 - ((distance / radius) ** 2)


def gen_rand_attributes():
    random_coord = random.randint(0, land_size - 1), random.randint(0, land_size - 1)
    random_radius = random.randint(10, 30)
    random_height = random.randint(50, 90)
    return random_coord, random_radius, random_height


def set_poses(random_coord):
    return posXOptions[int(random_coord[0] // (land_size / 3))], posYOptions[int(random_coord[1] // (land_size / 3))]


def generate_2d_visualization(name):
    #plt.rcParams['figure.dpi'] = 12
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    #bounds = np.arange(0,257,32)
    #norm = colors.Normalize(vmin=-160, vmax=160)
    #norm = colors.BoundaryNorm(bounds, 8)

    ax.imshow(land, cmap='terrain')
    #ax.imshow(land, cmap='plasma', norm=norm)

    cursor = mplcursors.cursor(hover=True)

    # Save the image to a file
    plt.savefig('images/' + str(name) + '.png', bbox_inches='tight', pad_inches=0)

    @cursor.connect("add")
    def on_add(sel):
        # Get the coordinates of the selected point
        x, y = int(sel.target[0]), int(sel.target[1])
        value = land[y][x]
        text = f"Point: ({x}, {y})\nValue: {value}"
        sel.annotation.set_text(text)


def generate_3d_visualization(name):
    lin_x = np.linspace(0, 1, land_size, endpoint=False)
    lin_y = np.linspace(0, 1, land_size, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, land, cmap='terrain')
    plt.savefig('images/' + str(name) + '_3d_visualisation.png')
    #plt.show()

def generate_terrain(name):
    global land
    land = np.full((land_size, land_size), 1)

    hillCount = random.randint(0, 3)
    basinCount = random.randint(0, 3)

    f = open('labels/' + str(name) + ".txt", "w")

    for loop in range(hillCount):
        rand_coord, randomRadius, randomHeight = gen_rand_attributes()
        generate_hill(randomRadius, randomHeight, rand_coord[0], rand_coord[1])
        posX, posY = set_poses(rand_coord)

        f.write("There is a Hill in the " + str(posY) + "-" + str(posX) + "\n")

    for loop in range(basinCount):
        rand_coord, randomRadius, randomHeight = gen_rand_attributes();
        generate_basin(randomRadius, randomHeight, rand_coord[0], rand_coord[1])
        posX, posY = set_poses(rand_coord)

        f.write("There is a Basin in the " + str(posY) + "-" + str(posX)+ "\n")

    f.close()

    land = land / np.max(np.abs(land))
    land += generate_noise()
    land = land / np.max(np.abs(land)) * 160

    for i in range(land_size):
        for j in range(land_size):
            land[i][j] = round_to_interval(land[i][j], 16)

    generate_2d_visualization(name)
    #generate_3d_visualization(name)

if __name__ == '__main__':
    for count in range(10):
        generate_terrain(count)
