import random
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

land_size = 100
posXOptions = ["Left", "Center", "Right"]
posYOptions = ["Top", "Middle", "Bottom"]


def generate_hill(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] += round_to_interval(round(max_height * quadratic_scaling(distance, radius)), 16);


def generate_basin(radius, max_height, cy, cx):
    for x in range(land_size):
        for y in range(land_size):
            # Calculate the distance from the current point to the center
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Check if the point is within the circular area
            if distance <= radius:
                land[x, y] -= round_to_interval(round(max_height * quadratic_scaling(distance, radius)), 16);


def round_to_interval(value, interval):
    return interval * round((value / interval));


def linear_scaling(distance, radius):
    return (radius - distance) / radius


def quadratic_scaling(distance, radius):
    return 1 - ((distance / radius) ** 2)


def get_rand_coord():
    return random.randint(0, land_size), random.randint(0, land_size)


if __name__ == '__main__':
    land = np.full((land_size, land_size), 50)

    hillCount = random.randint(0, 3)
    basinCount = random.randint(0, 3)

    for loop in range(hillCount):
        rand_coord = get_rand_coord()
        randomRadius = random.randint(10, 30)
        randomHeight = random.randint(160, 255)
        generate_hill(randomRadius, randomHeight, rand_coord[0], rand_coord[1])

        posX = posXOptions[int(rand_coord[0] // (land_size / 3))]
        posY = posYOptions[int(rand_coord[1] // (land_size / 3))]

        print("Hill", rand_coord, posX, posY)

    for loop in range(basinCount):
        rand_coord = get_rand_coord()
        randomRadius = random.randint(10, 30)
        randomHeight = random.randint(160, 255)
        generate_basin(randomRadius, randomHeight, rand_coord[0], rand_coord[1])

        posX = posXOptions[int(rand_coord[0] // (land_size / 3))]
        posY = posYOptions[int(rand_coord[1] // (land_size / 3))]

        print("Basin", rand_coord, posX, posY)



    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(land, cmap='nipy_spectral')

    # Add hover annotations
    cursor = mplcursors.cursor(im, hover=True)


    @cursor.connect("add")
    def on_add(sel):
        x, y = int(sel.target[0]), int(sel.target[1])
        value = land[x][y]
        sel.annotation.set_text(f'Value: {value}')


    plt.savefig('2d_vizualization.png')
    plt.show()

    lin_x = np.linspace(0, 1, land_size, endpoint=False)
    lin_y = np.linspace(0, 1, land_size, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, land, cmap='terrain')
    plt.savefig('3d_vizualization.png')
    plt.show()
