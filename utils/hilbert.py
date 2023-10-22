import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
import pickle

N = 3
M = 6


def bin_str(i):
    """Return a string representation of i with N bits."""
    out = ''
    for j in range(N - 1, -1, -1):
        if (i >> j) & 1 == 1:
            out += '1'
        else:
            out += '0'
    return out


def rotate_right(x, d):
    """Rotate x by d bits to the right."""
    d = d % N
    out = x >> d
    for i in range(d):
        bit = (x & (2 ** i)) >> i
        out |= bit << (N + i - d)
    return out


def rotate_left(x, d):
    """Rotate x by d bits to the left."""
    d = d % N
    out = x << d
    excess = out
    out = out & (2 ** N - 1)
    for i in range(d):
        bit = (x & (2 ** (N - 1 - d + 1 + i))) >> (N - 1 - d + 1 + i)
        out |= bit << i
    return out


def bit_component(x, i):
    """Return the i-th bit of x"""
    return (x & (2 ** i)) >> i


# verify that '~i & 2**N-1' performs the NOT operation in N-bit space
for i in range(2 ** N):
    not_i = ~i & (2 ** N - 1)
    assert not_i >= 0
    assert not_i < 2 ** N
    assert i & not_i == 0
    assert i | not_i == (2 ** N - 1)


# Define other functions and variables

def gc(i):
    """Return the Gray code index of i."""
    return i ^ (i >> 1)


def e(i):
    """Return the entry point of hypercube i."""
    if i == 0:
        return 0
    else:
        return gc(2 * int(math.floor((i - 1) // 2)))


def f(i):
    """Return the exit point of hypercube i."""
    return e(2 ** N - 1 - i) ^ 2 ** (N - 1)


def i_to_p(i):
    """Extract the 3D position from a 3-bit integer."""
    return [bit_component(i, j) for j in (0, 1, 2)]


def add_edges(edges):
    """Extend the list of edges from a hypercube to the list of
    edges of the hypercube of the next dimension."""
    old_edges = list(edges)
    old_points = set([x[0] for x in old_edges]) | set([x[1] for x in old_edges])
    edges = [((0,) + x[0], (0,) + x[1]) for x in old_edges]
    edges.extend([((1,) + x[0], (1,) + x[1]) for x in old_edges])
    for e in old_points:
        edges.append(((0,) + e, (1,) + e))
    return edges


def set_unit_cube(ax, side=1, set_view=(10, -67)):
    """Present the unit cube."""
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_xticks(range(side + 1))
    # ax.set_yticks(range(side + 1))
    # ax.set_zticks(range(side + 1))
    # ax.set_xlim(0, side)
    # ax.set_ylim(0, side)
    # ax.set_zlim(0, side)
    if set_view:
        ax.view_init(*set_view)


def inverse_gc(g):
    """The inverse gray code."""
    i = g
    j = 1
    while j < N:
        i = i ^ (g >> j)
        j = j + 1
    return i


def g(i):
    """The direction between subcube i and the next one"""
    return int(np.log2(gc(i) ^ gc(i + 1)))


def d(i):
    """The direction of the arrow whithin a subcube."""
    if i == 0:
        return 0
    elif (i % 2) == 0:
        return g(i - 1) % N
    else:
        return g(i) % N


def T(e, d, b):
    """Transform b."""
    out = b ^ e
    return rotate_right(out, d + 1)


def T_inv(e, d, b):
    """Inverse transform b."""
    return T(rotate_right(e, d + 1), N - d - 2, b)


def TR_algo2(p):
    """Return the Hilbert index of point p"""
    # h will contain the Hilbert index
    h = 0
    # ve and vd contain the entry point and dimension of the current subcube
    # we choose here a main traversal direction N-2 (i.e. z for a cube) to match
    # the illustrations
    ve = 0
    vd = 2
    for i in range(M - 1, -1, -1):
        # the cell label is constructed in two steps
        # 1. extract the relevant bits from p
        l = [bit_component(px, i) for px in p]
        # 2. construct a integer whose bits are given by l
        l = sum([lx * 2 ** j for j, lx in enumerate(l)])
        # transform l into the current subcube
        l = T(ve, vd, l)
        # obtain the gray code ordering from the label l
        w = inverse_gc(l)
        # compose (see [TR] lemma 2.13) the transform of ve and vd
        # with the data of the subcube
        ve = ve ^ (rotate_left(e(w), vd + 1))
        vd = (vd + d(w) + 1) % N
        # move the index to more significant bits and add current value
        h = (h << N) | w
    return h


def TR_algo3(h):
    """Return the coordinates for the Hilbert index h"""
    ve = 0
    vd = 2
    p = [0] * N
    for i in range(M - 1, -1, -1):
        w = [bit_component(h, i * N + ii) for ii in range(N)]
        # print(i, w)
        w = sum([wx * 2 ** j for j, wx in enumerate(w)])
        # print(i, w, gc(w))
        l = gc(w)
        l = T_inv(ve, vd, l)
        for j in range(N):
            p[j] += bit_component(l, j) << i
        ve = ve ^ rotate_left(e(w), vd + 1)
        vd = (vd + d(w) + 1) % N
    return p


def gen_coords(dimSize, size_exponent):
    global N, M
    N = dimSize
    M = size_exponent
    pointsX = []
    pointsY = []
    pointsZ = []
    # points = []
    for h in range(2 ** (dimSize * size_exponent)):
        if dimSize == 3:
            x, y, z = TR_algo3(h)
            pointsX.append(x)
            pointsY.append(y)
            pointsZ.append(z)
        else:
            x, y = TR_algo3(h)
            pointsX.append(x)
            pointsY.append(y)
        # points.append(x, y, z)
        # print(x, y, z)
    if dimSize == 3:
        return pointsX, pointsY, pointsZ
    else:
        return pointsX, pointsY

if __name__ == '__main__':
    x, y, z = gen_coords(N, M)
    save = (x, y, z)
    with open("hilbert3D", "wb") as f:
        pickle.dump(save, f)

    for i in range(10):
        print("(", x[i], y[i], z[i], ")")

    x, y = gen_coords(2, 9)
    save = (x, y)
    with open("hilbert2D", "wb") as f:
        pickle.dump(save, f)

    for i in range(10):
        print("ok (", x[i], y[i], ")")
    #
    # print(len(x))
    #
    # points = np.column_stack((x, y, z))
    # cloud = pv.PolyData(points)
    #
    # plotter = pv.Plotter()
    #
    # # Add the 3D scatter plot
    # # plotter.add_points(cloud, color='b', point_size=5, render_points_as_spheres=True)
    #
    # # Add lines to connect points
    # for i in range(1, len(x)):
    #     line = pv.Line([y[i - 1], z[i - 1], x[i - 1]], [y[i], z[i], x[i]])
    #     plotter.add_mesh(line, color='r')
    #
    # # Display the plot
    # plotter.show()
