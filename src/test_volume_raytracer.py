import pytest
from volume_raytracer import OpticalVolume
import logging
import cupy as cp
import numpy as np


@pytest.mark.parametrize("ndim", [2, 3])
def test_gradient(ndim):
    shape = [100] + [10] * (ndim - 1)

    grid = cp.meshgrid(*[cp.linspace(0, 1, shp) for shp in shape], indexing='ij')
    ior = cp.clip(grid[0] * 3, 1, 2)
    transculency = cp.full(shape=shape, fill_value=1, dtype=cp.float32)
    scale = 1
    scale = [scale] * ndim
    volume = OpticalVolume(ior, transculency, scale)
    volume.update()
    positions = cp.zeros((2, ndim), dtype=cp.float32)
    directions = cp.zeros((2, ndim), dtype=cp.float32)
    positions[0, :] = cp.asarray([5] + [5] * (ndim - 1))
    positions[1, :] = cp.asarray([95] + [5] * (ndim - 1))
    directions[0, :] = cp.asarray([10] + [0] * (ndim - 1))
    directions[1, :] = cp.asarray([-10] + [0] * (ndim - 1))
    start_norm = cp.linalg.norm(directions, axis=-1).get()
    iterations = cp.full((1,), 10, dtype=cp.uint32)
    bounds = cp.array(shape, dtype=cp.float32)
    trajectories = []
    for i in range(1000):
        trajectories.append(positions.get())
        iterations[:] = 10
        volume.trace_rays(positions, directions, iterations, bounds)
    trajectories = np.asarray(trajectories)
    end_norm = cp.linalg.norm(directions, axis=-1).get()
    from matplotlib import pyplot as plt
    #plt.imshow(np.swapaxes(volume.gradient.get()[:, :, 0], 0, 1), origin='lower', extent=(0, ior.shape[0], 0, ior.shape[1]))
    #plt.imshow(ior.get(), extent=(0, ior.shape[1], 0, ior.shape[0]), cmap='gray')
    #plt.plot(trajectories[:,:, 0], trajectories[:,:, 1], 'ro')
    #plt.show()
    np.testing.assert_allclose(start_norm[0], end_norm[0] / 2, rtol=1e-2)
    np.testing.assert_allclose(start_norm[1], end_norm[1] * 2, rtol=1e-2)

def btest_sphere():
    logging.basicConfig(level=logging.DEBUG)
    shape = (10, 10)
    ior = cp.random.rand(*shape, dtype=cp.float32) * 0 + 1
    transculency = cp.full(shape=shape, fill_value=1, dtype=cp.float32)
    scale = (0.1,0.1)
    xx, yy = cp.meshgrid(cp.linspace(-1, 1, shape[0]), cp.linspace(-1, 1, shape[1]), indexing='ij')
    ior[xx ** 2 + yy ** 2 > 0.7] = 0.5
    volume = OpticalVolume(ior, transculency, scale)
    show_optical_figure(volume)


def btest_2d_tube():
    logging.basicConfig(level=logging.DEBUG)
    shape = (10, 10)
    ior = cp.random.rand(*shape, dtype=cp.float32) * 0 + 1
    transculency = cp.full(shape=shape, fill_value=1, dtype=cp.float32)
    scale = (1,1)
    xx, yy = cp.meshgrid(cp.linspace(-1, 1, shape[0]), cp.linspace(-1, 1, shape[1]), indexing='ij')
    ior[xx ** 2 > 0.6] = 0.1
    volume = OpticalVolume(ior, transculency, scale)
    show_optical_figure(volume)


def show_optical_figure(volume):
    #set logging level to DEBUG
    num_rays = 20
    positions = cp.random.rand( num_rays, 2, dtype=cp.float32) * 3 + 4
    directions = (cp.random.rand(num_rays, 2, dtype=cp.float32) - 0.5) * 20
    directions *= 100 / cp.linalg.norm(directions, axis=-1, keepdims=True)
    iterations = cp.full(num_rays, 10, dtype=cp.uint32)
    bounds = cp.array(volume.shape, dtype=cp.float32)
    trajectories = []
    for i in range(1000):
        trajectories.append(positions.get())
        iterations[:] = 10
        volume.trace_rays(positions, directions, iterations, bounds)
    from matplotlib import pyplot as plt
    trajectories = np.asarray(trajectories)
    #plt.imshow(np.concatenate((volume.gradient.get()[:, :, :], transculency[...,np.newaxis].get()), axis=-1))
    #create a figure with 2 subfigures
    fig, ax = plt.subplots(1, 2)
    volume.gradient[0, 4, 0] = 0
    ax[0].imshow(np.swapaxes(volume.gradient.get()[:, :, 0], 0, 1), extent=(0, volume.shape[1], 0, volume.shape[0]), origin='lower')
    ax[1].imshow(np.swapaxes(volume.gradient.get()[:, :, 1], 0, 1), extent=(0, volume.shape[1], 0, volume.shape[0]), origin='lower')
    ax[0].set_title('Gradient X')
    ax[1].set_title('Gradient Y')
    #set axis labels
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.swapaxes(volume.ior.get(), 0, 1), extent=(0, volume.shape[1], 0, volume.shape[0]), origin='lower', cmap='gray')
    ax.plot(trajectories[:, :, 0], trajectories[:, :, 1])
    ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], c='red', s=10, label='Start Positions')
    plt.show()
