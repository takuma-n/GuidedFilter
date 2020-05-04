from guided_filter.results.smooth_noise import run
import numpy as np

if __name__ == '__main__':
    radius = np.arange(1, 51, 1)
    eps = np.arange(0.01, 0.11, 0.02)

    run('guide.png', 'src.png', radius, eps, same=True)