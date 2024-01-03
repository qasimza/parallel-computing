import numpy as np
import time
from numba import jit, prange

@jit(nopython=True)
def mandel(x, y, max_iters):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


@jit(nopython=True, parallel=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in prange(width):
        real = min_x + x * pixel_size_x
        for y in prange(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


image = np.zeros((1024, 2024), dtype=np.uint8)

# Here compilation time is included
start = time.time()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
print('Elapsed (with compilation) = % s' % (end - start))

from matplotlib.pylab import imshow, show
imshow(image)
show()

# Here the pre-compiled code is timed
start = time.time()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
print('Elapsed (after compilation) = % s' % (end - start))

create_fractal.parallel_diagnostics(level=4)

imshow(image)
show()