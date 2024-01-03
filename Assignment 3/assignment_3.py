import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from multiprocessing import Pool
import time


def stencil(img, x, y):
    val = np.array([0, 0, 0])
    for i in range(-50, 50):  # 50 by 50 stencil columns
        for j in range(-50, 50):  # 50 by 50 stencil rows
            if (x + i < img.shape[1]) and (x + i >= 0) and \
                    (y + j < img.shape[0]) and (y + j >= 0):  # IndexOutOfBounds fix
                val += img[y + j, x + i]
    return img[y, x] * val // np.sum(val)


def apply_stencil(args):
    return stencil(*args)


def blurfilter(in_img, out_img):
    with Pool(8) as p:
        for x in range(in_img.shape[1]):
            args = ((in_img, x, y) for y in range(in_img.shape[0]))
            results = p.map(apply_stencil, args)
            out_img[:, x] = results
            print(f"Completed Blurring for Column {x}")
    return out_img


if __name__ == "__main__":
    img = np.array(Image.open('noisy1.jpg'))
    imgblur = img.copy()

    # Calculate the amount of time taken
    start = time.time()
    blurfilter(img, imgblur)
    end = time.time()
    print('Elapsed = % s' % (end - start))

    # Display and save blurred image
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Before')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(imgblur)
    ax.set_title('After')
    img2 = Image.fromarray(imgblur)
    img2.save('blurred.jpg')

