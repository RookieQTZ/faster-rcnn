import numpy as np
import cv2


FILE = "../data/test/sobel/004430.jpg"


def sobel(img, threshold):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    # mag = np.copy(img)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))

    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q, 0] < threshold:
                mag[p, q] = 0
            if mag[p, q, 1] < threshold:
                mag[p, q] = 0
            if mag[p, q, 2] < threshold:
                mag[p, q] = 0
    return mag

if __name__ == '__main__':
    img = cv2.imread(FILE)
    cv2.imshow('img', img)

    sobel_img = sobel(img, 100)
    cv2.imshow("sobel", sobel_img)
    cv2.waitKey(0)