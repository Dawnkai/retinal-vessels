## Image manipulation and morphology

import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import listdir
from skimage.filters import frangi, hessian
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math

# Input image directory
IMG_DIR = 'dataset/test'
# List of files in the folder above
image_names = [f"{IMG_DIR}/images/{img}" for img in listdir(f"{IMG_DIR}/images")]
mask_names = [f"{IMG_DIR}/ground_truth/{img}" for img in listdir(f"{IMG_DIR}/ground_truth")]

def invert_colors(img):
    '''
    Invert colors in the image.
    :param img: - input image
    :return new_image: - image with inverted colors
    '''
    new_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for x, row in enumerate(img):
        for y, _ in enumerate(row):
            new_image[x][y] = abs(255 - img[x][y])
    return new_image


def remove_background(img, mask):
    '''
    Remove background using mask, in other words set pixels to black if the pixel of the
    same coordinates in mask is not black.
    :param img: - input image
    :param mask: - mask image used to remove background
    :return new_image: - image without the background
    '''
    new_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for x, row in enumerate(img):
        for y, _ in enumerate(row):
            new_image[x][y] = img[x][y] if mask[x][y] == 0 else 0
    return new_image


# 1. Read image
def read_image(filepath):
    '''
    Standard function to read image
    :param filepath: - image filepath
    :return img: - image as numpy array
    '''
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # OpenCV 2 reads images in BGR, so we convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 2. Extract background mask from image
def get_mask(img, threshold = 5):
    '''
    Extract background mask from image by removing pixels above threshold and converting those
    below it to white.
    :param img: - input image
    :param threshold: - the threshold
    :return img_mask: - result mask
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for x, row in enumerate(img_mask):
        for y, _ in enumerate(row):
            # Thresholding
            img_mask[x][y] = 255 if img[x][y] < threshold else 0
    return img_mask


# 3. Extract green channel from the image
def get_green_channel(img):
    '''
    Extract green channel from image.
    :param img: - input image
    :return green_img: - result image with only green channel
    '''
    _, green_img, _ = cv2.split(img)
    green_img = invert_colors(green_img)
    return green_img


# 4. Small noise removal
def remove_noise(img, min_size = 600):
    '''
    Remove small dots from image.
    :param img: - input image
    :param min_size: - minimum size of objects to not be removed
    :return clear_img: - image without noise
    '''
    # Detect connected objects in image
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # Extract sizes
    sizes = stats[1:, -1]
    clear_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(0, nb_components - 1):
        # If the object is bigger than min_size, redraw it on output image
        if sizes[i] >= min_size:
            clear_img[output == i + 1] = 255
    return clear_img


# 5. Remove big circle around retina
def remove_retina_circle(img, base_offset=2000):
    '''
    Remove big circle around retina using Hough circles.
    Algorithm will search until it finds one circle of specific size in the image.
    If that is not possible, no changes are made.
    :param img: - input image
    :param base_offset: - start searching from this size (speeds up computation)
    :return no_retina_img: - image without the circle
    '''
    no_retina_img = img.copy()
    circle = np.zeros(img.shape, np.uint8)
    offset = 0
    detections = []
    while True:
        # Detect Hough circles from base_offset + offset size
        detections = cv2.HoughCircles(no_retina_img, cv2.HOUGH_GRADIENT, 1.5, base_offset + offset)
        # Stop if no circles can be detected
        if detections is None:
            return no_retina_img
        # If only one circle is detected, finish the search
        if len(detections[0]) == 1:
            break
        # IOtherwise increase size
        offset += 100
    for (x, y, r) in detections[0]:
        # Draw a circle to be removed
        cv2.circle(circle, (int(x), int(y)), int(r), (255, 255, 255), 40)
    # Circle removal
    no_retina_img = no_retina_img - circle
    return no_retina_img
    

def detect_veins(filepath):
    '''
    Function that merges the functionality above.
    :param filepath: - input image path
    '''
    print("Reading image...")
    img = read_image(filepath)
    print("Extracting mask...")
    img_mask = get_mask(img)
    print("Extracting green channel...")
    img = get_green_channel(img)
    print("Removing background...")
    bck_img = remove_background(img, img_mask)
    print("Equalizing histograms...")
    eq_img = cv2.equalizeHist(bck_img)
    print("Extracting veins using hessian...")
    hes_img = hessian(eq_img)
    veins = remove_background(hes_img, img_mask)
    print("Using bilateral filter...")
    filtered_img = cv2.bilateralFilter(veins,11,75,75)
    print("Removing noise...")
    clear_img = remove_noise(filtered_img)
    print("Removing retina's circle...")
    result = remove_retina_circle(clear_img)
    print("Image processing done.")
    return result


if __name__ == '__main__':
    # replace input_file with your file path
    veins = detect_veins(input_image)
    plt.imshow(veins, cmap='gray')
    plt.show()
