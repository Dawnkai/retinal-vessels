import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from random import randint, seed
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math


# Images used to create classifiers
TRAIN_DIR = 'dataset/train/images'
# Images used to test the classifier
TEST_DIR = 'dataset/test/images'
# Ground truth images for classifier creation
TRAIN_MASKS_DIR = 'dataset/train/ground_truth'
# Ground truth images for classifier testing
TEST_MASKS_DIR = 'dataset/test/ground_truth'
# All images will be resized to the size below
img_size = (1024, 1024)
# Subimage fragment size extracted to create classifier
sample_size = 5

def get_fragment(img, x, y, size):
    '''
    Cut subimage so that (x,y) are the coordinates of central pixel.
    Image will be expanded to allow extraction of border pixels.
    :param img: - input image
    :param x: - OX coordinate of middle pixel
    :param y: - OY coordinate of middle pixel
    :param size: - size of the fragment
    '''
    expanded_image = np.pad(img, [(size // 2, size // 2), (size // 2, size // 2)], 'constant', constant_values=(0,0))
    return expanded_image[x:x+size, y:y+size]


def join_tables(tables):
    '''
    Flattening arrays. Example:
    [[0, 1, 2],
     [1, 3, 4]]
    will return [0, 1, 2, 1, 3, 4].
    :param tables: - arrays to flatten
    '''
    result = []
    for table in tables:
        result.extend(table)
    return result


def get_label(img, x, y):
    '''
    Get prediction for pixel of coordinates x i y.
    Positive for white pixels in ground truth image, negative otherwise.
    :param img: - input image
    :param x: - OX coordinate of middle pixel
    :param y: - OY coordinate of middle pixel
    '''
    return 1 if img[x][y] == 255 else 0


def get_moments(sample):
    '''
    Calculate Hu moments for subimage
    :param sample: - subimage
    '''
    moments = cv2.moments(sample)
    huMoments = cv2.HuMoments(moments)
    # Normalize Hu moments using log for better results
    for idx, moment in enumerate(huMoments):
        if moment[0] == 0:
            huMoments[idx][0] = 0
        else:
            huMoments[idx][0] = -1 * math.copysign(1.0, moment[0]) * math.log10(abs(moment[0]))
    return [ elem[0] for elem in huMoments ]


def get_merged_prediction(img, model):
    '''
    Get prediction for subimage.
    :param img: - input subimage
    :param model: - classifier
    '''
    moments = get_moments(img)
    return model.predict([moments + join_tables(img)])[0]


def get_merged_samples(img, mask, sample_size, maxSamples = 200, guaranteedVeins = 50):
    '''
    Split image into subimages, calculate their Hu moments, merge them with
    pixel intensities and finally get predictions for those subimages.
    :param img: - input image
    :param mask: - ground truth image
    :param sample_size: - subimage size
    :param maxSamples: - max samples extracted for subimage
    :param guaranteedVeins: - minimum amount of positive samples
    '''
    result = [[], []]
    seed(42) # Random seeding
    veinsSoFar = 0
    samples = 0
    
    while (samples < maxSamples):
        # Get random pixel coordinates
        x = randint(0, img.shape[0] - 1)
        y = randint(0, img.shape[1] - 1)
        # Extract subimage with central pixel in those coordinates
        label = get_label(mask, x, y)
        # Guarantee that the amount of positive samples is higher than
        # guaranteedVeins variable
        if label == 1 and veinsSoFar < guaranteedVeins:
            veinsSoFar += 1
            fragment = get_fragment(img, x, y, sample_size)
            # Get Hu moments and image intensities
            result[0].append(get_moments(fragment) + join_tables(fragment))
            # Get prediction for this pixel
            result[1].append(label)
            samples += 1
        # After fulfilling the minimum positive samples limit
        elif veinsSoFar >= guaranteedVeins:
            fragment = get_fragment(img, x, y, sample_size)
            # Save Hu moments and image intensities
            result[0].append(get_moments(fragment) + join_tables(fragment))
            # Get prediction for this pixel
            result[1].append(label)
            samples += 1

    return result


def get_merged_model(fragment_size=sample_size, n_neighbors=10, truths=75, num_samples=200, img_size = (1024, 1024)):
    train_data = []
    train_labels = []
    data = []

    num_images = len(listdir(TRAIN_DIR))
    for image in listdir(TRAIN_DIR):
        # Read image
        img = cv2.imread(f'{TRAIN_DIR}/{image}', cv2.IMREAD_COLOR)
        # Resize image
        img = cv2.resize(img, img_size)
        # Extract green channel of the image
        _, img, _ = cv2.split(img)
        # Equalize color histograms for better contrast
        img = cv2.equalizeHist(img)

        # Read ground truth image
        mask = cv2.imread(f'{TRAIN_MASKS_DIR}/{image.split(".")[0] + ".tif"}', cv2.IMREAD_GRAYSCALE)
        # Resize ground truth image
        mask = cv2.resize(mask, img_size)

        # Extract data for classifier
        data = get_merged_samples(img, mask, fragment_size, num_samples, truths)

        train_data += data[0]
        train_labels += data[1]

    # Dataset information
    num = 0
    for label in train_labels:
        if label == 1:
            num += 1
    print(f"Number of positive samples: {num}")
    print(f"Number of all samples: {len(train_labels)}")

    # Create KNN classifier
    merged_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Fit the classifier with training data
    merged_model.fit(train_data, train_labels)
    
    return merged_model
    

def get_merged_predictions(filepath, fragment_size=sample_size, n_neighbors=10, truths=75, img_size = (1024, 1024)):
    '''
    Get image of prediction for entire image using KNN classifier.
    :param filepath: - input image filepath
    :param fragment_size: - size of subimage fragments
    :param n_neighbors: - number of neighbors in KNN classifier
    :param truths: - minimum amount of positive samples in classifier
    :param img_size: - input image size
    '''
    # Read image
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # Resize image for faster computation
    img = cv2.resize(img, img_size)
    # Extract green channel from image
    _, g_img, _ = cv2.split(img)
    # Equalize histogram of colors
    g_img = cv2.equalizeHist(g_img)
    result = np.zeros((g_img.shape[0], g_img.shape[1]))
    # Create KNN classifier
    model = get_merged_model(fragment_size, n_neighbors, truths, 100, (img.shape[0], img.shape[1]))
    # Get predictions for image
    for x, row in enumerate(g_img):
        for y, _ in enumerate(row):
            result[x][y] = get_merged_prediction(get_fragment(g_img, x, y, fragment_size), model)
    return result


if __name__ == '__main__':
    # replace input_file with your file path
    result = get_merged_predictions(input_file)
    plt.imshow(result, cmap='gray')
    plt.show()
    