# Retinal Vessel Detection
This repository contains three popular approaches for detecting Retinal Vessels:
1. Standard image manipulation and morphology
2. Classifiers
3. Deep Neural Network

Each solution has its own folder where you will find Jupyter Notebooks with descriptions of how each approach has been developed and `.html` files for people who cannot view
it with Jupyter.

## Short description
For those that do not want to spend time viewing this implementation here is a quick rundown on how each approach is made:
1. Image manipulation
* Read image
* Detect background (non retina part of the image), cut it out and save a mask
* Extract green channel (best contrast of vessels to rest of the image), later remove background with mask
* Equalize histogram of colors
* Apply Hessian filter, remove background again
* Apply bilateral filter
* Remove small noise
* Remove white circle around retina created by previous steps

2. Classifiers
* Cut out subimages randomly from image of specified size
* Get decision for middle pixel of subimage
* Calculate Hu moments of entire subimage fragment
* Merge Hu moments with pixel intensity
* Feed to KNN classifier
* Repeat on output image during prediction to get output image

3. Deep Neural Network
* Construct U-Net network
* Train network
* Use it for prediction

## Libraries used
* [matplotlib](https://pypi.org/project/matplotlib/)
* [OpenCV 2](https://pypi.org/project/opencv-python/)
* [numpy](https://pypi.org/project/numpy/)
* [tqdm](https://pypi.org/project/tqdm/)
* [skimage](https://pypi.org/project/scikit-image/)
* [sklearn](https://pypi.org/project/scikit-learn/)
* [pytorch](https://pypi.org/project/torch/)
* [Keras](https://pypi.org/project/keras/)
* [operator](https://pypi.org/project/pyoperators/)

## Results
You can see the results in the notebooks, but here is a quick preview of all approaches:

### Image manipulation
![alt text](https://github.com/Dawnkai/computed-tomography/blob/main/morphology.png)
### Classifiers
![alt text](https://github.com/Dawnkai/computed-tomography/blob/main/knn.png)
### Deep Neural Network
![alt text](https://github.com/Dawnkai/computed-tomography/blob/main/dnn.png)
