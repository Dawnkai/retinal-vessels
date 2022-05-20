# Retinal Vessel Detection
This repository contains three popular approaches for detecting Retinal Vessels:
1. Standard image manipulation and morphology
2. Classifiers
3. Deep Neural Network

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

The following measures were calculated for all approaches based on their results for images in `images` folder:
* Accuracy : (`TP` + `TN`) / (`TP` + `TN` + `FP` + `FN`) 
* Sensitivity : `TP` / (`TP` + `FN`) 
* Specificity : `TN` / (`TN` + `FP`) 
* Balanced Accuracy : (Sensitivity + Specificity) / 2 

Where:
`TP` - True Positive
`TN` - True Negative
`FP` - False Positive
`FN` - False Negative

All calculated by comparing every pixel of input and output image.

Below the metrics you can see example image result (left is expected result, right is actual result).

### Image manipulation
| Image name | Accuracy | Sensitivity | Specificity | Balanced Accuracy |
|:----------:|:--------:|:-----------:|:-----------:|:-----------------:|
|  11_dr.jpg |    93%   |     53%     |     98%     |        75%        |
|  11_g.jpg  |    93%   |     53%     |     98%     |        76%        |
|  11_h.jpg  |    94%   |     55%     |     98%     |        77%        |
|  12_dr.JPG |    93%   |     48%     |     98%     |        73%        |
|  12_g.jpg  |    93%   |     59%     |     97%     |        78%        |

![alt text](https://github.com/Dawnkai/retinal-vessels/blob/master/morphology.png)
### Classifiers
| Image name | Accuracy | Sensitivity | Specificity | Balanced Accuracy |
|:----------:|:--------:|:-----------:|:-----------:|:-----------------:|
|  11_dr.jpg |    94%   |     68%     |     96%     |        82%        |
|  11_g.jpg  |    94%   |     71%     |     95%     |        83%        |
|  11_h.jpg  |    94%   |     75%     |     95%     |        85%        |
|  12_dr.JPG |    94%   |     65%     |     95%     |        80%        |
|  12_g.jpg  |    93%   |     76%     |     94%     |        85%        |

![alt text](https://github.com/Dawnkai/retinal-vessels/blob/master/knn.png)
### Deep Neural Network
| Image name | Accuracy | Sensitivity | Specificity | Balanced Accuracy |
|:----------:|:--------:|:-----------:|:-----------:|:-----------------:|
|  11_dr.jpg |    95%   |     86%     |     96%     |        91%        |
|  11_g.jpg  |    96%   |     85%     |     97%     |        91%        |
|  11_h.jpg  |    96%   |     86%     |     97%     |        92%        |
|  12_dr.JPG |    96%   |     71%     |     98%     |        85%        |
|  12_g.jpg  |    96%   |     85%     |     97%     |        91%        |
![alt text](https://github.com/Dawnkai/retinal-vessels/blob/master/dnn.png)
