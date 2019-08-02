# Unsupervised_Edge_Connection

This is Pytorch code for edge connection with GAN,in which a autoencoder structure is utilized for generator

# Folder Structure

* DefectDataset
  * Single
   * gt
   * noise
  * Multi
   * gt
    * 00
    * 01
    * ...
   * noise
    * 00
    * 01
    * ...
  * background
* GAN_Image
* Model
    * DIS
    * GAN
* Model_Multi
    * model
* Template
    * bin_contour
* Test_Image
    * input
    * output
* src
  * AAE.py
  * AAEMultiModel.py
  * AAEWithClassifier.py
  * datasetGenerate.py
  * testAAE.py
  * testAAEWithClassifier.py
  * region.py
* README.md

# Usage

## 0. Install Library

`pytorch` <br>
`PIL` <br>
`cv2` <br>

## 1. Use AutoEncoder


```
## Generate  dataset

python3 datasetGenerate.py

## Train AEGAN
python3 AAE.py

## Test autoencoder
python3 testAAE.py
```
