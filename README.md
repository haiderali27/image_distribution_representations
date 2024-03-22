# image_distribution_representations
This repository contains code, that describes the distribution of images with different metrics and their feature representation on latent space


## About
- [Details](#Details)
- [Run](#Run-The-Code)

## Details
`Image Distributions is created using SSIM,  MSE, and  Wasserstein distance metrics. To represent the same dataset image distribution,  the distance among images in the dataset with each metric is calculated. To represent Cross Dataset Distribution (one dataset to another dataset), the distance of all images in one dataset to the distance of all images in the other dataset is calculated. X-axis represents the distance, Y-axis represents the count of images. `
`Since MvTec Dataset Structure is used. We have 3 folders for each class, train, test, ground_truth. For our purpose, We will use train and test only. Since those images contain the actual images. Our sole purpose is to show how they are distributed concerning metrics(SSIM, MSE, Wasserstein distance). The distribution of the train/good, test/good, and all the sub-folders in the test are measured and also merged distribution of all sub-folders in the test. For cross-distribution, train/good vs test/all-merged comparison is used`
`To represent latent space, Principal Component Analysis, and t-distributed Stochastic Neighbor Embedding are used. `
`Images of train/good, test/all-merged and (train/good + test/all-merged) are represented in 2D and 3D latent representations`


## Run The Code
1. `git clone https://github.com/haiderali27/image_distribution_representations.git` or `git clone git@github.com:haiderali27/image_distribution_representations.git`
2. Run the requirement File `pip install -r requirements.txt`
3. You can run each Python file to recreate the image representation. (Representations are available in the data folder)
   
