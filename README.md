# Compact-Bilinear-Pooling
This is a re-implementation of the 2016 cvpr paper "Compact Bilinear Pooling" for fine-grained image classification using Pytorch  
The repo itself does not include any raw data. Save your images according to the heirarchical structure as below:

Level 0: class1, class2, ..., classC  
Level 1: 1.jpg, 2.jpg, ..., N.jpg

Namely, there are N images in each folder "class#" (the number of images from each class can be different)


**compact_bilinear_cnn_fc.py**: training the network without finetuning the convolutional layers in the pre-trained VGG 16 model.
**compact_bilinear_cnn_all.py**:training the whole network include the convolutional layers in the pre-train VGG 16 model.

In order to obtain the same accuracy as claimed in the paper, the readers should first train the network using **compact_bilinear_cnn_fc.py** with large learning rate, save the results. Then, use the results as the initial values for the network, and train the whole network using **compact_bilinear_cnn_all** with small learning rate. The default leanring rates are given in the files.

The work is constructed based on the repo: https://github.com/HaoMood/bilinear-cnn
