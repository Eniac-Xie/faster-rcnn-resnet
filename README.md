# Faster-RCNN-ResNet

This code extends [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by adding ResNet implementation
and Online Hard Example Mining.


This is a ResNet Implementation for Faster-RCNN.
The faster rcnn code is based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).
The ohem code is based on [ohem](https://github.com/abhi2610/ohem).
To reduce the memory usage, we use batchnorm layer in [Microsoft's caffe](https://github.com/Microsoft/caffe)

# Modification
1. The [caffe-fast-rcnn](https://github.com/Eniac-Xie/caffe-fast-rcnn.git) we use is a little different from the one [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) use,
   it uses the batchnorm layer from [Microsoft's caffe](https://github.com/Microsoft/caffe) to reduce the memory usage.
2. Using the in-place eltwise sum within the [PR](https://github.com/BVLC/caffe/pull/3708)
3. To reduce the memory usage, we also release a pretrained ResNet-101 model in which batchnorm layer's parameters is
   merged into scale layer's, see tools/merge_bn_scale.py form more detail.
4. Use Online-Hard-Example-Mining while training.

# Installation
The usage is similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).

1. Clone this repository
  ```Shell
  git clone https://github.com/Eniac-Xie/faster-rcnn-resnet.git
  ```
  We'll call the directory that you cloned faster-rcnn-resnet `ROOT`

2. Clone the modified caffe-fast-rcnn

  ```Shell
  cd $ROOT/
  git clone https://github.com/Eniac-Xie/caffe-fast-rcnn.git
  ```

3. Build Cython module

  ```Shell
   cd $ROOT/lib/
   make
  ```

4. Build Caffe

  ```Shell
   cd $ROOT/caffe-fast-rcnn
   make all -j8
   make pycaffe
  ```
# Result

|                        | training data       | test data             |   ohem |    mAP@0.5    |
|------------------------|:-------------------:|:---------------------:|:------:|:-------------:|
|Faster-RCNN, ResNet-50  | VOC 07+12 trainval  | VOC 07 test           |  False |   78.78%      |           
|Faster-RCNN, ResNet-101 | VOC 07+12 trainval  | VOC 07 test           |  True  |   79.44%      |     


# Testing
Download faster-rcnn-resnet weights from:

[faster-rcnn-resnet without ohem (BaiduYun)](http://pan.baidu.com/s/1kUKXgVH)

[faster-rcnn-resnet without ohem (OneDrive)](https://1drv.ms/u/s!AgkRygoHQVTXigHNLWT6gRbTHo2f)

[faster-rcnn-resnet with ohem (BaiduYun)](http://pan.baidu.com/s/1o8CtJwI)

[faster-rcnn-resnet with ohem (OneDrive)](https://1drv.ms/u/s!AgkRygoHQVTXigInqoym2V6z4CNA)

then you can do as follow:

  ```Shell
   cd $ROOT/
   sh experiments/scripts/train_resnet101_bn_scale_merged_0712_end2end.sh
   make
  ```
or

  ```Shell
   cd $ROOT/
   sh experiments/scripts/train_resnet101_bn_scale_merged_0712_end2end_ohem.sh
   make
  ```

# Training
Download resnet-101 pretrained model, note that we use a modified version in which batchnorm layer's parameters is
merged into scale layer's, you can download the model from [Baidu Yun](http://pan.baidu.com/s/1qX7VFjA) or [OneDrive](https://1drv.ms/u/s!AgkRygoHQVTXigBCR-5cnmAkfGfy)

then you can do as follow:
  ```Shell
   cd $ROOT/
   sh experiments/scripts/train_resnet101_bn_scale_merged_0712_end2end.sh
  ```
or
  ```Shell
   cd $ROOT/
   sh experiments/scripts/train_resnet101_bn_scale_merged_0712_end2end_ohem.sh
  ```
