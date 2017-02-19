import _init_paths
import numpy as np
import caffe

caffe.set_mode_cpu()

net = caffe.Net('data/CNN-models/resnet-caffe/ResNet-101-deploy.prototxt',
                'data/CNN-models/resnet-caffe/ResNet-101-model.caffemodel',
                caffe.TEST)

new_net = caffe.Net('models/pascal_voc/ResNet101_BN_SCALE_Merged/ResNet101_BN_SCALE_Merged_deploy.prototxt',
                'data/CNN-models/resnet-caffe/ResNet-101-model.caffemodel',
                caffe.TEST)

for layer_name in net.params.keys():
    if layer_name[:2] == 'bn':
        scale_layer_name = 'scale' + layer_name[2:]
        mu = net.params[layer_name][0].data
        var = net.params[layer_name][1].data
        gamma = net.params[scale_layer_name][0].data
        beta = net.params[scale_layer_name][1].data
        new_gamma = gamma / (np.power(var, 0.5) + 1e-5)
        new_beta = beta - gamma * mu / (np.power(var, 0.5) + 1e-5)

        new_net.params[scale_layer_name][0].data[...] = new_gamma
        new_net.params[scale_layer_name][1].data[...] = new_beta

new_net.save('data/imagenet_models/ResNet101_BN_SCALE_Merged_deploy.caffemodel')
