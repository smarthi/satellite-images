import os
import glob
import math

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imsave, imread
from shutil import copyfile

import mxnet as mx
import mxnet.ndarray as nd

from mxnet import gluon
from mxnet.image import color_normalize


ctx = mx.gpu(0)
mean = mx.nd.array([0.485, 0.456, 0.406], ctx=ctx).reshape((3,1,1))
std =  mx.nd.array([0.229, 0.224, 0.225], ctx=ctx).reshape((3,1,1))
augs = [ 
     mx.image.CenterCropAug((224, 224))
]


def preprocess(data, augs):
    data = mx.nd.array(data).astype('float32').as_in_context(ctx)
    for aug in augs:
        data = aug(data)
    data = mx.nd.transpose(data, (2,0,1))
    data = color_normalize(data/255, mean, std)
    return data


def read_img(filename):
    img = mx.image.imread(filename)
    return preprocess(img, augs)


def load_batch(filenames):
    batch = mx.nd.empty((len(filenames),3,224,224))
    for idx,fn in enumerate(filenames):
        batch[idx] = read_img(fn)
    
    return batch


def find_clear(root, batch_size, net, ext='.png'):
    print("Scanning dir {}".format(root))
    clear = []
    files = glob.glob(root + '*wms*' + ext)
    print("Found {} images".format(len(files)))
    nbatches = math.ceil(len(files)/batch_size)
    
    for n in range(nbatches):
        files_batch = files[n*batch_size:(n+1)*batch_size]
        batch = load_batch(files_batch)
        batch = batch.as_in_context(ctx)
        preds = mx.nd.argmax(net(batch), axis=1)
        idxs = np.arange(len(files_batch))[preds.asnumpy() == 0]
        clear.extend([files_batch[i] for i in idxs])

    print('Found {} clear images'.format(len(clear)))
    
    return clear


def main():
    IMG_DIRS = ['../data/tulips/bloom/16/img/']
                #, '../data/tulips/bloom/17/img/']
    CHECKPOINTS_DIR = 'checkpoints'

    # Load the pretrained network and use the saved weights 
    net = gluon.model_zoo.vision.resnet101_v2(classes=2, ctx=ctx)
    net.load_params(os.path.join(CHECKPOINTS_DIR, 'resnet100-bin', '43-0.params'), ctx)

    batch_size = 64
    clear = []
    
    for folder in IMG_DIRS:
        clear.extend(find_clear(folder, batch_size, net))

    # Move clear images to new directory
    newdir = '../data/tulips/bloom/filtered'
    os.makedirs(newdir, exist_ok=True)

    for file in clear:
        copyfile(file, os.path.join(newdir, os.path.basename(file)))


if __name__ == "__main__":
    main()
















