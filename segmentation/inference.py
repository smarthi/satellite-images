# Satellite image segmentation
# U-Net inference

import os
import unet
import argparse
import glob
import math

import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn

from mxnet.gluon.data import Dataset, DataLoader
from mxnet.gluon.loss import Loss
from mxnet import image
from skimage.io import imsave, imread
from datetime import datetime

#-------------------------------------
# Execution
#-------------------------------------
class ImageReader:
    def __init__(self, imgsize, ctx):
        self.imgsize = imgsize
        self.ctx = ctx

    def preprocess(self, data):
        data = mx.nd.array(data).astype('float32').as_in_context(self.ctx)
        data = mx.nd.transpose(data, (2,0,1))
        return data


    def read_img(self, filename):
        img = mx.image.imread(filename)
        return self.preprocess(img)


    def load_batch(self, filenames):
        batch = mx.nd.empty((len(filenames),3,self.imgsize,self.imgsize))
        for idx,fn in enumerate(filenames):
            batch[idx] = self.read_img(fn)
        return batch


def save_batch(filenames, predictions):
    for idx, fn in enumerate(filenames):
        base, ext = os.path.splitext(os.path.basename(fn))
        mask_name = base + "_predicted_mask" + ext
        imsave(os.path.join(os.path.dirname(fn), mask_name) , predictions[idx].asnumpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", 
        help="Directory containing the image files (.png) we'll run inference on. Relative to the root of the project (tulip-fields/)")
    args = parser.parse_args()

    ctx = mx.gpu(0)

    batch_size = 8
    img_size  = 256

    root = os.path.dirname(__file__)
    imgdir = os.path.join(root, os.pardir, args.dir)
    checkpoint_dir = os.path.join(root, 'checkpoints', 'unet')

    # Instantiate a U-Net and train it
    net = unet.Unet()
    net.load_params(os.path.join(checkpoint_dir, 'best_unet.params'), ctx)

    print("Scanning dir {}".format(imgdir))
    files = glob.glob(os.path.join(imgdir, '*wms*.png'))
    print("Found {} images".format(len(files)))
    nbatches = math.ceil(len(files)/batch_size)
    
    reader = ImageReader(img_size, ctx)

    for n in range(nbatches):
        files_batch = files[n*batch_size:(n+1)*batch_size]
        batch = reader.load_batch(files_batch)
        batch = batch.as_in_context(ctx)
        preds = nd.argmax(net(batch), axis=1)
        save_batch(files_batch, preds)
        

if __name__ == "__main__":
    main()





