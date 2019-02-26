import mxnet as mx
import apache_beam

from skimage.io import imsave

import mxnet.ndarray as nd

import unet

import os

class ImageReader:
    def __init__(self, imgsize, ctx):
        self.imgsize = imgsize
        self.ctx = ctx

    def preprocess(self, data):
        data = mx.nd.array(data).astype('float32').as_in_context(self.ctx)
        data = data / 255
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

class UNetInferenceFn(apache_beam.DoFn):

    def __init__(self, model, output):
        super(UNetInferenceFn, self).__init__()
        self.ctx = mx.cpu(0)
        self.net = unet.Unet()
        self.net.load_parameters(os.path.join(os.path.dirname(model), 'unet_RGB.params'), self.ctx, allow_missing=True)
        self.img_size = 256
        self.reader = ImageReader(self.img_size, self.ctx)
        self.output = output

    def process(self, element):
       """
       Returns clear images after filtering the cloudy ones
       :param element:
       :return:
       """
       batch = self.reader.load_batch(element)
       batch = batch.as_in_context(self.ctx)
       preds = nd.argmax(self.net(batch), axis=1)
       self.save_batch(element, preds)

    def save_batch(self, filenames, predictions):
        for idx, fn in enumerate(filenames):
            base, ext = os.path.splitext(os.path.basename(fn))
            mask_name = base + "_predicted_mask" + ext
            imsave(os.path.join(self.output, mask_name) , predictions[idx].asnumpy())
