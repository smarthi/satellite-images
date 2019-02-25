import mxnet as mx
import apache_beam

from mxnet import gluon
from mxnet.image import color_normalize

import numpy as np

import os

class FilterCloudyFn(apache_beam.DoFn):

    def __init__(self, model):
        super(FilterCloudyFn, self).__init__()
        self.ctx = mx.cpu(0)
        self.mean = mx.nd.array([0.485, 0.456, 0.406], ctx=self.ctx).reshape((3, 1, 1))
        self.std = mx.nd.array([0.229, 0.224, 0.225], ctx=self.ctx).reshape((3, 1, 1))
        self.augs = [
            mx.image.CenterCropAug((224, 224))
        ]

        self.net = gluon.model_zoo.vision.resnet50_v2(classes=2, ctx=self.ctx)
        self.net.load_parameters(os.path.join(os.path.abspath(model), 'resnet50_finetuned.params'),
                             ctx=self.ctx, allow_missing=True)


    def preprocess(self, img, augs):
        data = mx.nd.array(img).astype('float32').as_in_context(self.ctx)
        for aug in augs:
            data = aug(data)
        data = mx.nd.transpose(data, (2, 0, 1))
        data = color_normalize(data / 255, self.mean, self.std)
        return data

    def read_img(self, filename):
        img = mx.image.imread(filename)
        return self.preprocess(img, self.augs)

    def load_batch(self, filenames):
        batch = mx.nd.empty((len(filenames), 3, 224, 224))
        for idx, fn in enumerate(filenames):
            batch[idx] = self.read_img(fn)
        return batch

    def process(self, element):
       """
       Returns clear images after filtering the cloudy ones
       :param element:
       :return:
       """
       clear_images = []
       batch = self.load_batch(element)
       batch = batch.as_in_context(self.ctx)
       preds = mx.nd.argmax(self.net(batch), axis=1)
       idxs = np.arange(len(element))[preds.asnumpy() == 0]
       clear_images.extend([element[i] for i in idxs])
       yield clear_images









