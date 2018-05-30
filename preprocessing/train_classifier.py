import os
import time

import pandas as pd
import numpy as np
import mxnet as mx

from mxnet import gluon
from mxnet.gluon.model_zoo import vision
from mxnet.image import color_normalize
from mxnet.gluon.data.vision import ImageRecordDataset

ctx = mx.gpu(0)

#-------------------------------------
# Paths
#-------------------------------------
root = os.path.dirname(__file__)
data_dir = os.path.join(root, '../data/cloud-classif/')
img_dir = os.path.join(data_dir, 'jpg/')
checkpoints_dir = os.path.join(root, 'checkpoints')


#-------------------------------------
# Data loading and augmentation
#-------------------------------------
mean = mx.nd.array([0.485, 0.456, 0.406], ctx=ctx).reshape((1,3,1,1))
std =  mx.nd.array([0.229, 0.224, 0.225], ctx=ctx).reshape((1,3,1,1))

train_augs = [
    mx.image.HorizontalFlipAug(0.5),
    mx.image.SaturationJitterAug(.1),
    mx.image.ContrastJitterAug(.1),
    mx.image.RandomCropAug((224, 224))
]

test_augs = [
    mx.image.CenterCropAug((224, 224))
]


def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = mx.nd.transpose(data, (2,0,1))
    return data, mx.nd.array([label]).asscalar().astype('float32')


#-------------------------------------
# Training utility functions
#-------------------------------------
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        data = color_normalize(data/255, mean, std)
        output = net(data)
        prediction = mx.nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]


def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])


def train_util(net, train_iter, val_iter, loss_fn, trainer, ctx, epochs, checkpoint_dir, init_epoch=0):
    '''
    Params:
    - net: network to train
    - train_iter: gluon.data.DataLoader with the training data
    - val_iter: "                      "  validation data
    - loss_fn: loss function to use for training
    - trainer: gluon.Trainer to use for training
    - ctx: context where we will operate (GPU or CPU)
    - epochs: number of epochs to train for
    - batch_size
    - checkpoint_dir: directory where checkpoints are saved every 100 batches
    - init_epoch: set to the initial epoch in case training is resumed from a previous execution'''
    batch_size = train_iter._batch_sampler._batch_size
    res = {'train':[],'val':[]}

    for epoch in range(1+init_epoch, epochs+init_epoch+1):
        metric = mx.metric.create(['acc'])
        for i, (data, label) in enumerate(train_iter):
            st = time.time()
            # ensure context            
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # normalize images
            data = color_normalize(data/255, mean, std)
            
            with mx.autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(data.shape[0], ignore_stale_grad=True)
            
            #  Keep a moving average of the losses
            metric.update([label], [output])
            names, accs = metric.get()
            if i%50 == 0:
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(epoch, i, batch_size/(time.time()-st), metric_str(names, accs)))
            if i%200 == 0:
                # Store accuracy and reset the metric
                metric.reset()
                if i != 0:
                    res['train'].append(accs)

        names, train_acc = metric.get()
        val_acc = evaluate_accuracy(val_iter, net)
        
        # Only save model params if results are better than the previous ones
        if res['val'] and val_acc > max(res['val']):
            net.save_params('%s/%d.params'%(checkpoint_dir, epoch))

        res['train'].append(train_acc)
        res['val'].append(val_acc)
        print("Epoch %s | training_acc %s | val_acc %s " % (epoch, train_acc, val_acc))
        
    return res


def train(net, data, ctx, epochs=10, learning_rate=0.01, checkpoint_dir='checkpoints', init_epoch=0):
    train_data = data['train']
    val_data = data['val']
    
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', 
            {'learning_rate': learning_rate})
    
    return train_util(net, train_data, val_data, loss, trainer, ctx, epochs,
                      checkpoint_dir, init_epoch=init_epoch)


#-------------------------------------
# Execution
#-------------------------------------
def main():
    # Create Datasets from rec files
    batch_size = 32

    train_bin = os.path.join(data_dir, 'train/clouds-binary.rec')
    valid_bin = os.path.join(data_dir, 'valid/clouds-binary.rec')
    test_bin  = os.path.join(data_dir, 'test/s2test-binary.rec')

    trainIterBin = ImageRecordDataset(
        filename=train_bin, 
        transform=lambda X, y: transform(X, y, train_augs)
    )

    validIterBin = ImageRecordDataset(
        filename=valid_bin, 
        transform=lambda X, y: transform(X, y, train_augs)
    )

    testIterBin = ImageRecordDataset(
        filename=test_bin,
        transform=lambda X, y: transform(X, y, test_augs)
    )

    train_data = gluon.data.DataLoader(trainIterBin, batch_size, shuffle=True)
    val_data   = gluon.data.DataLoader(validIterBin, batch_size)
    test_data  = gluon.data.DataLoader(testIterBin,  batch_size)

    data = {'train':train_data, 'val':val_data}

    # Create dir where we'll save the params of our model
    checkpoints = os.path.join(checkpoints_dir, 'resnet101')
    os.makedirs(checkpoints, exist_ok=True)

    # Load a pretrained network
    rn_pretrained = vision.resnet101_v2(pretrained=True) 

    # Load the network to train, using the same prefix to avoid problems when saving and loading params,
    # as we will assign the features  part of the pretrained network to this one
    rn101 = vision.resnet101_v2(classes=2, prefix=rn_pretrained.prefix)
    rn101.features = rn_pretrained.features
    rn101.output.initialize(mx.init.Xavier())

    rn101_acc = train(rn101, data, ctx, epochs=2, learning_rate=0.003,checkpoint_dir=checkpoints)

    # Save
    np.save(checkpoints + '/accuracy-results.npy', rn101_acc)

    # Find best scoring model 
    best = rn101_acc['val'].index(max(rn101_acc['val'])) + 1
    best_params = os.path.join(checkpoints, '{}.params'.format(best))

    # Test it on the test dataset
    rn101.load_params(best_params, ctx)
    test_acc = evaluate_accuracy(test_data, rn101)
    print('Best model on validation set saved in: {}'.format(best_params))
    print('Accuracy on test set = {}'.format(test_acc))


if __name__ == "__main__":
    main()


        