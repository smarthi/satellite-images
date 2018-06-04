# ## Satellite image segmentation

import os
import time
import collections
import unet

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon

from mxnet.gluon.data import Dataset
from mxnet.gluon.loss import Loss
from datetime import datetime
from shutil import copyfile


#-------------------------------------
# Data loading
#-------------------------------------
class ImageWithMaskDataset(Dataset):
    """
    A dataset for loading images (with masks).
    Based on: mxnet.incubator.apache.org/tutorials/python/data_augmentation_with_masks.html
    
    Parameters
    ----------
    root : str
        Path to root directory.
    imgdir: str
        Path to folder containing the images, relative to root
    maskdir: str 
        Path to folder containing the masks/ground truth, relative to root
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::
        transform = lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, imgdir, maskdir, transform=None):
        self._imgdir = os.path.expanduser(imgdir)
        self._maskdir = os.path.expanduser(maskdir)
        self._transform = transform
        self.imgdir = imgdir
        self._exts = ['.png']
        self._geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905'}
        self._list_images(self._imgdir)

    def _list_images(self, root):
        images = collections.defaultdict(dict)
        for filename in sorted(os.listdir(root)):
            name, ext = os.path.splitext(filename)
            mask_flag = "geopedia" in name
            if ext.lower() not in self._exts:
                continue
            if not mask_flag:
                patch_id = filename.split('_')[1]
                year = datetime.strptime(filename.split('_')[3], "%Y%m%d-%H%M%S").year
                mask_fn = 'tulip_{}_geopedia_{}.png'.format(patch_id, self._geopedia_layers['tulip_field_{}'.format(year)])
                images[name]["base"] = filename
                images[name]["mask"] = mask_fn
        self._image_list = list(images.values())

    def __getitem__(self, idx):
        assert 'base' in self._image_list[idx], "Couldn't find base image for: " + image_list[idx]["mask"]
        base_filepath = os.path.join(self._imgdir, self._image_list[idx]["base"])
        base = mx.image.imread(base_filepath)
        assert 'mask' in self._image_list[idx], "Couldn't find mask image for: " + image_list[idx]["base"]
        mask_filepath = os.path.join(self._maskdir, self._image_list[idx]["mask"])
        mask = mx.image.imread(mask_filepath, flag=0)
        if self._transform is not None:
            return self._transform(base, mask)
        else:
            return base, mask

    def __len__(self):
        return len(self._image_list)


def transform(base, mask):
    ### Convert types
    base = base.astype('float32')/255
    mask = mask.astype('float32')/255
    
    # Convert mask to binary
    mask = (mask > 0.4).astype('float32')
    
    # Reshape the tensors so the order is now (channels, w, h)
    base = mx.nd.transpose(base, (2,0,1))
    mask = mx.nd.transpose(mask, (2,0,1))
    
    return base, mask


#-------------------------------------
# Loss
#-------------------------------------
class DiceCoeffLoss(Loss):
    """
    Soft dice coefficient loss.
    Based on https://github.com/Lasagne/Recipes/issues/99
    Input:
       pred: (batch size, c, w, h) network output, must sum to 1 over c channel (such as after softmax)
       label:(batch size, c, w, h) one hot encoding of ground truth
       eps; smoothing factor to avoid division by zero
    :param eps: 
    Output:
        Loss tensor with shape (batch size) 
    """

    def __init__(self, eps=1e-7, _weight = None, _batch_axis= 0, **kwards):
        Loss.__init__(self, weight=_weight, batch_axis=_batch_axis, **kwards)
        self.eps = eps

    def hybrid_forward(self, F, label, pred):  
        # One-hot encode the label
        label = nd.concatenate([label != 1, label], axis=1)
        
        axes = tuple(range(2, len(pred.shape)))
        intersect = nd.sum(pred * label, axis=axes)
        denom = nd.sum(pred + label, axis=axes)
        return - (2. * intersect / (denom + self.eps)).mean(axis=1)


#-------------------------------------
# Evaluation metric
#-------------------------------------
class IouMetric(mx.metric.EvalMetric):
    """Stores a moving average of the intersection over union metric"""
    def __init__(self, axis=[2,3], smooth=1e-7):
        super(IouMetric, self).__init__('IoU')
        self.name = 'IoU'
        self.axis = axis
        self.smooth = smooth
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, label, pred):
        """
        Implementation of updating metrics
        """
        i = nd.sum((pred==1)*(label==1), axis=self.axis)
        u = nd.sum(pred, axis=self.axis) + nd.sum(label, axis=self.axis) - i
        iou = (i + self.smooth) / (u + self.smooth)
        self.sum_metric += nd.sum(iou, axis=0)
        self.num_inst += pred.shape[0]
        
    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        value = (self.sum_metric / self.num_inst).asscalar() if self.num_inst != 0 else float('nan')
        return (self.name, value)


def metric_str(names, vals):
    return '{}={}'.format(names, vals)


#-------------------------------------
# Training
#-------------------------------------
def evaluate(data_iterator, net):
    metric = IouMetric()
    for data, label in data_iterator:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # data = color_normalize(data/255, mean, std)
        output = net(data)
        pred = nd.reshape(np.argmax(output, axis=1), (0, 1, img_width, img_height))
        # prediction = mx.nd.argmax(output, axis=1)
        metric.update(label, pred)
    return metric.get()[1]


def train_util(net, train_iter, val_iter, loss_fn,
               trainer, ctx, epochs, checkpoint_dir, init_epoch=0):
    '''
    Function to train the neural network.
    
    PARAMS:
    - net: network to train
    - train_iter: gluon.data.DataLoader with the training data
    - validation_iter: "                      "  validation data
    - loss_fn: loss function to use for training
    - trainer: gluon.Trainer to use for training
    - ctx: context where we will operate (GPU or CPU)
    - epochs: number of epochs to train for
    - checkpoint_dir: directory where checkpoints are saved every 100 batches
    - init_epoch: set to the initial epoch in case training is resumed from a previous execution
    '''
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_size = train_iter._batch_sampler._batch_size
    res = {'train':[],'val':[]}

    for epoch in range(1 + init_epoch, epochs + init_epoch+1):
        metric = IouMetric()
        for i, (data, label) in enumerate(train_iter):
            st = time.time()
            # Ensure context            
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Normalize images?
            # data = color_normalize(data/255, mean, std)
            
            with mx.autograd.record():
                output = net(data)
                loss = loss_fn(label, output)
                pred = nd.reshape(np.argmax(output, axis=1), (0, 1, img_width, img_height))
                
            loss.backward()
            trainer.step(data.shape[0], ignore_stale_grad=True)
            
            #  Keep a moving average of the losses
            metric.update(label, pred)
            names, vals = metric.get()
            if i%25 == 0:
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(epoch, i, batch_size/(time.time()-st), metric_str(names, vals)))
            if i!=0 and i%500 == 0:
                # Every 500 batches, save params and evaluate performance on the val set
                names, train_acc = metric.get()
                metric.reset()
                val_acc = evaluate(val_iter, net)

                if res['val'] and val_acc > max(res['val']):      
                    net.save_params('%s/%d-%d.params'%(checkpoint_dir, epoch, i))
                
                res['train'].append(train_acc)
                res['val'].append(val_acc)
                print("Epoch %s Batch %d| train IoU: %s | val IoU: %s " % (epoch, i, train_acc, val_acc))
        
        # Only save model params if results are better than the previous ones
        if res['val'] and val_acc > max(res['val']):      
            net.save_params('%s/%d-%d.params'%(checkpoint_dir, epoch, 0))
        
        names, train_acc = metric.get()
        val_acc = evaluate(val_iter, net)
        res['train'].append(train_acc)
        res['val'].append(val_acc)
        print("Epoch %s | train IoU %s | val IoU %s " % (epoch, train_acc, val_acc))
        metric.reset()
        
    return res


#-------------------------------------
# Execution
#-------------------------------------
ctx = mx.gpu(0)

batch_size = 8
img_width  = 256
img_height = 256

root = os.path.dirname(__file__)
train_dir = os.path.join(root, '../data/tulips/bloom/train/')
val_dir   = os.path.join(root, '../data/tulips/bloom/val/')
mask_dir  = os.path.join(root, '../data/tulips/bloom/masks/')

# Create train and validation DataLoaders from our Datasets
train_ds = ImageWithMaskDataset(train_dir, train_dir, transform=transform)
train_iter = gluon.data.DataLoader(train_ds, batch_size)

val_ds  = ImageWithMaskDataset(val_dir, mask_dir, transform=transform)
val_iter= gluon.data.DataLoader(val_ds, batch_size)

# Instantiate a U-Net and train it
net = unet.Unet()
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
net.hybridize()
loss = DiceCoeffLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', 
            {'learning_rate': 1e-4, 'beta1':0.9, 'beta2':0.99})

epochs = 50
checkpoint_dir = os.path.join(root, 'checkpoints', 'unet')

results = train_util(net, train_iter, val_iter, loss, trainer, ctx, epochs, checkpoint_dir)
np.save(checkpoint_dir + '/results.npy', results)

# Find best scoring model 
best = results['val'].index(max(results['val'])) + 1
epoch, batch = divmod(best, 5)
epoch += 1
batch = ((batch + 1)%5)*500
best_params = os.path.join(checkpoint_dir, '{}-{}.params'.format(epoch, batch))

# Copy it to <checkpoints>/best_unet.params
save_filename = os.path.join(checkpoint_dir, 'best_unet.params')
copyfile(best_params, save_filename)
print('Best model on validation set: {}, saved in: {}'.format(best_params, save_filename))





