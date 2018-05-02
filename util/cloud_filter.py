"""
Naive filter to detect and discard images wich are mostly clouds.
"""
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import os


def filter_dir(imgdir, labeldir, threshold, whiteness, percent):
    """
    Searches the given dir and moves all images with either 
        a) average white level above the threshold
        b) more than x percent of very bright pixels
    to a subfolder.

    Params:
    imgdir: dir to search
    discard: subfolder to move the discarded imgs
    threshold: avg grey level above which images are discarded (~120-140 has proven to work reasonably well)
    whiteness: level above which a pixel is considered to be part of a cloud
    percent: (0, 1.0) 
    """
    # Subdir where the cloudy imgs are moved to
    newdir = os.path.join(imgdir, labeldir)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    images = glob.glob(os.path.join(imgdir, '*sat*'))
    filtered = 0
    
    avgs   = []
    bright = []

    # Find all satellite images in the images dir and discard those in which
    # white is the dominant color (they are most likely clouds)
    for fn in images:
        img = Image.open(fn).convert('L')
        mean = np.mean(np.square(np.asmatrix(img).astype(float)))

        if labeldir == 'clear/':
            bright_px = np.sum(np.array(img) >= whiteness)
            w,h = img.size
            if bright_px < w*h*percent:
                os.rename(fn, os.path.join(newdir, os.path.basename(fn)))
                filtered += 1    

        elif mean > threshold:
            os.rename(fn, os.path.join(newdir, os.path.basename(fn)))
            filtered += 1
        else:
            bright_px = np.sum(np.array(img) >= whiteness)
            w,h = img.size
            if bright_px > w*h*percent:
                os.rename(fn, os.path.join(newdir, os.path.basename(fn)))
                filtered += 1
    
    print("{} imgs found: {}. Filtered: {}".format(labeldir, len(images), filtered))


if __name__ == "__main__":
    imgdir = "../data/cloud-classif/s2test/" 
    whiteness = 220

    labeldir = "cloudy/"
    threshold = 42000
    percent = 0.8
    filter_dir(imgdir, labeldir, threshold, whiteness, percent)

    labeldir = "partly_cloudy/"
    threshold = 20000
    percent = 0.35
    filter_dir(imgdir, labeldir, threshold, whiteness, percent)

    labeldir = "clear/"
    percent = 0.02
    filter_dir(imgdir, labeldir, threshold, whiteness, percent)


