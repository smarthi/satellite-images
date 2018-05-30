# Data preparation
# To get data for the segmetation step:
# 
#  1. Download data from Sentinel-2 using the DataRequest/download_data.py script
#  2. Filter out cloudy images using the preprocessing/filtering.py script
#  3. Run this script, which:
#     3.1. Sets apart 20% of the data to save it for validation
#     3.2. Uses the Augmentor library for data augmentation
# 
# Augmentor: https://github.com/mdbloice/Augmentor
import os
import sys
import glob
import random
sys.path.append('../util')
import myaugmentor

IMG_DIR = os.path.join(os.path.dirname(__file__), '../data/tulips/bloom/filtered/')
ext = '.png'

# Set apart a fraction of the images, which will be used for validation. These images shouldnt be used as part of the base for the augmentations.
root = os.path.abspath(os.path.join(IMG_DIR, os.pardir))
val_dir = os.path.join(root, 'val')
os.makedirs(val_dir, exist_ok=True)

image_list = glob.glob(IMG_DIR + '*' + ext)
val = random.sample(image_list, int(len(image_list)*0.2))

for img in val:
    os.rename(img, os.path.join(val_dir, os.path.basename(img)))


p = myaugmentor.MyPipeline(IMG_DIR, output_directory=os.path.join(root, "train"))
p.ground_truth(os.path.join(root, "masks"))

# Define transformations to be applied to our images. 
# Details of the transformations here:
# https://github.com/mdbloice/Augmentor#main-features
p.skew(probability=0.5, magnitude=0.5)
p.shear(probability=0.3, max_shear_left=15, max_shear_right=15)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.rotate_random_90(probability=0.75)
p.rotate(probability=0.75, max_left_rotation=20, max_right_rotation=24)

# Number of images to generate
N = 20000
p.sample(N)
