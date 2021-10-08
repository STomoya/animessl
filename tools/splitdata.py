'''make filelist as required by vissl "disk_filelist" data source'''
import random, glob
import numpy as np

images = sorted(glob.glob('/home/vissluser/workspace/data/danbooru/*/*', recursive=True))
random.seed(0)
random.shuffle(images)
TRAIN_SIZE = 0.9

num_train_split = int(len(images) * TRAIN_SIZE)
train_split = images[:num_train_split]
test_split  = images[num_train_split:]

np.save('train_filelist.npy', train_split)
np.save('test_filelist.npy', test_split)
