from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

class Binary_MNIST_DS(Dataset):
    """ Binary MNIST Dataset.
    """

    def __init__(self, path_file, transform=None):

        self.transform = transform

        #self.train = train  # training set or test set

        if not os.path.exists(path_file):
            raise RuntimeError('Dataset not found @ {}!'.format(self.path_file))

        self.data = np.loadtxt(path_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        img = img.reshape((28, 28)).astype('uint8')*255

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)




#dataset = Binary_MNIST_DS('data/binarized_mnist_train.amat', transform=None)
#plt.imshow(dataset[0])
#plt.show()

'''
data = np.loadtxt('data/pr_mean.csv', delimiter=',')
plt.plot(data[:,1], data[:,2], label='PR Mean')
data = np.loadtxt('data/rt_mean.csv', delimiter=',')
plt.plot(data[:,1], data[:,2], label='RT Mean')
data = np.loadtxt('data/rr_stdev.csv', delimiter=',')
plt.plot(data[:,1], data[:,2], label='RR Std Dev.')
data = np.loadtxt('data/user_id.csv', delimiter=',')
plt.plot(data[:,1], data[:,2], label='User ID')

plt.ylabel('Metric value')
plt.xlabel('Epochs')

plt.legend(loc='lowe right')

plt.show()
'''