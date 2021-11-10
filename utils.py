import torch
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
import pdb

def categorical_dice(prediction, truth, k):
    # Dice overlap metric for label value k
    A = (np.argmax(prediction, axis=1) == k)
    B = (np.argmax(truth, axis=1) == k)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B)+1e-6)

def convert_to_1hot_3D(label, n_class):
    # Convert a label map (N x 1 x D x H x W) into a one-hot representation (N x C x D x H x W)
    label_swap = label.swapaxes(1, 4)
    label_flat = label_swap.flatten()
    n_data = len(label_flat)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label_swap.shape[0], label_swap.shape[1], label_swap.shape[2], label_swap.shape[3], n_class))
    label_1hot = label_1hot.swapaxes(1, 4)
    return label_1hot

def huber_loss_3d(x):
    bsize, csize, depth, height, width = x.size()
    d_x = torch.index_select(x, 4, torch.arange(1, width).cuda()) - torch.index_select(x, 4, torch.arange(width-1).cuda())
    d_y = torch.index_select(x, 3, torch.arange(1, height).cuda()) - torch.index_select(x, 3, torch.arange(height-1).cuda())
    d_z = torch.index_select(x, 2, torch.arange(1, depth).cuda()) - torch.index_select(x, 2, torch.arange(depth-1).cuda())
    err = torch.sum(torch.mul(d_x, d_x))/width + torch.sum(torch.mul(d_y, d_y))/height + torch.sum(torch.mul(d_z, d_z))/depth
    err /= bsize
    tv_err = torch.sqrt(0.01+err)
    return tv_err

